"""Score extraction against hand-authored page fixtures.

Each directory under ``fixtures/`` contains ``page.html``, ``expected.json``,
and optionally ``screenshot.png`` and ``url.txt``. The harness derives the
same flattened plaintext used in production, extracts document metadata, runs
the real model call, flattens its nested result, and exact-compares it with the
answer key.

Every expected holder states the complete flat schema. A holding is keyed by
person, title, organisation, start date, and end date so distinct terms of the
same office remain separate. The remaining person and position fields are
compared on matched holdings. Evidence quotes remain part of production output
but are deliberately outside evaluation scope for now. Duplicate observations
are reported rather than silently collapsed, and empty agreement scores as
perfect.

The harness spends model tokens and reads ``.env``, but does not use Pravda or
the database.

Usage:

    uv run python evaluate.py
    uv run python evaluate.py -v
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
from bs4 import BeautifulSoup
from openai import OpenAI

from kolkhoz.config import load_config
from kolkhoz.extract import extract, flatten_persons, metadata_from_html

log = logging.getLogger("evaluate")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The flat extraction/storage schema. Organisation and term dates join person
# and title in the observation key so repeated terms of the same office remain
# distinct. Remaining fields are exact-compared on matched observations.
HOLDER_PERSON_NAME = "person_name"
HOLDER_POSITION_NAME = "position_name"
HOLDER_KEY_FIELDS = [
    HOLDER_PERSON_NAME,
    HOLDER_POSITION_NAME,
    "position_organization",
    "position_start_date",
    "position_end_date",
]
HOLDER_FIELDS = [
    "person_dob",
    "person_bio",
    "person_countries",
    "position_description",
    "position_jurisdiction",
]
HOLDER_EXPECTED_KEYS = [*HOLDER_KEY_FIELDS, *HOLDER_FIELDS]


# ===========================================================================
# Schema — a fixture directory on disk
# ===========================================================================


@dataclass
class Fixture:
    """One authored page on disk plus its answer key.

    ``html`` is the page body; the model reads the plaintext derived from it
    (same derivation the live pipeline applies to Pravda's captured HTML).
    ``expected`` is the list of complete flat holder dicts. ``screenshot`` is
    attached iff ``screenshot.png`` sits beside the
    HTML.
    """

    id: str
    html: str
    url: str
    expected: list[dict]
    screenshot: bytes | None


# ===========================================================================
# Text derivation — HTML → the plaintext the model reads
# ===========================================================================


def html_to_text(html: str) -> str:
    """Derive the plaintext the model reads from the page HTML.

    ``get_text`` is a ``textContent``-style walk — it returns all text in the
    DOM regardless of CSS, unlike ``inner_text`` which drops anything not
    visibly rendered (opensanctions/pravda#14). This is the same derivation
    the live pipeline applies to Pravda's captured HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())


# ===========================================================================
# Pair-level P/R/F1 — exact (person_name, position_name) set match
# ===========================================================================


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_page(
    expected: set[tuple],
    got: set[tuple],
) -> tuple[int, int, int]:
    """TP/FP/FN for exact set equality of holding observation keys."""
    tp = len(expected & got)
    fp = len(got - expected)
    fn = len(expected - got)
    return tp, fp, fn


def page_prf(
    expected: set[tuple],
    got: set[tuple],
) -> tuple[float, float, float]:
    """Per-fixture P/R/F1, scoring empty-agreement as perfect.

    A holder-free page (expected empty) that extracts nothing is a correct
    result, but ``prf`` returns 0.0 when there are no positives on either
    side. Treat that case as a flawless 1.0/1.0/1.0 so it does not drag down
    the macro average. Micro is unaffected: it sums raw TP/FP/FN, where an
    empty page correctly contributes nothing.
    """
    if not expected and not got:
        return 1.0, 1.0, 1.0
    tp, fp, fn = score_page(expected, got)
    return prf(tp, fp, fn)


# ===========================================================================
# Loading — strict validation of the full-schema answer key
# ===========================================================================


def _holder_key(holder: dict) -> tuple:
    return tuple(holder[name] for name in HOLDER_KEY_FIELDS)


def _parse_holder(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise TypeError(f"expected holder must be an object, got {type(raw).__name__}")
    if set(raw) != set(HOLDER_EXPECTED_KEYS):
        raise ValueError(
            f"expected holder must have exactly {HOLDER_EXPECTED_KEYS}, "
            f"got {sorted(raw)}"
        )
    holder: dict = {}
    for key in HOLDER_EXPECTED_KEYS:
        value = raw[key]
        if key == "person_countries":
            if not isinstance(value, list) or not all(
                isinstance(country, str) for country in value
            ):
                raise TypeError("person_countries must be a list of strings")
        elif value is not None and not isinstance(value, str):
            raise TypeError(
                f"{key} must be a string or null, got {type(value).__name__}"
            )
        holder[key] = value
    if not holder[HOLDER_PERSON_NAME]:
        raise ValueError("person_name must be a non-empty string")
    if not holder[HOLDER_POSITION_NAME]:
        raise ValueError("position_name must be a non-empty string")
    return holder


def _parse_expected(raw: object) -> list[dict]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected.json must be an object, got {type(raw).__name__}")
    if set(raw) != {"holders"}:
        raise ValueError(f"expected.json must have key {{holders}}, got {sorted(raw)}")
    raw_holders = raw["holders"]
    if not isinstance(raw_holders, list):
        raise TypeError("holders must be a list")
    holders = [_parse_holder(h) for h in raw_holders]
    # The answer key must be unambiguous: each full holding observation
    # appears at most once. A duplicate key is our bug — fail loudly.
    seen: set[tuple] = set()
    for holder in holders:
        key = _holder_key(holder)
        if key in seen:
            raise ValueError(f"duplicate holding observation in expected.json: {key}")
        seen.add(key)
    return holders


def load_fixtures(fixtures_dir: Path) -> list[Fixture]:
    """Load every fixture directory under ``fixtures_dir``.

    A fixture is a subdirectory (its name is the fixture id) containing
    ``page.html`` and ``expected.json`` (the full-schema answer key), plus an
    optional ``screenshot.png`` and ``url.txt``. Malformed or duplicate keys
    raise loudly. The answer key includes all explicit person-position
    relationships and omits names not tied to a stated position.
    """
    fixtures: list[Fixture] = []
    for d in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        html = (d / "page.html").read_text(encoding="utf-8")
        data = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        holders = _parse_expected(data)
        url_file = d / "url.txt"
        url = (
            url_file.read_text(encoding="utf-8").strip()
            if url_file.exists()
            else f"https://fixtures.invalid/{d.name}/"
        )
        shot = d / "screenshot.png"
        screenshot = shot.read_bytes() if shot.exists() else None
        fixtures.append(
            Fixture(
                id=d.name,
                html=html,
                url=url,
                expected=holders,
                screenshot=screenshot,
            )
        )
    return fixtures


# ===========================================================================
# Actual indexing — flatten_persons output keyed by the match key
# ===========================================================================


def _index_actual(
    rows: list[dict],
) -> tuple[dict[tuple, dict], list[tuple]]:
    """Index flattened holders by their full observation key.

    Returns the index plus any duplicate keys the model emitted. Duplicates
    are reported rather than silently collapsed.
    """
    actual: dict[tuple, dict] = {}
    duplicates: list[tuple] = []
    for row in rows:
        key = tuple(row[name] for name in HOLDER_KEY_FIELDS)
        if key in actual:
            duplicates.append(key)
        else:
            actual[key] = row
    return actual, duplicates


# ===========================================================================
# Per-holder field scoring
# ===========================================================================


def _field_mismatches(expected_holder: dict, actual_row: dict) -> list[str]:
    """Exact-compare every non-key scalar field; return a diagnostic per gap."""
    gaps: list[str] = []
    for name in HOLDER_FIELDS:
        want = expected_holder[name]
        got = actual_row[name]
        if want != got:
            gaps.append(f"{name}: want {want!r} got {got!r}")
    return gaps


@dataclass
class HolderDiag:
    key: tuple
    matched: bool
    field_mismatches: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.matched and not self.field_mismatches


@dataclass
class FixtureResult:
    fx: Fixture
    tp: int
    fp: int
    fn: int
    diags: list[HolderDiag]
    expected_keys: frozenset[tuple]
    actual_keys: frozenset[tuple]
    extra_keys: list[tuple]
    duplicate_keys: list[tuple]

    @property
    def n_expected(self) -> int:
        return len(self.diags)

    @property
    def n_matched(self) -> int:
        return sum(1 for d in self.diags if d.matched)

    @property
    def n_complete(self) -> int:
        return sum(1 for d in self.diags if d.complete)


def _score_fixture(
    fx: Fixture, actual: dict[tuple, dict], duplicates: list[tuple]
) -> FixtureResult:
    diags: list[HolderDiag] = []
    for holder in fx.expected:
        key = _holder_key(holder)
        if key in actual:
            row = actual[key]
            gaps = _field_mismatches(holder, row)
            diags.append(HolderDiag(key=key, matched=True, field_mismatches=gaps))
        else:
            diags.append(HolderDiag(key=key, matched=False))
    expected_keys = frozenset(d.key for d in diags)
    actual_keys = frozenset(actual)
    return FixtureResult(
        fx=fx,
        tp=len(expected_keys & actual_keys),
        fp=len(actual_keys - expected_keys),
        fn=len(expected_keys - actual_keys),
        diags=diags,
        expected_keys=expected_keys,
        actual_keys=actual_keys,
        extra_keys=sorted(actual_keys - expected_keys, key=repr),
        duplicate_keys=sorted(set(duplicates), key=repr),
    )


# ===========================================================================
# Run
# ===========================================================================


def _shape(fixture: Fixture) -> str:
    """Table label for the pipeline path a fixture drives: the image path
    when a screenshot is attached, the text path otherwise."""
    return "image" if fixture.screenshot is not None else "text"


def _pair_label(key: tuple) -> str:
    person, position, organization, start_date, end_date = key
    details = [value for value in (organization, start_date, end_date) if value]
    suffix = f" — {', '.join(details)}" if details else ""
    return f"{person} ({position}){suffix}"


def _fixture_notes(r: FixtureResult) -> str:
    notes: list[str] = []
    if r.fp:
        notes.append("extra: " + "; ".join(_pair_label(k) for k in r.extra_keys))
    if r.fn:
        missed = [d.key for d in r.diags if not d.matched]
        notes.append("missed: " + "; ".join(_pair_label(k) for k in missed))
    if r.duplicate_keys:
        notes.append(
            "duplicate actual: " + "; ".join(_pair_label(k) for k in r.duplicate_keys)
        )
    for d in r.diags:
        if not d.matched or d.complete:
            continue
        notes.append(
            f"{_pair_label(d.key)}: fields [" + "; ".join(d.field_mismatches) + "]"
        )
    return ", ".join(notes)


def _print_holder_details(r: FixtureResult) -> None:
    for d in r.diags:
        if d.complete:
            print(f"    - {_pair_label(d.key)}: complete", file=sys.stderr)
            continue
        if not d.matched:
            print(f"    - {_pair_label(d.key)}: MISSED", file=sys.stderr)
            continue
        print(
            f"    - {_pair_label(d.key)}: incomplete | "
            + "; ".join(d.field_mismatches),
            file=sys.stderr,
        )


def _print_table(results: list[FixtureResult], verbose: bool) -> None:
    print(file=sys.stderr)
    print(
        f"{'fixture':20} {'path':5} {'TP':>3} {'FP':>3} {'FN':>3} {'comp':>5}  notes",
        file=sys.stderr,
    )
    print("-" * 100, file=sys.stderr)
    for r in results:
        done = f"{r.n_complete}/{r.n_expected}"
        notes = _fixture_notes(r)
        print(
            f"{r.fx.id:20} {_shape(r.fx):5} {r.tp:3d} {r.fp:3d} {r.fn:3d} "
            f"{done:>5}  {notes}",
            file=sys.stderr,
        )
        if verbose:
            _print_holder_details(r)


def _print_summary(results: list[FixtureResult]) -> None:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)

    per_page = [page_prf(set(r.expected_keys), set(r.actual_keys)) for r in results]
    n = len(per_page)
    macro_p = sum(p for p, _, _ in per_page) / n if n else 0.0
    macro_r = sum(rc for _, rc, _ in per_page) / n if n else 0.0
    macro_f1 = sum(f1 for _, _, f1 in per_page) / n if n else 0.0

    total_expected = sum(r.n_expected for r in results)
    total_matched = sum(r.n_matched for r in results)
    total_complete = sum(r.n_complete for r in results)

    field_correct = 0
    field_total = 0
    for r in results:
        for d in r.diags:
            if d.matched:
                field_total += len(HOLDER_FIELDS)
                field_correct += len(HOLDER_FIELDS) - len(d.field_mismatches)
    field_pct = (100.0 * field_correct / field_total) if field_total else 0.0

    print(file=sys.stderr)
    print(f"{len(results)} fixture(s)", file=sys.stderr)
    print("                     precision   recall     F1", file=sys.stderr)
    print(
        f"micro (holdings)   {micro_p:8.3f}  {micro_r:8.3f}  {micro_f1:8.3f}",
        file=sys.stderr,
    )
    print(
        f"macro (holdings)   {macro_p:8.3f}  {macro_r:8.3f}  {macro_f1:8.3f}",
        file=sys.stderr,
    )
    print(f"                   (TP={tp} FP={fp} FN={fn})", file=sys.stderr)
    print(
        f"holders            {total_complete} complete / {total_matched} matched "
        f"/ {total_expected} expected",
        file=sys.stderr,
    )
    print(
        f"field accuracy     {field_correct}/{field_total} non-key scalars correct "
        f"among matched ({field_pct:.1f}%)",
        file=sys.stderr,
    )


def run(fixtures: list[Fixture], verbose: bool) -> None:
    """Derive text → extract → flatten → score each fixture, then print a
    per-fixture table and a micro/macro + completeness summary."""
    config = load_config()
    client = OpenAI()
    results: list[FixtureResult] = []
    for fx in fixtures:
        text = html_to_text(fx.html)
        metadata = metadata_from_html(fx.url, fx.html)
        extraction = extract(
            client,
            config.model,
            config.image,
            metadata,
            text,
            fx.screenshot,
        )
        rows = flatten_persons(extraction)
        actual, duplicates = _index_actual(rows)
        result = _score_fixture(fx, actual, duplicates)
        results.append(result)

        log.info(
            "%s: %d holder(s), %d matched, %d complete",
            fx.id,
            len(rows),
            result.n_matched,
            result.n_complete,
        )

    _print_table(results, verbose)
    _print_summary(results)


# ===========================================================================
# CLI
# ===========================================================================


@click.command()
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print a per-holder detail block for every fixture.",
)
@click.option(
    "--fixtures",
    "fixtures_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=str(FIXTURES_DIR),
    show_default=True,
    help="Directory of fixture subdirectories.",
)
def cli(verbose: bool, fixtures_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    fixtures = load_fixtures(Path(fixtures_dir))
    run(fixtures, verbose)


if __name__ == "__main__":
    cli()
