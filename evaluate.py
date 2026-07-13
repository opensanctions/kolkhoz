"""Score the extraction pipeline against hand-authored fixture pages.

The pipeline is meant to read holders *verbatim* from a page — names, titles,
dates copied as written. Scoring it against a third-party golden set does not
work: those sets come from scrapers that have already normalized (title-cased
names, expanded role codes, merged variants), so a faithful verbatim
extraction fails while a normalized guess passes. Here we own both ends
instead: each fixture is an authored HTML page whose holders we know exactly.
The harness derives the plaintext the model reads (the same derivation the
live pipeline applies to Pravda's captured HTML), runs the real
``kolkhoz.extract``, flattens the result with the production
``flatten_persons``, and scores against the answer key. Exact-match is honest
again.

Fixtures live as directories under ``fixtures/`` — one per page, named after
the fixture id. Each directory holds:

- ``page.html``      the page itself, authored in whatever shape the case
                     needs (a roster table, a prose bio, a news lead, ...).
                     Variety is the point: real failure modes live in the
                     layout, so fixtures differ in structure on purpose.
- ``expected.json``  the answer key (the full-schema format below).
- ``screenshot.png`` optional. When present it is tiled and attached as the
                     page screenshot, driving the image path. It is a browser
                     capture of ``page.html``, so the names appear in *both*
                     the derived text and the image: this exercises the image
                     path and cross-channel consistency (no double-counting),
                     not the "thin text, names only in pixels" case. To probe
                     that case the pixel-only names would have to live inside
                     real ``<img>`` elements rather than page text.

The answer key is an object::

    {
      "evidence_required": false,
      "holders": [
        {
          "person_name": "Marek Dvořák",
          "person_dob": null,
          "person_bio": null,
          "person_country": null,
          "position_name": "Chair of the Board",
          "position_description": null,
          "position_jurisdiction": null,
          "position_start_date": null,
          "position_end_date": null
        }
      ]
    }

- ``evidence_required`` is a fixture-level boolean. When true, every matched
  holder must carry at least one verbatim evidence quote (prose-based
  fixtures, where the page ties holders to roles in running text). When
  false, holders may carry none (table/list-only fixtures, where the schema
  permits empty evidence).
- ``holders`` is the complete flat extraction schema — all nine scalar keys
  explicit, including nulls. ``person_name`` and ``position_name`` are the
  match key; the other seven are compared field by field on matched holders.

Every name on the page that is *not* in ``holders`` is, by construction, a
distractor the pipeline must exclude — founders, former holders, donors,
contacts, honorees. A non-holder needs no separate declaration: it is just a
name in the HTML absent from the key.

Scoring has four layers:

1. **Pair match** — expected and actual holders are keyed by exact
   ``(person_name, position_name)``. Pair-level micro/macro P/R/F1 are
   reported as before. Duplicate keys in the answer key fail loudly
   (ambiguous key); duplicates in the model output are reported, never
   silently collapsed.
2. **Field comparison** — for every matched pair, the remaining seven scalar
   fields are compared exactly, and every gap is printed.
3. **Evidence** — every actual evidence quote is validated as a verbatim
   (whitespace-normalized, case-sensitive) substring of the page plaintext.
   A quote that is not on the page is invalid. Evidence is *required* when the
   fixture is prose-based; table/list-only fixtures permit none.
4. **Complete holders** — a matched holder is *complete* when its pair
   matches, every scalar field matches, and its evidence is valid (and, when
   required, present). Reported per fixture and in aggregate.

Empty fixtures (``holders: []``) score as a flawless empty extraction.

The harness calls the real model via ``kolkhoz.extract`` (it spends tokens and
reads ``.env``) and is fully decoupled from Pravda and the database: no
snapshots, no Pages, no Extractions on disk — just read, extract, score.

Usage:

    uv run python evaluate.py    # run all fixtures in fixtures/
    uv run python evaluate.py -v # per-holder detail
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
from kolkhoz.extract import extract, flatten_persons

log = logging.getLogger("evaluate")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The flat extraction/storage scalar schema. ``person_name`` and
# ``position_name`` are the match key; the rest are compared field by field on
# matched holders. Every expected holder must state all nine explicitly,
# including nulls. Actual rows from ``flatten_persons`` additionally carry
# ``evidence_quotes`` (a list), which is validated — not pinned — against the
# page plaintext.
HOLDER_PERSON_NAME = "person_name"
HOLDER_POSITION_NAME = "position_name"
HOLDER_FIELDS = [
    "person_dob",
    "person_bio",
    "person_country",
    "position_description",
    "position_jurisdiction",
    "position_start_date",
    "position_end_date",
]
HOLDER_SCALAR_KEYS = [HOLDER_PERSON_NAME, HOLDER_POSITION_NAME, *HOLDER_FIELDS]
EVIDENCE_QUOTES = "evidence_quotes"


# ===========================================================================
# Schema — a fixture directory on disk
# ===========================================================================


@dataclass
class Fixture:
    """One authored page on disk plus its answer key.

    ``html`` is the page body; the model reads the plaintext derived from it
    (same derivation the live pipeline applies to Pravda's captured HTML).
    ``expected`` is the list of flat holder dicts — each carries every scalar
    key. ``screenshot`` is attached iff ``screenshot.png`` sits beside the
    HTML.
    """

    id: str
    html: str
    evidence_required: bool
    expected: list[dict[str, str | None]]
    screenshot: bytes | None


# ===========================================================================
# Text derivation — HTML → the plaintext the model reads
# ===========================================================================


def html_to_text(html: str) -> str:
    """Derive the plaintext the model reads from the page HTML.

    ``get_text`` is a ``textContent``-style walk — it returns all text in the
    DOM regardless of CSS, unlike ``inner_text`` which drops anything not
    visibly rendered (opensanctions/pravda#14). This is the same derivation
    the live pipeline applies to Pravda's captured HTML, and the same text
    evidence quotes are validated against.
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
    expected: set[tuple[str | None, str | None]],
    got: set[tuple[str | None, str | None]],
) -> tuple[int, int, int]:
    """TP/FP/FN for exact set equality of (person_name, position_name) pairs."""
    tp = len(expected & got)
    fp = len(got - expected)
    fn = len(expected - got)
    return tp, fp, fn


def page_prf(
    expected: set[tuple[str | None, str | None]],
    got: set[tuple[str | None, str | None]],
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


def _holder_key(holder: dict[str, str | None]) -> tuple[str | None, str | None]:
    return (holder[HOLDER_PERSON_NAME], holder[HOLDER_POSITION_NAME])


def _parse_holder(raw: object) -> dict[str, str | None]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected holder must be an object, got {type(raw).__name__}")
    if set(raw) != set(HOLDER_SCALAR_KEYS):
        raise ValueError(
            f"expected holder must have exactly {HOLDER_SCALAR_KEYS}, got {sorted(raw)}"
        )
    holder: dict[str, str | None] = {}
    for key in HOLDER_SCALAR_KEYS:
        value = raw[key]
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"{key} must be a string or null, got {type(value).__name__}"
            )
        holder[key] = value
    if not holder[HOLDER_PERSON_NAME]:
        raise ValueError("person_name must be a non-empty string")
    return holder


def _parse_expected(raw: object) -> tuple[bool, list[dict[str, str | None]]]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected.json must be an object, got {type(raw).__name__}")
    if set(raw) != {"evidence_required", "holders"}:
        raise ValueError(
            "expected.json must have keys {evidence_required, holders}, "
            f"got {sorted(raw)}"
        )
    evidence_required = raw["evidence_required"]
    if not isinstance(evidence_required, bool):
        raise TypeError("evidence_required must be a boolean")
    raw_holders = raw["holders"]
    if not isinstance(raw_holders, list):
        raise TypeError("holders must be a list")
    holders = [_parse_holder(h) for h in raw_holders]
    # The answer key must be unambiguous: each (person_name, position_name)
    # pair appears at most once. A duplicate key is our bug — fail loudly
    # rather than silently collapse it.
    seen: set[tuple] = set()
    for holder in holders:
        key = _holder_key(holder)
        if key in seen:
            raise ValueError(
                f"duplicate (person_name, position_name) in expected.json: {key}"
            )
        seen.add(key)
    return evidence_required, holders


def load_fixtures(fixtures_dir: Path) -> list[Fixture]:
    """Load every fixture directory under ``fixtures_dir``.

    A fixture is a subdirectory (its name is the fixture id) containing
    ``page.html`` and ``expected.json`` (the full-schema answer key), plus an
    optional ``screenshot.png``. Malformed keys or duplicate answer-key pairs
    raise loudly; names in the HTML absent from the key are distractors by
    construction.
    """
    fixtures: list[Fixture] = []
    for d in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        html = (d / "page.html").read_text(encoding="utf-8")
        data = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        evidence_required, holders = _parse_expected(data)
        shot = d / "screenshot.png"
        screenshot = shot.read_bytes() if shot.exists() else None
        fixtures.append(
            Fixture(
                id=d.name,
                html=html,
                evidence_required=evidence_required,
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
    """Index flattened holders by (person_name, position_name).

    Returns the index plus any duplicate keys the model emitted. Duplicates
    are reported (never silently collapsed): the same pair emitted twice is a
    model defect worth surfacing even though pair-set scoring counts it once.
    """
    actual: dict[tuple, dict] = {}
    duplicates: list[tuple] = []
    for row in rows:
        key = (row[HOLDER_PERSON_NAME], row[HOLDER_POSITION_NAME])
        if key in actual:
            duplicates.append(key)
        else:
            actual[key] = row
    return actual, duplicates


# ===========================================================================
# Per-holder field + evidence scoring
# ===========================================================================


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _field_mismatches(
    expected_holder: dict[str, str | None], actual_row: dict
) -> list[str]:
    """Exact-compare every non-key scalar field; return a diagnostic per gap."""
    gaps: list[str] = []
    for name in HOLDER_FIELDS:
        want = expected_holder[name]
        got = actual_row[name]
        if want != got:
            gaps.append(f"{name}: want {want!r} got {got!r}")
    return gaps


def _evidence_status(
    quotes: list[str], text: str, required: bool
) -> tuple[int, list[str], bool]:
    """Validate actual evidence quotes against the page plaintext.

    Each quote must be a verbatim (whitespace-normalized, case-sensitive)
    substring of the normalized page plaintext; a quote not on the page is
    invalid. Returns ``(n_valid, invalid_quotes, ok)`` where ``ok`` is true
    when no quote is invalid and, if evidence is required for this fixture,
    at least one valid quote exists. Quote-span equality with an expected
    string is *not* required — only that each quote actually appears on the
    page.
    """
    valid = 0
    invalid: list[str] = []
    for quote in quotes:
        if _normalize_ws(quote) in text:
            valid += 1
        else:
            invalid.append(quote)
    ok = not invalid and (not required or valid >= 1)
    return valid, invalid, ok


@dataclass
class HolderDiag:
    key: tuple[str | None, str | None]
    matched: bool
    field_mismatches: list[str] = field(default_factory=list)
    n_valid_quotes: int = 0
    invalid_quotes: list[str] = field(default_factory=list)
    evidence_ok: bool = False

    @property
    def complete(self) -> bool:
        # Complete = matched pair, every scalar field correct, and valid
        # evidence (present when the fixture requires it, never fabricated).
        return self.matched and not self.field_mismatches and self.evidence_ok


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
    fx: Fixture, text: str, actual: dict[tuple, dict], duplicates: list[tuple]
) -> FixtureResult:
    diags: list[HolderDiag] = []
    for holder in fx.expected:
        key = _holder_key(holder)
        if key in actual:
            row = actual[key]
            gaps = _field_mismatches(holder, row)
            valid, invalid, ok = _evidence_status(
                row[EVIDENCE_QUOTES], text, fx.evidence_required
            )
            diags.append(
                HolderDiag(
                    key=key,
                    matched=True,
                    field_mismatches=gaps,
                    n_valid_quotes=valid,
                    invalid_quotes=invalid,
                    evidence_ok=ok,
                )
            )
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
        extra_keys=sorted(actual_keys - expected_keys),
        duplicate_keys=sorted(set(duplicates)),
    )


# ===========================================================================
# Run
# ===========================================================================


def _shape(fixture: Fixture) -> str:
    """Table label for the pipeline path a fixture drives: the image path
    when a screenshot is attached, the text path otherwise."""
    return "image" if fixture.screenshot is not None else "text"


def _pair_label(key: tuple[str | None, str | None]) -> str:
    person, position = key
    return f"{person} ({position})"


def _holder_evidence_note(d: HolderDiag, required: bool) -> str:
    if d.invalid_quotes:
        return f"{len(d.invalid_quotes)} invalid quote(s)"
    if required and d.n_valid_quotes == 0:
        return "missing required evidence"
    return ""


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
        bits: list[str] = []
        if d.field_mismatches:
            bits.append("fields [" + "; ".join(d.field_mismatches) + "]")
        ev = _holder_evidence_note(d, r.fx.evidence_required)
        if ev:
            bits.append(ev)
        notes.append(f"{_pair_label(d.key)}: " + ", ".join(bits))
    return ", ".join(notes)


def _print_holder_details(r: FixtureResult) -> None:
    for d in r.diags:
        if d.complete:
            print(f"    - {_pair_label(d.key)}: complete", file=sys.stderr)
            continue
        if not d.matched:
            print(f"    - {_pair_label(d.key)}: MISSED", file=sys.stderr)
            continue
        bits = ["incomplete"]
        if d.field_mismatches:
            bits.append("; ".join(d.field_mismatches))
        ev = _holder_evidence_note(d, r.fx.evidence_required)
        if ev:
            bits.append(ev)
        bits.append(f"{d.n_valid_quotes} valid quote(s)")
        print(f"    - {_pair_label(d.key)}: " + " | ".join(bits), file=sys.stderr)


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
        f"micro (pairs)      {micro_p:8.3f}  {micro_r:8.3f}  {micro_f1:8.3f}",
        file=sys.stderr,
    )
    print(
        f"macro (pairs)      {macro_p:8.3f}  {macro_r:8.3f}  {macro_f1:8.3f}",
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
        extraction = extract(client, config.model, config.image, text, fx.screenshot)
        rows = flatten_persons(extraction)
        actual, duplicates = _index_actual(rows)
        result = _score_fixture(fx, text, actual, duplicates)
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
