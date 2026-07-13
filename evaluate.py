"""Score the extraction pipeline against hand-authored fixture pages.

The pipeline is meant to read holders *verbatim* from a page — names, titles,
dates copied as written. Scoring it against a third-party golden set does not
work: those sets come from scrapers that have already normalized (title-cased
names, expanded role codes, merged variants), so a faithful verbatim
extraction fails while a normalized guess passes. Here we own both ends
instead: each fixture is an authored HTML page whose holders we know exactly.
The harness derives the plaintext the model reads (the same derivation the
live pipeline applies to Pravda's captured HTML), runs the real
``kolkhoz.extract``, and scores the returned (person, position) pairs against
the answer key. Exact-match is honest again.

Fixtures live as directories under ``fixtures/`` — one per page, named after
the fixture id. Each directory holds:

- ``page.html``      the page itself, authored in whatever shape the case
                     needs (a roster table, a prose bio, a news lead, ...).
                     Variety is the point: real failure modes live in the
                     layout, so fixtures differ in structure on purpose.
- ``expected.json``  the answer key: a list of ``{"person", "position"}``
                     objects for every holder the page states. Every name on
                     the page that is *not* here is, by construction, a
                     distractor the pipeline must exclude — founders, former
                     holders, donors, contacts, honorees. A non-holder needs
                     no separate declaration: it is just a name in the HTML
                     absent from the key.
- ``screenshot.png`` optional. When present it is tiled and attached as the
                     page screenshot, driving the image path. It is a browser
                     capture of ``page.html``, so the names appear in *both*
                     the derived text and the image: this exercises the image
                     path and cross-channel consistency (no double-counting),
                     not the "thin text, names only in pixels" case. To probe
                     that case the pixel-only names would have to live inside
                     real ``<img>`` elements rather than page text.

The harness calls the real model via ``kolkhoz.extract`` (it spends tokens and
reads ``.env``) and is fully decoupled from Pravda and the database: no
snapshots, no Pages, no Extractions on disk — just read, extract, score.

Usage:

    uv run python evaluate.py    # run all fixtures in fixtures/
    uv run python evaluate.py -v # per-fixture detail
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from bs4 import BeautifulSoup

from kolkhoz import extract

log = logging.getLogger("evaluate")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ===========================================================================
# Schema — a fixture directory on disk
# ===========================================================================


@dataclass
class Fixture:
    """One authored page on disk: its HTML, the holders we expect, and an
    optional browser screenshot that drives the image path.

    ``html`` is the page body; the model reads the plaintext derived from it
    (same derivation the live pipeline applies to Pravda's captured HTML).
    ``expected`` is the answer key. ``screenshot`` is attached iff
    ``screenshot.png`` sits beside the HTML.
    """

    id: str
    html: str
    expected: set[tuple[str, str]]
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
# Scoring — (person, position) pairs, exact match, micro/macro
# ===========================================================================


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_page(
    expected: set[tuple[str, str]], got: set[tuple[str, str]]
) -> tuple[int, int, int]:
    """TP/FP/FN for exact set equality of (person, position) pairs."""
    tp = len(expected & got)
    fp = len(got - expected)
    fn = len(expected - got)
    return tp, fp, fn


def page_prf(
    expected: set[tuple[str, str]], got: set[tuple[str, str]]
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
# Loading
# ===========================================================================


def load_fixtures(fixtures_dir: Path) -> list[Fixture]:
    """Load every fixture directory under ``fixtures_dir``.

    A fixture is a subdirectory (its name is the fixture id) containing
    ``page.html`` and ``expected.json`` — a list of ``{person, position}``
    objects — plus an optional ``screenshot.png``. Missing keys raise loudly;
    names in the HTML absent from the key are distractors by construction.
    """
    fixtures: list[Fixture] = []
    for d in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        html = (d / "page.html").read_text(encoding="utf-8")
        data = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        expected = {(h["person"], h["position"]) for h in data}
        shot = d / "screenshot.png"
        screenshot = shot.read_bytes() if shot.exists() else None
        fixtures.append(
            Fixture(id=d.name, html=html, expected=expected, screenshot=screenshot)
        )
    return fixtures


# ===========================================================================
# Run
# ===========================================================================


def _shape(fixture: Fixture) -> str:
    """Table label for the pipeline path a fixture drives: the image path
    when a screenshot is attached, the text path otherwise."""
    return "image" if fixture.screenshot is not None else "text"


def run(fixtures: list[Fixture], verbose: bool) -> None:
    """Derive text → extract → score each fixture, then print a summary table."""
    rows: list[tuple[Fixture, set, set, int, int, int]] = []
    for fx in fixtures:
        text = html_to_text(fx.html)
        extraction = extract(text, fx.screenshot)

        got = {
            (person.name, position.name)
            for person in extraction.persons
            for position in person.positions
        }
        tp, fp, fn = score_page(fx.expected, got)
        rows.append((fx, fx.expected, got, tp, fp, fn))

        log.info(
            "%s: %d holder(s)",
            fx.id,
            sum(len(p.positions) for p in extraction.persons),
        )

    # ---- per-fixture table ------------------------------------------------
    print(file=sys.stderr)
    print(
        f"{'fixture':20} {'path':10} {'TP':>3} {'FP':>3} {'FN':>3}  notes",
        file=sys.stderr,
    )
    print("-" * 90, file=sys.stderr)
    for fx, expected, got, tp, fp, fn in rows:
        notes: list[str] = []
        if fp:
            notes.append(
                "extra: " + "; ".join(f"{h} ({p})" for h, p in sorted(got - expected))
            )
        if fn:
            notes.append(
                "missed: " + "; ".join(f"{h} ({p})" for h, p in sorted(expected - got))
            )
        if verbose and not notes:
            notes.append(
                "expected: " + "; ".join(f"{h} ({p})" for h, p in sorted(expected))
            )
        print(
            f"{fx.id:20} {_shape(fx):10} {tp:3d} {fp:3d} {fn:3d}  {', '.join(notes)}",
            file=sys.stderr,
        )

    # ---- micro/macro summary over pairs -----------------------------------
    tp = sum(r[3] for r in rows)
    fp = sum(r[4] for r in rows)
    fn = sum(r[5] for r in rows)
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)

    per_page = [page_prf(exp, got) for _, exp, got, *_ in rows]
    n = len(per_page)
    macro_p = sum(p for p, _, _ in per_page) / n if n else 0.0
    macro_r = sum(r for _, r, _ in per_page) / n if n else 0.0
    macro_f1 = sum(f for _, _, f in per_page) / n if n else 0.0

    print(file=sys.stderr)
    print(f"{len(rows)} fixture(s)", file=sys.stderr)
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


# ===========================================================================
# CLI
# ===========================================================================


@click.command()
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Also print the expected pairs for clean fixtures.",
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
