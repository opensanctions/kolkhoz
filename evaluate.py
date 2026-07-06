"""Score the extraction pipeline against hand-authored synthetic fixtures.

The pipeline is meant to read holders *verbatim* from a page — names, titles,
dates copied as written. Scoring it against a third-party golden set does not
work: those sets come from scrapers that have already normalized (title-cased
names, expanded role codes, merged variants), so a faithful verbatim
extraction fails while a normalized guess passes. Here we own both ends
instead: each fixture is an authored page whose holders we know exactly. The
harness renders it to HTML, derives the plaintext the model reads, runs the
real ``kolkhoz.extract``, and scores the returned (human, position) pairs
against what we put in. Exact-match is honest again.

Fixtures live as JSON in ``fixtures/`` (one file per page: an organization, a
list of holders, and optional distractor HTML). The filename stem is the
fixture id. Every fixture renders through the same single layout — a roster
table of its holders — so the only thing that varies between cases is the
data, not the chrome.

Scope is deliberately narrow:

- **Two modes, one per fixture.** Each fixture is either ``text`` (the
  default) or ``screenshot``. A ``text`` fixture renders to HTML, derives the
  plaintext, and calls ``extract(text, None)`` — the text-only path. A
  ``screenshot`` fixture additionally rasterizes that same HTML to a PNG with
  PIL (a mini-renderer for the one template ``render_html`` emits, not a
  general HTML engine) and calls ``extract(text, screenshot_blob)`` — the
  screenshot path: image tiling, base64 encoding, and the model reading
  holders from pixels. Because the screenshot is rendered from the same roster
  the text comes from, a correct extraction still recovers every holder, so a
  screenshot fixture is a clean pass/fail on the image plumbing rather than an
  OCR-hardship test. The genuinely hard case (names baked into real
  photographs, org charts, headshot rosters) stays out of reach — a future
  fixture class.

- **(human, position) pairs only**, scored per fixture by exact string
  equality. Richer fields (dob, bio, dates) can be added once fixtures carry
  them. ``page_type`` is part of kolkhoz's extraction output but is not scored
  or reported here.

- **Hand-authored fixtures.** Each is a deliberate, legible case that probes
  one behavior: clean recall, distractor precision, verbatim preservation of
  non-ASCII names, hallucination resistance on a holder-free page, one person
  holding two titles, and roster completeness at scale.

The harness calls the real model via ``kolkhoz.extract`` (it spends tokens and
reads ``.env``) and is fully decoupled from Pravda and the database: no
snapshots, no Pages, no Extractions on disk — just render, extract, score.

Usage:

    uv run python evaluate.py    # run all fixtures in fixtures/
    uv run python evaluate.py -v # per-fixture detail
"""

import io
import json
import logging
import sys
from html import escape
from pathlib import Path
from typing import Literal

import click
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from kolkhoz import extract

log = logging.getLogger("evaluate")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ===========================================================================
# Schema — the authored ground truth, loaded from JSON
# ===========================================================================


class SyntheticHolder(BaseModel):
    """A single (human, position) pair the page states and we expect back."""

    human: str
    position: str


class SyntheticPage(BaseModel):
    """One authored page: the org it is about, the holders it names, and any
    distractor HTML that must *not* be extracted.

    ``id`` is the JSON filename stem; the file *is* the fixture, so there is no
    id field in the JSON itself. ``holders`` is the answer key — exactly the
    (human, position) pairs a correct extraction returns. ``extra_html`` is
    unstructured noise (footers, contact lines, history) sprinkled into the
    rendered page to test precision against distractors. ``mode`` picks the
    pipeline path: ``text`` (default) sends plaintext only; ``screenshot``
    also rasterizes the rendered HTML to a PNG and attaches it, exercising
    the image path of ``extract``.
    """

    id: str
    organization: str
    holders: list[SyntheticHolder] = Field(default_factory=list)
    extra_html: str = ""
    mode: Literal["text", "screenshot"] = "text"


# ===========================================================================
# Rendering — one layout (a roster table) → HTML → plaintext
# ===========================================================================


def render_html(page: SyntheticPage) -> str:
    """Render a page to standalone HTML through a single roster layout.

    Holders become a Name/Position table; a holder-free page renders no table
    (just the organization header and any distractor HTML). The model reads
    the derived plaintext, so what varies the reading task between fixtures is
    the data, not the markup.
    """
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{escape(page.organization)}</title></head><body>",
        f"<h1>{escape(page.organization)}</h1>",
    ]
    if page.holders:
        parts.append(
            "<table><thead><tr><th>Name</th><th>Position</th></tr></thead><tbody>"
        )
        for h in page.holders:
            parts.append(
                f"<tr><td>{escape(h.human)}</td><td>{escape(h.position)}</td></tr>"
            )
        parts.append("</tbody></table>")
    if page.extra_html:
        parts.append(page.extra_html)
    parts.append("</body></html>")
    return "".join(parts)


def html_to_text(html: str) -> str:
    """Derive the plaintext the model reads from rendered HTML.

    ``get_text`` is a ``textContent``-style walk — it returns all text in the
    DOM regardless of CSS, unlike ``inner_text`` which drops anything not
    visibly rendered (opensanctions/pravda#14).
    """
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())


def render_screenshot(html: str) -> bytes:
    """Rasterize a page's HTML to a PNG the model reads as pixels.

    A mini-renderer for the single template ``render_html`` emits — the
    organization ``<h1>``, an optional Name/Position ``<table>``, then any
    stray block text from ``extra_html`` — not a general HTML engine. Two
    passes: measure to size the canvas, then draw, so the image is exactly as
    tall as its content. Used only by ``mode="screenshot"`` fixtures to drive
    the screenshot path of the pipeline (image tiling, encoding, the model
    reading holders from pixels) without a headless browser.
    """
    soup = BeautifulSoup(html, "html.parser")
    font = ImageFont.load_default(size=24)
    head_font = ImageFont.load_default(size=32)
    margin = 40
    width = 1000
    gap = 14

    def textw(text: str, f: ImageFont.ImageFont) -> int:
        return int(f.getlength(text))

    def line_height(f: ImageFont.ImageFont) -> int:
        return f.getbbox("Ag")[3] + gap

    def wrap(text: str, f: ImageFont.ImageFont, max_w: int) -> list[str]:
        words, out, cur = text.split(), [], ""
        for w in words:
            trial = w if not cur else cur + " " + w
            if textw(trial, f) <= max_w:
                cur = trial
            elif cur:
                out.append(cur)
                cur = w
            else:
                out.append(w)
                cur = ""
        if cur:
            out.append(cur)
        return out or [""]

    # Pull the pieces the known template emits out of the DOM.
    head_el = soup.find("h1")
    head = head_el.get_text(strip=True) if head_el else ""
    rows: list[tuple[str, str]] = []
    table = soup.find("table")
    if table:
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == 2:
                rows.append((cells[0], cells[1]))
    distractors: list[str] = []
    body = soup.body or soup
    for child in body.children:
        if getattr(child, "name", None) in (None, "h1", "table"):
            continue
        txt = child.get_text(separator=" ", strip=True)
        if txt:
            distractors.append(txt)

    # Pass 1: lay out (text, x, font, y) ops against a running y cursor.
    ops: list[tuple[str, int, ImageFont.ImageFont, int]] = []
    y = margin
    h_lh = line_height(head_font)
    f_lh = line_height(font)
    if head:
        ops.append((head, margin, head_font, y))
        y += h_lh + gap
    if rows:
        name_w = max(textw(name, font) for name, _ in rows)
        pos_x = margin + name_w + 60
        ops.append(("Name", margin, font, y))
        ops.append(("Position", pos_x, font, y))
        y += f_lh
        for name, pos in rows:
            ops.append((name, margin, font, y))
            ops.append((pos, pos_x, font, y))
            y += f_lh
        y += gap
    content_w = width - 2 * margin
    for d in distractors:
        for ln in wrap(d, font, content_w):
            ops.append((ln, margin, font, y))
            y += f_lh
        y += gap

    if ops:
        last_text, _, last_font, last_y = ops[-1]
        height = last_y + last_font.getbbox(last_text)[3] + margin
    else:
        height = margin * 2
    img = Image.new("RGB", (width, max(height, 120)), "white")
    draw = ImageDraw.Draw(img)
    for text, x, f, ty in ops:  # pass 2: draw
        draw.text((x, ty), text, fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# Scoring — (human, position) pairs, exact match, micro/macro
# ===========================================================================


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_page(
    expected: set[tuple[str, str]], got: set[tuple[str, str]]
) -> tuple[int, int, int]:
    """TP/FP/FN for exact set equality of (human, position) pairs."""
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


def load_fixtures(fixtures_dir: Path) -> list[SyntheticPage]:
    """Load every ``*.json`` in ``fixtures_dir``, id = filename stem."""
    pages: list[SyntheticPage] = []
    for path in sorted(fixtures_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages.append(SyntheticPage(id=path.stem, **data))
    return pages


# ===========================================================================
# Run
# ===========================================================================


def run(fixtures: list[SyntheticPage], verbose: bool) -> None:
    """Render → extract → score each fixture, then print a summary table."""
    rows: list[tuple[SyntheticPage, set, set, int, int, int]] = []
    for page in fixtures:
        html = render_html(page)
        text = html_to_text(html)
        screenshot_blob = render_screenshot(html) if page.mode == "screenshot" else None
        extraction = extract(text, screenshot_blob)

        expected = {(h.human, h.position) for h in page.holders}
        got = {(h.human, h.position) for h in extraction.holders}
        tp, fp, fn = score_page(expected, got)
        rows.append((page, expected, got, tp, fp, fn))

        log.info("%s: %d holder(s)", page.id, len(extraction.holders))

    # ---- per-fixture table ------------------------------------------------
    print(file=sys.stderr)
    print(
        f"{'fixture':20} {'mode':10} {'TP':>3} {'FP':>3} {'FN':>3}  notes",
        file=sys.stderr,
    )
    print("-" * 90, file=sys.stderr)
    for page, expected, got, tp, fp, fn in rows:
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
            f"{page.id:20} {page.mode:10} {tp:3d} {fp:3d} {fn:3d}  {', '.join(notes)}",
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
    help="Directory of fixture JSON files.",
)
def cli(verbose: bool, fixtures_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    fixtures = load_fixtures(Path(fixtures_dir))
    run(fixtures, verbose)


if __name__ == "__main__":
    cli()
