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

Fixtures live as JSON in ``fixtures/`` (one file per page: an organization, two lists of holders (text and
screenshot), and optional distractor HTML). The filename stem is the
fixture id. Every fixture renders through the same single layout — a roster
table of its holders — so the only thing that varies between cases is the
data, not the chrome.

Scope is deliberately narrow:

- **Two holder lists per fixture, no mode switch.** Each fixture declares
  ``text_holders`` (rendered to the plaintext the model reads) and
  ``screenshot_holders`` (rasterized to a roster PNG, attached iff non-empty).
  The answer key is the union of both. The three pipeline paths fall out of
  how the two are populated: text-only; screenshot-only (thin text, the
  "thin text" trigger); or both with ``text_holders`` a subset of
  ``screenshot_holders`` (text/screenshot overlap — the "image-dense"
  trigger, where the model must neither double-count the shared names nor
  miss the pixel-only ones). Both screenshot triggers mirror real branches
  of ``kolkhoz.screenshot_reason``.

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

import click
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from kolkhoz import extract

log = logging.getLogger("evaluate")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Vendored TrueType for the synthetic screenshot renderer. Pillow's bundled
# default font lacks Latin-Extended glyphs (tofu for names like Dvořák / João /
# Wójcik), so we ship DejaVu Sans (permissive Bitstream-Vera-derived license;
# see assets/LICENSE), which covers the full range the fixtures use.
FONT_PATH = Path(__file__).parent / "assets" / "DejaVuSans.ttf"


# ===========================================================================
# Schema — the authored ground truth, loaded from JSON
# ===========================================================================


class SyntheticHolder(BaseModel):
    """A single (human, position) pair the page states and we expect back."""

    human: str
    position: str


class SyntheticPage(BaseModel):
    """One authored page: the org, the holders named in each channel, and any
    distractor HTML that must *not* be extracted.

    ``id`` is the JSON filename stem (the file *is* the fixture).
    ``text_holders`` is rendered to HTML → the plaintext the model reads;
    ``screenshot_holders`` is rasterized to a PNG, attached iff non-empty. The
    answer key is the union of both — everyone the page states in either
    channel. ``extra_html`` is distractor noise. No mode switch: the pipeline
    path falls out of whether ``screenshot_holders`` is empty.
    """

    id: str
    organization: str
    text_holders: list[SyntheticHolder] = Field(default_factory=list)
    screenshot_holders: list[SyntheticHolder] = Field(default_factory=list)
    extra_html: str = ""


# ===========================================================================
# Rendering — one layout (a roster table) → HTML → plaintext
# ===========================================================================


def render_html(page: SyntheticPage) -> str:
    """Render ``text_holders`` to standalone HTML through one roster layout
    (a Name/Position table, or just the header when empty). The model reads
    the derived plaintext, so only the data varies between fixtures.
    """
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{escape(page.organization)}</title></head><body>",
        f"<h1>{escape(page.organization)}</h1>",
    ]
    if page.text_holders:
        parts.append(
            "<table><thead><tr><th>Name</th><th>Position</th></tr></thead><tbody>"
        )
        for h in page.text_holders:
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


def render_screenshot(page: SyntheticPage) -> bytes:
    """Rasterize ``screenshot_holders`` to a roster PNG.

    A mini-renderer for the one roster layout — not a general HTML engine. Two
    passes: measure to size the canvas, then draw. Page-driven (not
    HTML-driven) so the screenshot can show the full roster regardless of what
    ``text_holders`` the text channel carries. Drives the image path of the
    pipeline without a headless browser.
    """
    font = ImageFont.truetype(str(FONT_PATH), 24)
    head_font = ImageFont.truetype(str(FONT_PATH), 32)
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

    # Drawn from screenshot_holders (not the rendered HTML), so the screenshot
    # can carry the full roster regardless of text_holders.
    head = page.organization
    rows = [(h.human, h.position) for h in page.screenshot_holders]
    distractors: list[str] = []
    if page.extra_html:
        extra = BeautifulSoup(page.extra_html, "html.parser")
        for child in extra.children:
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


def _shape(page: SyntheticPage) -> str:
    """Table label for which pipeline path a fixture drives, derived from the
    two holder lists (not stored): text / image / partial."""
    if not page.screenshot_holders:
        return "text"
    if not page.text_holders:
        return "image"
    return "partial"


def run(fixtures: list[SyntheticPage], verbose: bool) -> None:
    """Render → extract → score each fixture, then print a summary table."""
    rows: list[tuple[SyntheticPage, set, set, int, int, int]] = []
    for page in fixtures:
        # Text always carries text_holders (empty = thin text); the screenshot
        # is attached iff screenshot_holders is non-empty. The two lists cover
        # every path with no mode switch.
        text = html_to_text(render_html(page))
        screenshot_blob = render_screenshot(page) if page.screenshot_holders else None
        extraction = extract(text, screenshot_blob)

        expected = {(h.human, h.position) for h in page.text_holders} | {
            (h.human, h.position) for h in page.screenshot_holders
        }
        got = {(h.human, h.position) for h in extraction.holders}
        tp, fp, fn = score_page(expected, got)
        rows.append((page, expected, got, tp, fp, fn))

        log.info("%s: %d holder(s)", page.id, len(extraction.holders))

    # ---- per-fixture table ------------------------------------------------
    print(file=sys.stderr)
    print(
        f"{'fixture':20} {'path':10} {'TP':>3} {'FP':>3} {'FN':>3}  notes",
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
            f"{page.id:20} {_shape(page):10} {tp:3d} {fp:3d} {fn:3d}  {', '.join(notes)}",
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
