"""Score kolkhoz extractions against a golden sample.

The golden set (`golden.py sample`) is one row per (page, human, position)
that OpenSanctions recorded for the page. The extraction pipeline writes one
`Holder(human, position)` per latest Extraction per Page. This script joins
the two on the page URL and scores them.

Matching is exact string equality. No normalization, no fuzzy matching: the
golden side and the extraction side read different copies of the same page, so
the question we want to answer is whether the model read the names and titles
faithfully, not whether two near-identical strings should count as the same.

The position on an extracted pair is `holder.position`, falling back to the
input-CSV position on the Page when the extractor left it null — the same
rule `kolkhoz.py holder_to_row` applies at export time, so a pair scored
here is exactly the pair that would be emitted.

Scores are reported in four dimensions, each as micro/macro precision, recall
and F1:

* **pairs**     — the (human, position) pair, as before. The strict end-to-end
                  metric.
* **count**     — did we find the right *number* of holders? Treats holders as
                  anonymous: TP = min(golden, extracted), the rest is FP/FN.
                  Tells us whether the model over- or under-counts, independent
                  of name/position normalization.
* **human**     — human names only, as sets per page. Isolates name reading
                  from position reading.
* **position**  — position titles only, as sets per page. Isolates position
                  reading from name reading.

Usage:

    uv run python evaluate.py                          # golden_sample.csv vs the
                                                       # golden_sample_input dataset
    uv run python evaluate.py -d golden_sample_input -g data/golden_sample.csv
"""

import csv
import sys
from pathlib import Path

import click
from sqlalchemy import func, select

from kolkhoz import Session
from models import Extraction as ExtractionRow
from models import Holder as HolderRow
from models import Page as PageRow

DEFAULT_GOLDEN = Path("data/golden_sample.csv")
DEFAULT_DATASET = "golden_sample_input"
DEFAULT_OUT = Path("data/golden_sample_eval.csv")


def load_golden(golden_csv: Path) -> dict[str, set[tuple[str, str]]]:
    """Read the golden CSV, keyed by page, into sets of (human, position).

    Position is required: golden rows with a blank position can't form a pair
    and are skipped. Duplicate pairs on a page collapse into one.
    """
    pairs_by_page: dict[str, set[tuple[str, str]]] = {}
    with open(golden_csv) as f:
        for row in csv.DictReader(f):
            human = row["human"].strip()
            position = row["position"].strip()
            if not human or not position:
                continue
            pairs_by_page.setdefault(row["page"], set()).add((human, position))
    return pairs_by_page


def load_extracted(dataset: str) -> dict[str, set[tuple[str, str]]]:
    """Read the latest extraction per page for a dataset, as (human, position) pairs.

    Mirrors the `latest` subquery in `kolkhoz.py build_export_rows`: only the
    most recent extraction of each page is scored. Position is the holder's
    own title, or the Page's input-CSV position when the extractor left it
    null.
    """
    latest = (
        select(
            ExtractionRow.page_id.label("page_id"),
            func.max(ExtractionRow.id).label("extraction_id"),
        )
        .group_by(ExtractionRow.page_id)
        .subquery()
    )
    stmt = (
        select(HolderRow, PageRow)
        .join(latest, HolderRow.extraction_id == latest.c.extraction_id)
        .join(ExtractionRow, HolderRow.extraction_id == ExtractionRow.id)
        .join(PageRow, ExtractionRow.page_id == PageRow.id)
        .where(PageRow.dataset == dataset)
    )
    pairs_by_page: dict[str, set[tuple[str, str]]] = {}
    with Session() as session:
        for holder, page in session.execute(stmt).all():
            position = holder.position or page.position
            pairs_by_page.setdefault(page.url, set()).add((holder.human, position))
    return pairs_by_page


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision / recall / F1 from raw confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def set_confusion(golden: set, extracted: set) -> tuple[int, int, int]:
    """TP/FP/FN for set equality: intersection, extras, misses."""
    tp = len(golden & extracted)
    fp = len(extracted - golden)
    fn = len(golden - extracted)
    return tp, fp, fn


def count_confusion(golden: int, extracted: int) -> tuple[int, int, int]:
    """TP/FP/FN for an anonymous-count match (holders indistinguishable).

    TP is how many holders we can pair up by count; the surplus on either side
    is FP (over-counted) or FN (under-counted).
    """
    tp = min(golden, extracted)
    fp = max(0, extracted - golden)
    fn = max(0, golden - extracted)
    return tp, fp, fn


def report(title: str, page_stats: dict[str, tuple[int, int, int]]) -> None:
    """Print a micro/macro precision-recall-F1 table for one scoring dimension.

    `page_stats` maps each page to its (TP, FP, FN) for that dimension.
    """
    tp = sum(s[0] for s in page_stats.values())
    fp = sum(s[1] for s in page_stats.values())
    fn = sum(s[2] for s in page_stats.values())
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)

    per_page = [prf(*s) for s in page_stats.values()]
    n = len(per_page)
    macro_p = sum(p for p, _, _ in per_page) / n if n else 0.0
    macro_r = sum(r for _, r, _ in per_page) / n if n else 0.0
    macro_f1 = sum(f for _, _, f in per_page) / n if n else 0.0

    print(f"\n{title}", file=sys.stderr)
    print("          precision   recall     F1", file=sys.stderr)
    print(f"micro     {micro_p:8.3f}  {micro_r:8.3f}  {micro_f1:8.3f}", file=sys.stderr)
    print(f"macro     {macro_p:8.3f}  {macro_r:8.3f}  {macro_f1:8.3f}", file=sys.stderr)
    print(f"          (TP={tp} FP={fp} FN={fn})", file=sys.stderr)


@click.command()
@click.option(
    "-g",
    "--golden",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GOLDEN,
    help="Golden sample CSV (page, human, position, ...).",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=DEFAULT_DATASET,
    help="Kolkhoz dataset (PageRow.dataset) to score.",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT,
    help="Per-page eval CSV to write.",
)
def cli(golden: Path, dataset: str, out: Path) -> None:
    golden_by_page = load_golden(golden)
    extracted_by_page = load_extracted(dataset)
    print(
        f"golden: {len(golden_by_page)} page(s); "
        f"extracted dataset {dataset!r}: {len(extracted_by_page)} page(s)",
        file=sys.stderr,
    )

    pages = sorted(golden_by_page.keys() | extracted_by_page.keys())

    pair_stats: dict[str, tuple[int, int, int]] = {}
    count_stats: dict[str, tuple[int, int, int]] = {}
    human_stats: dict[str, tuple[int, int, int]] = {}
    position_stats: dict[str, tuple[int, int, int]] = {}
    per_page: dict[str, dict] = {}

    for page in pages:
        g_pairs = golden_by_page.get(page, set())
        e_pairs = extracted_by_page.get(page, set())

        g_humans = {h for h, _ in g_pairs}
        e_humans = {h for h, _ in e_pairs}
        g_positions = {p for _, p in g_pairs}
        e_positions = {p for _, p in e_pairs}

        pair_stats[page] = set_confusion(g_pairs, e_pairs)
        count_stats[page] = count_confusion(len(g_pairs), len(e_pairs))
        human_stats[page] = set_confusion(g_humans, e_humans)
        position_stats[page] = set_confusion(g_positions, e_positions)

        per_page[page] = {
            "golden": len(g_pairs),
            "extracted": len(e_pairs),
            "pair": prf(*pair_stats[page]),
            "count": prf(*count_stats[page]),
            "human": prf(*human_stats[page]),
            "position": prf(*position_stats[page]),
            "missed_pairs": sorted(g_pairs - e_pairs),
            "extra_pairs": sorted(e_pairs - g_pairs),
            "missed_humans": sorted(g_humans - e_humans),
            "extra_humans": sorted(e_humans - g_humans),
        }

    print(f"\n{len(pages)} page(s) scored", file=sys.stderr)
    report("pairs (human, position)", pair_stats)
    report("count (number of holders)", count_stats)
    report("human names", human_stats)
    report("position names", position_stats)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "page",
                "golden",
                "extracted",
                "pair_p",
                "pair_r",
                "pair_f1",
                "count_p",
                "count_r",
                "count_f1",
                "human_p",
                "human_r",
                "human_f1",
                "position_p",
                "position_r",
                "position_f1",
                "missed_humans",
                "extra_humans",
                "missed_pairs",
                "extra_pairs",
            ]
        )
        for page in pages:
            p = per_page[page]
            writer.writerow(
                [
                    page,
                    p["golden"],
                    p["extracted"],
                    f"{p['pair'][0]:.3f}",
                    f"{p['pair'][1]:.3f}",
                    f"{p['pair'][2]:.3f}",
                    f"{p['count'][0]:.3f}",
                    f"{p['count'][1]:.3f}",
                    f"{p['count'][2]:.3f}",
                    f"{p['human'][0]:.3f}",
                    f"{p['human'][1]:.3f}",
                    f"{p['human'][2]:.3f}",
                    f"{p['position'][0]:.3f}",
                    f"{p['position'][1]:.3f}",
                    f"{p['position'][2]:.3f}",
                    "; ".join(p["missed_humans"]),
                    "; ".join(p["extra_humans"]),
                    "; ".join(f"{h} ({pos})" for h, pos in p["missed_pairs"]),
                    "; ".join(f"{h} ({pos})" for h, pos in p["extra_pairs"]),
                ]
            )
    print(f"\nwrote per-page eval → {out}", file=sys.stderr)


if __name__ == "__main__":
    cli()
