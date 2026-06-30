"""Score kolkhoz extractions against a golden sample.

The golden set (`sample_golden.py`) is one row per (page, human, position)
that OpenSanctions recorded for the page. The extraction pipeline writes one
`Holder(human, position)` per latest Extraction per Page. This script joins
the two on the page URL and scores them at the (human, position)-pair level.

Matching is exact string equality. No normalization, no fuzzy matching: the
golden side and the extraction side read different copies of the same page, so
the question we want to answer is whether the model read the names and titles
faithfully, not whether two near-identical strings should count as the same.

The position on an extracted pair is `holder.position`, falling back to the
input-CSV position on the Page when the extractor left it null — the same
rule `kolkhoz.py build_ftm_entities` applies at export time, so a pair scored
here is exactly the pair that would be emitted.

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

    Mirrors the `latest` subquery in `kolkhoz.py build_ftm_entities`: only the
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


def score_page(golden: set, extracted: set) -> dict:
    tp = len(golden & extracted)
    fp = len(extracted - golden)
    fn = len(golden - extracted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_pairs": sorted(extracted - golden),
        "fn_pairs": sorted(golden - extracted),
    }


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

    per_page: dict[str, dict] = {}
    for page in pages:
        per_page[page] = score_page(
            golden_by_page.get(page, set()), extracted_by_page.get(page, set())
        )

    # Micro: pool TP/FP/FN across pages.
    tp = sum(p["tp"] for p in per_page.values())
    fp = sum(p["fp"] for p in per_page.values())
    fn = sum(p["fn"] for p in per_page.values())
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    # Macro: mean of per-page F1 (only pages that appear on at least one side).
    macro_p = sum(p["precision"] for p in per_page.values()) / len(per_page)
    macro_r = sum(p["recall"] for p in per_page.values()) / len(per_page)
    macro_f1 = sum(p["f1"] for p in per_page.values()) / len(per_page)

    print(f"\n{len(pages)} page(s) scored", file=sys.stderr)
    print("          precision   recall     F1", file=sys.stderr)
    print(f"micro     {micro_p:8.3f}  {micro_r:8.3f}  {micro_f1:8.3f}", file=sys.stderr)
    print(f"macro     {macro_p:8.3f}  {macro_r:8.3f}  {macro_f1:8.3f}", file=sys.stderr)
    print(f"          (TP={tp} FP={fp} FN={fn})", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "page",
                "golden",
                "extracted",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "missed",
                "extra",
            ]
        )
        for page in pages:
            p = per_page[page]
            writer.writerow(
                [
                    page,
                    len(golden_by_page.get(page, set())),
                    len(extracted_by_page.get(page, set())),
                    p["tp"],
                    p["fp"],
                    p["fn"],
                    f"{p['precision']:.3f}",
                    f"{p['recall']:.3f}",
                    f"{p['f1']:.3f}",
                    "; ".join(f"{h} ({pos})" for h, pos in p["fn_pairs"]),
                    "; ".join(f"{h} ({pos})" for h, pos in p["fp_pairs"]),
                ]
            )
    print(f"\nwrote per-page eval → {out}", file=sys.stderr)


if __name__ == "__main__":
    cli()
