"""Sample a balanced subset of the golden set for evaluation.

The golden set is overwhelmingly single-holder profile pages (~89%) with a
long tail of rosters. A plain random sample would measure kolkhoz almost
entirely on profiles. This script groups pages by how many distinct holders
they carry and samples evenly across buckets, so an evaluation run sees a
deliberate mix of profiles and rosters.

Buckets (by distinct holder names on the page):
  profile       exactly 1 holder
  small_roster  2–10 holders
  large_roster  11+ holders

Sampling is uniform within a bucket and reproducible via --seed.

Outputs (written under data/, named after --stem):
- {stem}.csv          golden rows for the sampled pages, same schema as
                      golden.csv. Compare a kolkhoz extraction (filtered to
                      those pages) against this.
- {stem}_input.csv    one row per sampled page (institute, position, url) in
                      the format `kolkhoz.py snapshot-csv` consumes, so the
                      sample feeds straight into the pipeline:
                        uv run python kolkhoz.py snapshot-csv data/{stem}_input.csv

    uv run python sample_golden.py                     # 30 pages, 10 per bucket
    uv run python sample_golden.py -n 60 --seed 7
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

DEFAULT_GOLDEN = Path("data/golden.csv")
DEFAULT_STEM = "golden_sample"

# Page buckets by distinct-holder count, in order. A page is a "roster" when
# it lists several people; the boundary between small and large is a
# convenience so a handful of slots always reach the deep directory-style pages.
BUCKETS = ("profile", "small_roster", "large_roster")


def classify(holder_count: int) -> str:
    """Return the bucket name for a page with this many distinct holders."""
    if holder_count <= 1:
        return "profile"
    if holder_count <= 10:
        return "small_roster"
    return "large_roster"


def load_pages(golden_csv: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Read golden.csv, grouping rows by page.

    Returns (rows_by_page, bucket_by_page). A page's bucket is derived from
    its count of distinct holder names.
    """
    rows_by_page: dict[str, list[dict]] = defaultdict(list)
    with open(golden_csv) as f:
        for row in csv.DictReader(f):
            rows_by_page[row["page"]].append(row)

    bucket_by_page: dict[str, str] = {}
    for page, rows in rows_by_page.items():
        holders = {r["human"] for r in rows if r["human"].strip()}
        bucket_by_page[page] = classify(len(holders))
    return rows_by_page, bucket_by_page


def modal_position(rows: list[dict]) -> str:
    """Pick the most common non-empty position among a page's holders.

    A roster lists several roles; the input CSV carries a single position per
    page, so we take the plurality. It is only a fallback label (the extractor
    sets each holder's own position), so exactness doesn't matter — we just
    need a non-empty value that `snapshot-csv` won't drop.
    """
    counts = Counter(r["position"] for r in rows if r["position"].strip())
    # load_pages is only called on golden rows, which always carry at least
    # one non-empty position per page, so counts is non-empty here.
    return counts.most_common(1)[0][0]


def sample_pages(
    bucket_by_page: dict[str, str],
    want: dict[str, int],
    rng: random.Random,
) -> list[str]:
    """Draw up to `want[b]` pages uniformly from each bucket.

    Caps at the bucket's size rather than failing: the large_roster bucket is
    small (~250 pages), so a request for more than it holds is satisfied with
    everything available. Empties a bucket entirely if its want is 0.
    """
    in_bucket: dict[str, list[str]] = {b: [] for b in BUCKETS}
    for page, bucket in bucket_by_page.items():
        in_bucket[bucket].append(page)

    chosen: list[str] = []
    for bucket in BUCKETS:
        pages = sorted(in_bucket[bucket])  # stable input to the shuffle
        rng.shuffle(pages)
        n = min(want.get(bucket, 0), len(pages))
        chosen.extend(pages[:n])
    return chosen


def write_sample_golden(
    rows_by_page: dict[str, list[dict]], pages: list[str], out: Path
) -> None:
    """Write the golden rows for the sampled pages, same schema as golden.csv."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["page", "datasets", "human", "position", "status"]
    written = 0
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for page in sorted(pages):
            for row in sorted(rows_by_page[page], key=lambda r: r["human"] or ""):
                writer.writerow([row[k] for k in fields])
                written += 1
    print(f"wrote {written} golden row(s) across {len(pages)} page(s) → {out}")


def write_sample_input(
    rows_by_page: dict[str, list[dict]], pages: list[str], out: Path
) -> None:
    """Write one input row per sampled page for `kolkhoz.py snapshot-csv`.

    institute = the page's (first) dataset key — a stable identifier for the
               publishing source; we don't have a human publisher name in the
               golden set, and this is what the export joins on.
    position  = the page's modal position (see modal_position).
    url       = the page.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["institute", "position", "url"])
        for page in pages:
            rows = rows_by_page[page]
            dataset = rows[0]["datasets"].split(";")[0]
            writer.writerow([dataset, modal_position(rows), page])
    print(f"wrote {len(pages)} input row(s) → {out}")


@click.command()
@click.option(
    "-g",
    "--golden",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GOLDEN,
    help="Input golden CSV to sample from.",
)
@click.option(
    "-s",
    "--stem",
    type=str,
    default=DEFAULT_STEM,
    help="Output stem: writes {stem}.csv and {stem}_input.csv under data/.",
)
@click.option(
    "-n",
    "--total",
    type=int,
    default=30,
    help="Total pages to sample, split evenly across buckets.",
)
@click.option("--seed", type=int, default=0, help="RNG seed for reproducibility.")
def cli(
    golden: Path,
    stem: str,
    total: int,
    seed: int,
) -> None:
    # Split evenly across buckets; hand any remainder to profiles.
    per, rem = divmod(total, len(BUCKETS))
    want = {b: per for b in BUCKETS}
    want[BUCKETS[0]] += rem

    rows_by_page, bucket_by_page = load_pages(golden)
    available = Counter(bucket_by_page.values())
    print(
        "available pages:",
        ", ".join(f"{b}={available[b]}" for b in BUCKETS),
        file=sys.stderr,
    )

    rng = random.Random(seed)
    pages = sample_pages(bucket_by_page, want, rng)

    drawn = Counter(bucket_by_page[p] for p in pages)
    print(
        "sampled pages:  ",
        ", ".join(f"{b}={drawn[b]}" for b in BUCKETS),
        file=sys.stderr,
    )

    out_dir = Path("data")
    write_sample_golden(rows_by_page, pages, out_dir / f"{stem}.csv")
    write_sample_input(rows_by_page, pages, out_dir / f"{stem}_input.csv")


if __name__ == "__main__":
    cli()
