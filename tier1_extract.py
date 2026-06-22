"""Tier 1 extraction: feed non-empty snapshot text to the LLM and record the
position holders it finds.

Reads records from data/tier0.jsonl. Results land in two files:
  - data/tier1.jsonl: every record (hits + misses), layered on top of tier 0.
  - data/tier1_misses.jsonl: the tier-0 records that missed, for tier 2.

Each record:
  {url, snapshot_id, captured_at, plaintext, screenshot,
   model, status, reason, holders, usage}
"""

import asyncio
import logging
from pathlib import Path

import click

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def requires(record: dict) -> str | None:
    text = pravda.read_text(record.get("plaintext"))
    return None if text.strip() else "no_text"


async def extract(record: dict) -> dict:
    text = pravda.read_text(record.get("plaintext"))
    extraction, usage = await extract_from_text(text)
    holders = [holder.model_dump() for holder in extraction.holders]
    status = "hit" if holders else "miss"
    reason = None if holders else "no_holders"
    return {"status": status, "reason": reason, "holders": holders, "usage": usage}


async def run(
    tier0_path: Path, out_path: Path, misses_path: Path, concurrency: int
) -> None:
    tier0 = read_jsonl(tier0_path)
    log.info("%d tier-0 record(s) to extract", len(tier0))
    if not tier0:
        return

    tier0_by_url = {r["url"]: r for r in tier0}
    results = await run_batch(
        tier0,
        out_path,
        requires=requires,
        extract=extract,
        concurrency=concurrency,
    )

    hits = [record for record in results if record["status"] == "hit"]
    misses = [record for record in results if record["status"] == "miss"]
    log.info("%d hit, %d miss → %s", len(hits), len(misses), out_path)

    # Pass the tier-0 records for misses on to tier 2 (eyes-only retry).
    if misses:
        miss_records = [tier0_by_url[r["url"]] for r in misses]
        write_jsonl(misses_path, miss_records)
        log.info("Wrote %d miss(es) → %s", len(miss_records), misses_path)
    reasons: dict[str, int] = {}
    for record in misses:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        log.info("  miss/%s: %d", reason, count)


@click.command(help=__doc__)
@click.option(
    "-i",
    "--tier0-path",
    type=click.Path(exists=True),
    default="data/tier0.jsonl",
    help="Input tier-0 JSONL path.",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/tier1.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-m",
    "--misses-path",
    type=click.Path(),
    default="data/tier1_misses.jsonl",
    help="Misses output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=5, help="Max concurrent LLM requests."
)
def main(tier0_path: str, out_path: str, misses_path: str, concurrency: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(run(Path(tier0_path), Path(out_path), Path(misses_path), concurrency))


if __name__ == "__main__":
    main()
