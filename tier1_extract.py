"""Tier 1 extraction: feed each URL's latest Pravda snapshot text to the LLM
and record the position holders it finds.

Reads URLs from data/tier0.jsonl (pass-only). Results land in two files:
  - data/tier1.jsonl: hits (extracted holders). Also serves as the cache.
  - data/tier1_misses.jsonl: misses with full snapshot data, for tier 2.

Each hit record:
  {url, snapshot_id, text_hash, model, prompt_version,
   holders: [{human, position}], usage}

Each miss record:
  {url, snapshot_id, snapshot}
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


def requires(snapshot: dict) -> str | None:
    return None if pravda.content(snapshot, pravda.TEXT) else "no_text"


async def extract(snapshot: dict) -> dict:
    text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
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

    snapshot_by_url = {r["url"]: r["snapshot"] for r in tier0}
    results = await run_batch(
        [record["url"] for record in tier0],
        out_path,
        requires=requires,
        extract=extract,
        concurrency=concurrency,
        timeout=60,
        snapshot_by_url=snapshot_by_url,
    )

    hits = [record for record in results if record["status"] == "hit"]
    misses = [record for record in results if record["status"] == "miss"]
    log.info("%d hit, %d miss → %s", len(hits), len(misses), out_path)

    # Write misses with snapshot data for tier 2
    if misses:
        miss_records = [
            {
                "url": r["url"],
                "snapshot_id": r["snapshot_id"],
                "snapshot": snapshot_by_url[r["url"]],
            }
            for r in misses
        ]
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
