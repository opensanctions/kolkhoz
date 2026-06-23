"""Stage 1 extraction: feed non-empty snapshot text to the LLM and record the
position holders it finds.

Reads records from data/stage0.jsonl. Results land in two files:
  - data/stage1.jsonl: every record (hits + misses), layered on top of stage 0.
  - data/stage1_misses.jsonl: the stage-0 records that missed, for stage 2.

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
    stage0_path: Path, out_path: Path, misses_path: Path, concurrency: int
) -> None:
    stage0 = read_jsonl(stage0_path)
    log.info("%d stage-0 record(s) to extract", len(stage0))
    if not stage0:
        return

    stage0_by_url = {r["url"]: r for r in stage0}
    results = await run_batch(
        stage0,
        out_path,
        requires=requires,
        extract=extract,
        concurrency=concurrency,
    )

    hits = [record for record in results if record["status"] == "hit"]
    misses = [record for record in results if record["status"] == "miss"]
    log.info("%d hit, %d miss → %s", len(hits), len(misses), out_path)

    # Pass the stage-0 records for misses on to stage 2 (eyes-only retry).
    if misses:
        miss_records = [stage0_by_url[r["url"]] for r in misses]
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
    "--stage0-path",
    type=click.Path(exists=True),
    default="data/stage0.jsonl",
    help="Input stage-0 JSONL path.",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/stage1.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-m",
    "--misses-path",
    type=click.Path(),
    default="data/stage1_misses.jsonl",
    help="Misses output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=5, help="Max concurrent LLM requests."
)
def main(stage0_path: str, out_path: str, misses_path: str, concurrency: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(run(Path(stage0_path), Path(out_path), Path(misses_path), concurrency))


if __name__ == "__main__":
    main()
