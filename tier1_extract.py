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

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

TIER0_PATH = Path("data/tier0.jsonl")
OUT_PATH = Path("data/tier1.jsonl")
MISSES_PATH = Path("data/tier1_misses.jsonl")
CONCURRENCY = 5


def requires(snapshot: dict) -> str | None:
    return None if pravda.content(snapshot, pravda.TEXT) else "no_text"


async def extract(snapshot: dict) -> dict:
    text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
    extraction, usage = await extract_from_text(text)
    holders = [holder.model_dump() for holder in extraction.holders]
    status = "hit" if holders else "miss"
    reason = None if holders else "no_holders"
    return {"status": status, "reason": reason, "holders": holders, "usage": usage}


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    tier0 = read_jsonl(TIER0_PATH)
    log.info("%d tier-0 record(s) to extract", len(tier0))
    if not tier0:
        return

    snapshot_by_url = {r["url"]: r["snapshot"] for r in tier0}
    results = await run_batch(
        [record["url"] for record in tier0],
        OUT_PATH,
        requires=requires,
        extract=extract,
        concurrency=CONCURRENCY,
        timeout=60,
        snapshot_by_url=snapshot_by_url,
    )

    hits = [record for record in results if record["status"] == "hit"]
    misses = [record for record in results if record["status"] == "miss"]
    log.info("%d hit, %d miss → %s", len(hits), len(misses), OUT_PATH)

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
        write_jsonl(MISSES_PATH, miss_records)
        log.info("Wrote %d miss(es) → %s", len(miss_records), MISSES_PATH)
    reasons: dict[str, int] = {}
    for record in misses:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        log.info("  miss/%s: %d", reason, count)


if __name__ == "__main__":
    asyncio.run(main())
