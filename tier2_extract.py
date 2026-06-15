"""Tier 2 extraction: for the pages tier 1 missed, re-run with the full-page
screenshot added to the text.

Reads misses from data/tier1_misses.jsonl (produced by tier 1). The screenshot
catches names that the rendered text didn't carry (JS shells, text baked into
images). Results land in data/tier2.jsonl.
"""

import asyncio
import logging
from pathlib import Path

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text_and_image
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl

log = logging.getLogger(__name__)

TIER1_MISSES_PATH = Path("data/tier1_misses.jsonl")
OUT_PATH = Path("data/tier2.jsonl")
CONCURRENCY = 3  # images are token-heavy; keep this lower than tier 1


def requires(snapshot: dict) -> str | None:
    return None if pravda.content(snapshot, pravda.SCREENSHOT) else "no_screenshot"


async def extract(snapshot: dict) -> dict:
    text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
    screenshot = pravda.read_blob(pravda.content(snapshot, pravda.SCREENSHOT)["path"])
    extraction, usage = await extract_from_text_and_image(text, screenshot)
    holders = [holder.model_dump() for holder in extraction.holders]
    status, reason = ("hit", None) if holders else ("miss", "no_holders")
    return {"status": status, "reason": reason, "holders": holders, "usage": usage}


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    misses = read_jsonl(TIER1_MISSES_PATH)
    log.info("%d tier-1 miss(es) to retry with screenshot", len(misses))
    if not misses:
        return

    results = await run_batch(
        [record["url"] for record in misses],
        OUT_PATH,
        requires=requires,
        extract=extract,
        concurrency=CONCURRENCY,
        timeout=120,
        snapshot_by_url={r["url"]: r["snapshot"] for r in misses},
    )

    rescued = sum(1 for record in results if record["status"] == "hit")
    still_miss = len(results) - rescued
    log.info("%d rescued, %d still miss → %s", rescued, still_miss, OUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
