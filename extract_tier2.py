"""Tier 2 extraction: for the pages tier 1 missed, re-run with the full-page
screenshot added to the text.

Reads only what tier 1 left behind locally (data/tier1.jsonl) — no CSV, no
sampling. The screenshot catches names that the rendered text didn't carry
(JS shells, text baked into images). Results land in data/tier2.jsonl, same
shape as tier 1, and the file doubles as the cache (keyed by text hash).
"""

import asyncio
from pathlib import Path

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text_and_image
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl

TIER1_PATH = Path("data/tier1.jsonl")
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
    tier1 = read_jsonl(TIER1_PATH)
    misses = [record for record in tier1 if record["status"] == "miss"]
    print(f"{len(misses)} tier-1 miss(es) to retry with screenshot")
    if not misses:
        return

    results = await run_batch(
        [record["url"] for record in misses],
        OUT_PATH,
        requires=requires,
        extract=extract,
        concurrency=CONCURRENCY,
        timeout=120,
    )

    rescued = sum(1 for record in results if record["status"] == "hit")
    print(f"  {rescued} rescued, {len(results) - rescued} still miss → {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
