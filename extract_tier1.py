"""Tier 1 extraction: feed each URL's latest Pravda snapshot text to the LLM
and record the position holders it finds.

Reads URLs from data/tier0.jsonl (pass-only). Results land in data/tier1.jsonl,
one object per URL, which also serves as the cache: a page whose snapshot text
was already extracted (same text hash, prompt version, model) is reused without
an LLM call. Repeated runs accumulate.

Each record:
  {url, snapshot_id, text_hash, model, prompt_version,
   status: "hit"|"miss", reason, holders: [{human, position}], usage}
"""

import asyncio
from pathlib import Path

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl

TIER0_PATH = Path("data/tier0.jsonl")
OUT_PATH = Path("data/tier1.jsonl")
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
    tier0 = read_jsonl(TIER0_PATH)
    passes = [record for record in tier0 if record["status"] == "pass"]
    print(f"{len(passes)} tier-0 pass(es) to extract")
    if not passes:
        return

    results = await run_batch(
        [record["url"] for record in passes],
        OUT_PATH,
        requires=requires,
        extract=extract,
        concurrency=CONCURRENCY,
        timeout=60,
        snapshot_by_url={r["url"]: r["snapshot"] for r in passes},
    )

    hits = sum(1 for record in results if record["status"] == "hit")
    misses = [record for record in results if record["status"] == "miss"]
    print(f"  {hits} hit, {len(misses)} miss → {OUT_PATH}")
    reasons: dict[str, int] = {}
    for record in misses:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"    miss/{reason}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
