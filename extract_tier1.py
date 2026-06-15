"""Tier 1 extraction: feed each URL's latest Pravda snapshot text to the LLM
and record the position holders it finds.

Samples N random URLs from the CSV (--sample) so you can get a feel for results
without running the whole set. Results land in data/tier1.jsonl, one object per
URL, which also serves as the cache: a page whose snapshot text was already
extracted (same text hash, prompt version, model) is reused without an LLM call.
Repeated runs accumulate — rows for URLs you didn't sample this time are kept.

Each record:
  {url, snapshot_id, text_hash, model, prompt_version,
   status: "hit"|"miss", reason, holders: [{human, position}], usage}
"""

import argparse
import asyncio
import csv
import random
from pathlib import Path

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text
from kolkhoz.pipeline import run_batch

DEFAULT_CSV = "data/hio_leadership.csv"
OUT_PATH = Path("data/tier1.jsonl")
CONCURRENCY = 5
DEFAULT_SAMPLE = 20
MIN_TEXT_CHARS = 200  # below this the page is likely a JS shell, not real content


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted(
            {row["pep_url"].strip() for row in reader if row["pep_url"].strip()}
        )


def classify(holders: list[dict], text: str) -> tuple[str, str | None]:
    if holders:
        return "hit", None
    if len(text.strip()) < MIN_TEXT_CHARS:
        return "miss", "text_too_short"  # tier-2 candidate (likely JS shell)
    return "miss", "no_holders"


def requires(snapshot: dict) -> str | None:
    return None if pravda.content(snapshot, pravda.TEXT) else "no_text"


async def extract(snapshot: dict) -> dict:
    text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
    extraction, usage = await extract_from_text(text)
    holders = [holder.model_dump() for holder in extraction.holders]
    status, reason = classify(holders, text)
    return {"status": status, "reason": reason, "holders": holders, "usage": usage}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument(
        "-n",
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help="Randomly sample N URLs (default: %(default)s)",
    )
    args = parser.parse_args()

    urls = load_urls(args.csv_path)
    if args.sample < len(urls):
        urls = random.sample(urls, args.sample)
    print(f"Processing {len(urls)} URL(s)")

    results = await run_batch(
        urls,
        OUT_PATH,
        requires=requires,
        extract=extract,
        concurrency=CONCURRENCY,
        timeout=60,
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
