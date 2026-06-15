"""Tier 0 filter: check whether each URL's latest Pravda snapshot has enough
text to be worth LLM extraction.

Reads a CSV of URLs (--csv), optionally samples N of them (--sample), fetches
the latest Pravda snapshot for each, and classifies the text as pass (enough
content) or miss (too short / missing). No LLM calls — just Pravda lookups.

Results land in data/tier0.jsonl, one record per URL. The file doubles as the
cache: a URL already present is reused without a new Pravda lookup.
"""

import argparse
import asyncio
import csv
import random
from pathlib import Path

import httpx

from kolkhoz import pravda
from kolkhoz.utils import read_jsonl, write_jsonl

DEFAULT_CSV = "data/hio_leadership.csv"
OUT_PATH = Path("data/tier0.jsonl")
MIN_TEXT_CHARS = 200  # below this the page is likely a JS shell, not real content
CONCURRENCY = 10
DEFAULT_SAMPLE = 20


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted(
            {row["pep_url"].strip() for row in reader if row["pep_url"].strip()}
        )


def classify(text: str) -> tuple[str, str | None]:
    if not text:
        return "miss", "no_text"
    if len(text.strip()) < MIN_TEXT_CHARS:
        return "miss", "text_too_short"
    return "pass", "text_ok"


async def check_url(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        snapshot = await pravda.latest_snapshot(client, url)
        if snapshot is None:
            return {
                "url": url,
                "snapshot_id": None,
                "text_hash": None,
                "status": "miss",
                "reason": "no_snapshot",
            }

        text_item = pravda.content(snapshot, pravda.TEXT)
        text_hash = pravda.content_hash(snapshot, pravda.TEXT)
        text = pravda.read_text(text_item)
        status, reason = classify(text)

        return {
            "url": url,
            "snapshot_id": snapshot["id"],
            "text_hash": text_hash,
            "snapshot": snapshot,
            "status": status,
            "reason": reason,
        }


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

    # Seed from existing output so repeated runs accumulate.
    by_url = {record["url"]: record for record in read_jsonl(OUT_PATH)}

    urls = load_urls(args.csv_path)
    if args.sample < len(urls):
        urls = random.sample(urls, args.sample)

    # Skip URLs we already checked.
    new_urls = [u for u in urls if u not in by_url]
    print(f"Processing {len(urls)} URL(s) ({len(new_urls)} new)")

    if new_urls:
        sem = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *[check_url(client, url, sem) for url in new_urls]
            )
        for record in results:
            by_url[record["url"]] = record
        write_jsonl(OUT_PATH, by_url.values())

    # Summary for this run's URLs.
    this_run = [by_url[u] for u in urls]
    passes = sum(1 for r in this_run if r["status"] == "pass")
    misses = [r for r in this_run if r["status"] == "miss"]
    print(f"  {passes} pass, {len(misses)} miss → {OUT_PATH}")
    reasons: dict[str, int] = {}
    for record in misses:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"    miss/{reason}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
