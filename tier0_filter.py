"""Tier 0 filter: check whether each URL's latest Pravda snapshot has enough
text to be worth LLM extraction.

Reads a CSV of URLs, fetches the latest Pravda snapshot for each, and keeps
only those with enough text content. No LLM calls — just Pravda lookups.

Passing records land in data/tier0.jsonl (without status/reason fields).
The file doubles as cache: a URL already present is reused without a new lookup.
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
MIN_TEXT_CHARS = 200
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

        text_hash = pravda.content_hash(snapshot, pravda.TEXT)
        text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
        status, reason = classify(text)

        return {
            "url": url,
            "snapshot_id": snapshot["id"],
            "text_hash": text_hash,
            "snapshot": snapshot,
            "status": status,
            "reason": reason,
        }


def record_for_output(record: dict) -> dict:
    """Strip classification fields, keep only what goes into tier0.jsonl."""
    return {k: v for k, v in record.items() if k not in ("status", "reason")}


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
    by_url = {r["url"]: r for r in read_jsonl(OUT_PATH)}

    urls = load_urls(args.csv_path)
    if args.sample < len(urls):
        urls = random.sample(urls, args.sample)

    new_urls = [u for u in urls if u not in by_url]
    print(f"Processing {len(urls)} URL(s) ({len(new_urls)} new)")

    results = []
    if new_urls:
        sem = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *[check_url(client, url, sem) for url in new_urls]
            )
        for record in results:
            if record["status"] == "pass":
                by_url[record["url"]] = record_for_output(record)
        write_jsonl(OUT_PATH, by_url.values())

    # Summary for this run's new URLs.
    passes = sum(1 for r in results if r["status"] == "pass")
    misses = [r for r in results if r["status"] == "miss"]
    print(f"  {passes} pass, {len(misses)} miss → {OUT_PATH}")

    reasons: dict[str, int] = {}
    for record in misses:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"    miss/{reason}: {count}")

    for record in misses:
        snapshot = record.get("snapshot")
        if snapshot is None:
            continue
        text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
        if text:
            preview = text.strip()[:500]
            if len(text.strip()) > 500:
                preview += "…"
            print(f"\n    --- {record['url']} ---")
            print(f"    {preview}")


if __name__ == "__main__":
    asyncio.run(main())
