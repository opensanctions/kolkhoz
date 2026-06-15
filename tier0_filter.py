"""Tier 0 filter: keep only URLs whose latest Pravda snapshot has enough text.

Reads a CSV of URLs, fetches the latest Pravda snapshot for each, and writes
those with enough text content to data/tier0.jsonl. No LLM calls.

The output file doubles as cache: a URL already present is reused without a
new lookup.
"""

import asyncio
import csv
import random
from pathlib import Path

import click
import httpx

from kolkhoz import pravda
from kolkhoz.utils import read_jsonl, write_jsonl

DEFAULT_CSV = Path.home() / "Documents" / "hio_leadership.csv"
OUT_PATH = Path("data/tier0.jsonl")
MIN_TEXT_CHARS = 200
CONCURRENCY = 10


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted(
            {row["pep_url"].strip() for row in reader if row["pep_url"].strip()}
        )


async def fetch_snapshot(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    async with sem:
        snapshot = await pravda.latest_snapshot(client, url)
        if snapshot is None:
            print(f"  skip {url} — no snapshot")
            return None

        text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
        if not text or len(text.strip()) < MIN_TEXT_CHARS:
            preview = (text or "").strip()
            print(
                f"  skip {url} — text too short ({len(text.strip()) if text else 0} chars)"
            )
            if preview:
                print(f"    {preview}")
            return None

        return {
            "url": url,
            "snapshot_id": snapshot["id"],
            "text_hash": pravda.content_hash(snapshot, pravda.TEXT),
            "snapshot": snapshot,
        }


async def run(csv_path: str, sample: int) -> None:
    by_url = {r["url"]: r for r in read_jsonl(OUT_PATH)}

    urls = load_urls(csv_path)
    if sample < len(urls):
        urls = random.sample(urls, sample)

    new_urls = [u for u in urls if u not in by_url]
    print(f"Processing {len(urls)} URL(s) ({len(new_urls)} new)")

    if new_urls:
        sem = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *[fetch_snapshot(client, url, sem) for url in new_urls]
            )
        for record in results:
            if record is not None:
                by_url[record["url"]] = record
        write_jsonl(OUT_PATH, by_url.values())

    kept = sum(1 for r in results if r is not None)
    print(f"  {kept} kept, {len(new_urls) - kept} skipped → {OUT_PATH}")


@click.command(help=__doc__)
@click.argument("csv_path", type=click.Path(exists=True), default=str(DEFAULT_CSV))
@click.option("-n", "--sample", type=int, default=20, help="Randomly sample N URLs.")
def main(csv_path: str, sample: int) -> None:
    asyncio.run(run(csv_path, sample))


if __name__ == "__main__":
    main()
