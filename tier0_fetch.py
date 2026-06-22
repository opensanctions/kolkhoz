"""Tier 0 fetch: cache the latest Pravda snapshot for each URL.

Reads a CSV of URLs, fetches the latest Pravda snapshot for each, and writes
the snapshots to data/tier0.jsonl. No filtering, no LLM calls — this is just a
snapshot cache so extraction tiers can be re-run without re-hitting Pravda.

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

        return {
            "url": url,
            "snapshot_id": snapshot["id"],
            "captured_at": snapshot["captured_at"],
            "plaintext": snapshot.get("plaintext"),
            "screenshot": snapshot.get("screenshot"),
        }


async def run(
    csv_path: str, sample: int | None, out_path: Path, concurrency: int
) -> None:
    by_url = {r["url"]: r for r in read_jsonl(out_path)}

    urls = load_urls(csv_path)
    if sample is not None and sample < len(urls):
        urls = random.sample(urls, sample)

    new_urls = [u for u in urls if u not in by_url]
    print(f"Processing {len(urls)} URL(s) ({len(new_urls)} new)")

    results: list[dict | None] = []
    if new_urls:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *[fetch_snapshot(client, url, sem) for url in new_urls]
            )
        for record in results:
            if record is not None:
                by_url[record["url"]] = record
        write_jsonl(out_path, by_url.values())

    kept = sum(1 for r in results if r is not None)
    print(f"  {kept} kept, {len(new_urls) - kept} skipped → {out_path}")


@click.command(help=__doc__)
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-n", "--sample", type=int, default=None, help="Randomly sample N URLs.")
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/tier0.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=10, help="Max concurrent Pravda requests."
)
def main(csv_path: str, sample: int | None, out_path: str, concurrency: int) -> None:
    asyncio.run(run(csv_path, sample, Path(out_path), concurrency))


if __name__ == "__main__":
    main()
