"""Snapshot all pep_urls from a CSV through Pravda."""

import argparse
import asyncio
import csv

from snapshot_url import async_snapshot_url, format_snapshot

DEFAULT_CSV = "data/hio_leadership.csv"
DEFAULT_CONCURRENCY = 10


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted({row["pep_url"] for row in reader if row["pep_url"].strip()})


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent requests to Pravda (default: %(default)s)",
    )
    args = parser.parse_args()

    urls = load_urls(args.csv_path)
    print(f"Found {len(urls)} unique URLs in {args.csv_path}")

    sem = asyncio.Semaphore(args.concurrency)

    async def limited_snapshot(url: str) -> dict:
        async with sem:
            return await async_snapshot_url(url)

    for coro in asyncio.as_completed(limited_snapshot(url) for url in urls):
        data = await coro
        print(format_snapshot(data))


if __name__ == "__main__":
    asyncio.run(main())
