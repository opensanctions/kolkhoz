"""Snapshot all pep_urls from a CSV through Pravda."""

import asyncio
import csv

import click

from snapshot_url import async_snapshot_url, format_snapshot


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted({row["pep_url"] for row in reader if row["pep_url"].strip()})


async def run(csv_path: str, concurrency: int) -> None:
    urls = load_urls(csv_path)
    print(f"Found {len(urls)} unique URLs in {csv_path}")

    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async def limited_snapshot(url: str) -> dict:
        async with sem:
            return await async_snapshot_url(url)

    for url in urls:
        tasks.append(asyncio.create_task(limited_snapshot(url)))

    for task in asyncio.as_completed(tasks):
        data = await task
        print(format_snapshot(data))


@click.command(help=__doc__)
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=5,
    help="Max concurrent requests to Pravda.",
)
def main(csv_path: str, concurrency: int) -> None:
    asyncio.run(run(csv_path, concurrency))


if __name__ == "__main__":
    main()
