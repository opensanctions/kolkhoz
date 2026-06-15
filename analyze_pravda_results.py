"""Analyze Pravda snapshot results for all URLs in the CSV."""

import asyncio
import csv
from collections import defaultdict
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

PRAVDA_URL = os.environ["PRAVDA_URL"]
DEFAULT_CSV = Path.home() / "Documents" / "hio_leadership.csv"
CONCURRENCY = 10


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted(
            {row["pep_url"].strip() for row in reader if row["pep_url"].strip()}
        )


async def check_snapshot(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(f"{PRAVDA_URL}/snapshots", params={"url": url})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "items": [], "total": 0}


async def run(csv_path: str) -> None:
    urls = load_urls(csv_path)
    print(f"Loaded {len(urls)} unique URLs from {csv_path}\n")

    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited_check(client: httpx.AsyncClient, url: str) -> dict:
        async with sem:
            return await check_snapshot(client, url)

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(*[limited_check(client, u) for u in urls])

    total_urls = len(urls)
    ok_urls = 0
    error_urls = 0
    connection_errors = 0
    total_snapshots = 0
    http_statuses = defaultdict(int)
    content_types = defaultdict(int)
    missing_content = defaultdict(int)
    artifact_counts = defaultdict(int)

    for url, result in zip(urls, results):
        if "error" in result and not result.get("items"):
            connection_errors += 1
            continue

        n = result.get("total", 0)
        if n == 0:
            error_urls += 1
            continue

        total_snapshots += n
        url_has_200 = False
        for item in result.get("items", []):
            status = item.get("http_status")
            if status:
                http_statuses[status] += 1
                if status == 200:
                    url_has_200 = True

            contents = item.get("contents", [])
            artifact_counts[len(contents)] += 1
            seen = set()
            for c in contents:
                ct = c.get("content_type", "unknown")
                content_types[ct] += 1
                seen.add(ct)
            for missing in {
                "multipart/related",
                "image/png",
                "text/html",
                "text/plain",
            } - seen:
                missing_content[missing] += 1

        if url_has_200:
            ok_urls += 1

    # Summary
    print("=" * 60)
    print("PRAVDA SNAPSHOT ANALYSIS")
    print("=" * 60)
    print(f"\nURLs:           {total_urls}")
    print(f"Snapshots:      {total_snapshots}")
    print(f"200 OK:         {ok_urls} ({ok_urls / total_urls * 100:.1f}%)")
    print(f"Non-200:        {total_urls - ok_urls - connection_errors}")
    print(f"No snapshots:   {error_urls}")
    print(f"Conn errors:    {connection_errors}")

    print("\nHTTP Status Codes:")
    for status, count in sorted(http_statuses.items()):
        print(f"  {status}: {count}")

    print(f"\n{'=' * 60}")
    print("CONTENT TYPES")
    print(f"{'=' * 60}")
    print("\nArtifacts per snapshot:")
    for n in sorted(artifact_counts):
        print(f"  {n}: {artifact_counts[n]}")
    print(f"\nStored (of {total_snapshots} snapshots):")
    for ct in sorted(content_types):
        print(f"  {ct}: {content_types[ct]}")
    print("\nMissing:")
    for ct in sorted(missing_content):
        print(f"  {ct}: {missing_content[ct]}")


@click.command(help=__doc__)
@click.argument("csv_path", type=click.Path(exists=True), default=str(DEFAULT_CSV))
def main(csv_path: str) -> None:
    asyncio.run(run(csv_path))


if __name__ == "__main__":
    main()
