"""Extract political position holders from web pages.

Reads a CSV of URLs, fetches the latest Pravda snapshot for each, runs an
LLM extraction step, and writes the results to data/extracted.jsonl.

The output file doubles as cache: a URL already present is reused without a
new Pravda lookup or LLM call.
"""

import asyncio
import csv
import json
import logging
import os
import random
from pathlib import Path

import click
import httpx

from kolkhoz import extract as kolkhoz_extract
from kolkhoz import pravda

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted(
            {row["pep_url"].strip() for row in reader if row["pep_url"].strip()}
        )


def has_content(record: dict) -> str | None:
    """Return a miss reason when the page has neither usable text nor screenshot."""
    text = pravda.read_text(record.get("plaintext"))
    shot_path = record.get("screenshot")
    screenshot_available = bool(shot_path) and not pravda.is_blank(
        pravda.read_blob(shot_path)
    )
    if not text.strip() and not screenshot_available:
        return "no_content"
    return None


async def fetch_snapshot(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    async with sem:
        snapshot = await pravda.latest_snapshot(client, url)
        if snapshot is None:
            log.info("  skip %s — no snapshot", url)
        return snapshot


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


async def run(
    csv_path: str,
    out_path: Path,
    sample: int | None,
    fetch_concurrency: int,
    extract_concurrency: int,
) -> None:
    # --- Fetch snapshots from Pravda ------------------------------------------
    by_url = {r["url"]: r for r in read_jsonl(out_path)}

    urls = load_urls(csv_path)
    if sample is not None and sample < len(urls):
        urls = random.sample(urls, sample)

    new_urls = [u for u in urls if u not in by_url]
    log.info(
        "%d URL(s) total, %d already done, %d new",
        len(urls),
        len(by_url),
        len(new_urls),
    )

    if new_urls:
        sem = asyncio.Semaphore(fetch_concurrency)
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *[fetch_snapshot(client, url, sem) for url in new_urls]
            )
        kept = sum(1 for r in results if r is not None)
        for record in results:
            if record is not None:
                by_url[record["url"]] = record
        write_jsonl(out_path, by_url.values())
        log.info("fetch: %d kept, %d skipped", kept, len(new_urls) - kept)

    # --- Extract with LLM -----------------------------------------------------
    pending = [r for r in by_url.values() if "status" not in r]
    log.info("%d record(s) to extract", len(pending))

    if not pending:
        return

    sem = asyncio.Semaphore(extract_concurrency)
    results = await asyncio.gather(*[extract_one(record, sem) for record in pending])
    for record in results:
        by_url[record["url"]] = record
    write_jsonl(out_path, by_url.values())
    log.info("wrote %d record(s) → %s", len(by_url), out_path)

    hits = [r for r in results if r["status"] == "hit"]
    misses = [r for r in results if r["status"] == "miss"]
    log.info("extraction: %d hit, %d miss", len(hits), len(misses))

    # Page-type distribution
    pt_counts: dict[str, int] = {}
    for r in results:
        pt = r.get("page_type")
        if pt is not None:
            pt_counts[pt] = pt_counts.get(pt, 0) + 1
    log.info(
        "page_type: roster=%d profile=%d other=%d",
        pt_counts.get("roster", 0),
        pt_counts.get("profile", 0),
        pt_counts.get("other", 0),
    )

    # Miss reason breakdown
    reasons: dict[str, int] = {}
    for r in misses:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        log.info("  miss/%s: %d", reason, count)


async def extract_one(record: dict, sem: asyncio.Semaphore) -> dict:
    """Extract holders from a single fetched record."""
    async with sem:
        out = {**record, "model": os.environ["OPENAI_MODEL"]}

        missing = has_content(record)
        if missing is not None:
            log.info("%s → miss (%s)", record["url"], missing)
            out.update(status="miss", reason=missing, holders=[], provenance=None)
            return out

        text = pravda.read_text(record.get("plaintext"))
        shot_path = record.get("screenshot")
        screenshot_blob = pravda.read_blob(shot_path) if shot_path else None
        if screenshot_blob is not None and pravda.is_blank(screenshot_blob):
            screenshot_blob = None

        log.info("%s → extracting …", record["url"])
        extraction, provenance = await kolkhoz_extract.extract(text, screenshot_blob)
        holders = [holder.model_dump() for holder in extraction.holders]
        status = "hit" if holders else "miss"
        reason = None if holders else "no_holders"

        out.update(
            status=status,
            reason=reason,
            page_type=extraction.page_type.value,
            holders=holders,
            provenance=provenance,
        )
        log.info(
            "%s → %s (%d holder(s))",
            record["url"],
            status,
            len(holders),
        )
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(help=__doc__)
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-n", "--sample", type=int, default=None, help="Randomly sample N URLs.")
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/extracted.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "--fetch-concurrency", type=int, default=10, help="Max concurrent Pravda requests."
)
@click.option(
    "--extract-concurrency", type=int, default=5, help="Max concurrent LLM requests."
)
def main(
    csv_path: str,
    sample: int | None,
    out_path: str,
    fetch_concurrency: int,
    extract_concurrency: int,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(
        run(csv_path, Path(out_path), sample, fetch_concurrency, extract_concurrency)
    )


if __name__ == "__main__":
    main()
