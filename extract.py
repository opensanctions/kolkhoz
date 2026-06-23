"""Single agentic extraction: feed page text (and optionally the
screenshot via a tool) to the LLM and record the position holders it finds.

Reads records from data/fetched.jsonl. The model always receives the rendered
page text; if the text is insufficient it can call ``get_screenshot`` to pull
overlapping tiles of the full-page screenshot. Each record is also classified
by page type (roster / profile / other).

Output: data/extracted.jsonl — every record layered on the fetched snapshot:
  {url, snapshot_id, captured_at, plaintext, screenshot,
   model, status, reason, page_type, holders, looked_at, provenance}
"""

import asyncio
import logging
from pathlib import Path

import click

from kolkhoz import extract as kolkhoz_extract
from kolkhoz import pravda
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl

log = logging.getLogger(__name__)


def requires(record: dict) -> str | None:
    """Return a miss reason when the page has neither usable text nor screenshot."""
    text = pravda.read_text(record.get("plaintext"))
    shot_path = record.get("screenshot")
    screenshot_available = bool(shot_path) and not pravda.is_blank(
        pravda.read_blob(shot_path)
    )
    if not text.strip() and not screenshot_available:
        return "no_content"
    return None


async def extract(record: dict) -> dict:
    text = pravda.read_text(record.get("plaintext"))
    shot_path = record.get("screenshot")
    screenshot_blob = pravda.read_blob(shot_path) if shot_path else None
    if screenshot_blob is not None and pravda.is_blank(screenshot_blob):
        screenshot_blob = None

    extraction, provenance = await kolkhoz_extract.extract(text, screenshot_blob)
    holders = [holder.model_dump() for holder in extraction.holders]
    status = "hit" if holders else "miss"
    reason = None if holders else "no_holders"
    return {
        "status": status,
        "reason": reason,
        "page_type": extraction.page_type.value,
        "holders": holders,
        "provenance": provenance,
    }


async def run(fetched_path: Path, out_path: Path, concurrency: int) -> None:
    records = read_jsonl(fetched_path)
    log.info("%d fetched record(s) to extract", len(records))
    if not records:
        return

    results = await run_batch(
        records,
        out_path,
        requires=requires,
        extract=extract,
        concurrency=concurrency,
    )

    hits = [r for r in results if r["status"] == "hit"]
    misses = [r for r in results if r["status"] == "miss"]
    log.info("%d hit, %d miss → %s", len(hits), len(misses), out_path)

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

    # How many records pulled the screenshot (a get_screenshot function_call
    # appears in any of the per-round response dumps)
    n_pulled = sum(
        1
        for r in results
        if any(
            o.get("type") == "function_call" and o.get("name") == "get_screenshot"
            for resp in (r.get("provenance") or [])
            for o in resp.get("output", [])
        )
    )
    log.info("screenshot pulled: %d/%d", n_pulled, len(results))

    # Miss reason breakdown
    reasons: dict[str, int] = {}
    for r in misses:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        log.info("  miss/%s: %d", reason, count)


@click.command(help=__doc__)
@click.option(
    "-i",
    "--fetched-path",
    type=click.Path(exists=True),
    default="data/fetched.jsonl",
    help="Input fetched-records JSONL path.",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/extracted.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=5, help="Max concurrent LLM requests."
)
def main(fetched_path: str, out_path: str, concurrency: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(run(Path(fetched_path), Path(out_path), concurrency))


if __name__ == "__main__":
    main()
