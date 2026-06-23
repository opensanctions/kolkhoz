"""Single agentic extraction stage: feed page text (and optionally the
screenshot via a tool) to the LLM and record the position holders it finds.

Reads records from data/stage0.jsonl. The model always receives the rendered
page text; if the text is insufficient it can call ``get_screenshot`` to pull
overlapping tiles of the full-page screenshot. Each record is also classified
by page type (roster / profile / other).

Output: data/stage1.jsonl — every record layered on stage 0:
  {url, snapshot_id, captured_at, plaintext, screenshot,
   model, status, reason, page_type, holders, looked_at, usage}
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

    extraction, usage, looked_at = await kolkhoz_extract.extract(text, screenshot_blob)
    holders = [holder.model_dump() for holder in extraction.holders]
    status = "hit" if holders else "miss"
    reason = None if holders else "no_holders"
    return {
        "status": status,
        "reason": reason,
        "page_type": extraction.page_type.value,
        "holders": holders,
        "looked_at": looked_at,
        "usage": usage,
    }


async def run(stage0_path: Path, out_path: Path, concurrency: int) -> None:
    stage0 = read_jsonl(stage0_path)
    log.info("%d stage-0 record(s) to extract", len(stage0))
    if not stage0:
        return

    results = await run_batch(
        stage0,
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

    # How many records pulled the screenshot
    n_pulled = sum(1 for r in results if "screenshot" in r.get("looked_at", []))
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
    "--stage0-path",
    type=click.Path(exists=True),
    default="data/stage0.jsonl",
    help="Input stage-0 JSONL path.",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/stage1.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=5, help="Max concurrent LLM requests."
)
def main(stage0_path: str, out_path: str, concurrency: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(run(Path(stage0_path), Path(out_path), concurrency))


if __name__ == "__main__":
    main()
