"""Tier 2 extraction: for the pages tier 1 missed, re-run with the full-page
screenshot added to the text.

Reads misses from data/tier1_misses.jsonl (produced by tier 1). The screenshot
catches names that the rendered text didn't carry (JS shells, text baked into
images). Results land in data/tier2.jsonl.
"""

import asyncio
import logging
from pathlib import Path

import click

from kolkhoz import pravda
from kolkhoz.extract import extract_from_text_and_image
from kolkhoz.pipeline import run_batch
from kolkhoz.utils import read_jsonl

log = logging.getLogger(__name__)


def requires(snapshot: dict) -> str | None:
    return None if pravda.content(snapshot, pravda.SCREENSHOT) else "no_screenshot"


async def extract(snapshot: dict) -> dict:
    text = pravda.read_text(pravda.content(snapshot, pravda.TEXT))
    screenshot = pravda.read_blob(pravda.content(snapshot, pravda.SCREENSHOT)["path"])
    extraction, usage = await extract_from_text_and_image(text, screenshot)
    holders = [holder.model_dump() for holder in extraction.holders]
    status, reason = ("hit", None) if holders else ("miss", "no_holders")
    return {"status": status, "reason": reason, "holders": holders, "usage": usage}


async def run(misses_path: Path, out_path: Path, concurrency: int) -> None:
    misses = read_jsonl(misses_path)
    log.info("%d tier-1 miss(es) to retry with screenshot", len(misses))
    if not misses:
        return

    results = await run_batch(
        [record["url"] for record in misses],
        out_path,
        requires=requires,
        extract=extract,
        concurrency=concurrency,
        timeout=120,
        snapshot_by_url={r["url"]: r["snapshot"] for r in misses},
    )

    rescued = sum(1 for record in results if record["status"] == "hit")
    still_miss = len(results) - rescued
    log.info("%d rescued, %d still miss → %s", rescued, still_miss, out_path)


@click.command(help=__doc__)
@click.option(
    "-i",
    "--misses-path",
    type=click.Path(exists=True),
    default="data/tier1_misses.jsonl",
    help="Input tier-1 misses JSONL path.",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(),
    default="data/tier2.jsonl",
    help="Output JSONL path.",
)
@click.option(
    "-c", "--concurrency", type=int, default=3, help="Max concurrent LLM requests."
)
def main(misses_path: str, out_path: str, concurrency: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    asyncio.run(run(Path(misses_path), Path(out_path), concurrency))


if __name__ == "__main__":
    main()
