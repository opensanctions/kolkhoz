"""Extract political position holders from web pages.

Reads a CSV of URLs, fetches the latest Pravda snapshot for each, runs an
LLM extraction step, and writes the results to data/extracted.jsonl.

The output file doubles as cache: a URL already present is reused without a
new Pravda lookup or LLM call.
"""

import asyncio
import base64
import csv
import io
import json
import logging
import os
import random
from enum import Enum
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pravda read helpers — read captured blobs straight off Pravda's
# content-addressed storage (the API returns file paths; we read them
# directly, as Pravda intends).
# ---------------------------------------------------------------------------


async def latest_snapshot(client: httpx.AsyncClient, url: str) -> dict | None:
    """Return the most recent snapshot for *url*, or None if there are none."""
    resp = await client.get(
        f"{os.environ['PRAVDA_URL']}/snapshots", params={"url": url}
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return items[0] if items else None  # the API returns newest first


def read_blob(path: str) -> bytes:
    """Read a Pravda blob directly from shared storage."""
    return Path(path).read_bytes()


def read_text(path: str | None) -> str:
    """Decode the plaintext blob at *path*, or "" if none was captured."""
    if not path:
        return ""
    return read_blob(path).decode("utf-8", errors="replace")


def is_blank(blob: bytes) -> bool:
    """True if the image is a single solid colour (a blank or failed render).

    ``getcolors(1)`` returns a list iff the image has at most one distinct
    colour, else None — so a blank white/black/any-colour page reads as blank.
    """
    image = Image.open(io.BytesIO(blob))
    return image.getcolors(1) is not None


def split_image(blob: bytes, tile: int, overlap: float) -> list[bytes]:
    """Slice an image into *overlap*-fraction overlapping *tile*-px tall strips.

    Screenshots are hardclipped for width, so only the height axis ever needs
    slicing: each strip keeps the full width. Strips are laid out on a stride
    of ``tile * (1 - overlap)``, with a shorter remainder strip at the end if
    needed. Images no taller than *tile* come back as a single strip.
    """
    image = Image.open(io.BytesIO(blob))
    width, height = image.size

    def spans(size: int) -> list[tuple[int, int]]:
        if size <= tile:
            return [(0, size)]
        stride = round(tile * (1 - overlap))
        result: list[tuple[int, int]] = []
        start = 0
        while start + tile <= size:
            result.append((start, start + tile))
            start += stride
        if start < size:
            result.append((start, size))
        return result

    tiles: list[bytes] = []
    for top, bottom in spans(height):
        buf = io.BytesIO()
        image.crop((0, top, width, bottom)).save(buf, format="PNG")
        tiles.append(buf.getvalue())
    return tiles


# ---------------------------------------------------------------------------
# LLM extraction — always sends the rendered page *text* to the model, and
# exposes a ``get_screenshot`` function tool so the model can pull the
# full-page screenshot (tiled into overlapping squares) only when the text is
# insufficient. Uses the OpenAI Responses API with structured outputs and low
# reasoning effort.
# ---------------------------------------------------------------------------

client = AsyncOpenAI()

REASONING_EFFORT = "low"
MAX_ROUNDS = 2

INSTRUCTIONS = """\
You extract political position holders from the content of a single web page.

A "holder" is a specific, named human who holds a named position (office, seat,
title, or role) at the organisation the page is about — e.g. a council member,
board director, judge, minister, or chair.

Rules:
- Return one entry per (human, position) pair. If one person holds two
  positions, return two entries.
- Use the person's full name exactly as written on the page.
- Use the most specific position title shown. If the page is about an
  organisation and the title omits it, you may name the body (e.g. "Council
  Member").
- Do not invent people. Only extract humans actually named on the page.
- Ignore names that are not position holders (authors, contacts, mentions).
- If the page names no position holders, return an empty list.
- First, classify the page as `roster`, `profile`, or `other` (see field
  descriptions).
- The page TEXT is given to you. The full-page SCREENSHOT is NOT included by
  default. If the text is insufficient to read the holders — names appear to
  be in images, the page is JS-rendered, or the text is too thin even to tell
  what kind of page it is — call `get_screenshot` and then extract. Otherwise
  do not call it; answer from the text.
- `page_type=other` should be used for generic pages (about, contact, article,
  landing) that are not in the business of listing position holders.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_screenshot",
        "description": (
            "Return the full-page screenshot of this page, tiled into overlapping squares. "
            "Call this ONLY when the text alone is not enough to read the position holders — "
            "for example the names seem to be in images, a JS-rendered org chart, or the text "
            "is too thin to tell what the page is. Do NOT call it if the text already names "
            "the holders, or if the page clearly has no roster."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


class PageType(str, Enum):
    roster = "roster"  # page that lists named position holders (board, council, staff, directory)
    profile = "profile"  # a single person's bio / CV / appointment page
    other = "other"  # about/contact/landing/article — not expected to list holders


class Holder(BaseModel):
    human: str = Field(
        description="Full name of the person, exactly as written on the page."
    )
    position: str = Field(
        description="Specific position title the person holds, e.g. 'Council Member'."
    )


class Extraction(BaseModel):
    page_type: PageType = Field(description="The kind of page this is.")
    holders: list[Holder] = Field(
        description="Position holders found on the page. Empty if none."
    )


async def extract(
    text: str, screenshot_blob: bytes | None
) -> tuple[Extraction, list[dict]]:
    """Extract holders from one page.

    Always sends *text*. The model may pull the *screenshot* via the
    ``get_screenshot`` tool.

    Returns (extraction, provenance) where provenance is the full dump of
    every model response, one per round (id, model, token usage, reasoning,
    function calls, and the final parsed answer all live in there). Whether
    the model pulled the screenshot is recoverable from the function_call
    items in those dumps.
    """
    tile = int(os.environ["IMAGE_TILE_SIZE"])
    overlap = float(os.environ["IMAGE_TILE_OVERLAP"])

    input_items = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    provenance: list[dict] = []

    for _ in range(MAX_ROUNDS):
        response = await client.responses.parse(
            model=os.environ["OPENAI_MODEL"],
            instructions=INSTRUCTIONS,
            input=input_items,
            tools=TOOLS,
            text_format=Extraction,
            reasoning={"effort": REASONING_EFFORT},
        )
        provenance.append(response.model_dump(mode="json"))

        # Does the model want the screenshot?
        tool_calls = [
            o
            for o in response.output
            if o.type == "function_call" and o.name == "get_screenshot"
        ]
        if not tool_calls:
            # No tool call -> final structured answer.
            if response.output_parsed is None:
                raise ValueError(
                    f"Model returned no parsed output (possible refusal): {response.output}"
                )
            return response.output_parsed, provenance

        # It asked for the screenshot. Append the assistant's call(s) + our
        # tool result, loop again.
        if screenshot_blob is None:
            # Can't fulfil the request. Tell the model so it can answer from
            # text alone.
            out_content = "No screenshot is available for this page."
        else:
            tiles = split_image(screenshot_blob, tile, overlap)
            out_content = [
                {
                    "type": "input_text",
                    "text": (
                        f"The {len(tiles)} image(s) below are overlapping tiles "
                        "of a single full-page screenshot of the same page. Read "
                        "them together as one page. Tiles overlap, so the same "
                        "person/position may recur — extract each holder once."
                    ),
                },
                *[
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(t).decode(),
                    }
                    for t in tiles
                ],
            ]
        input_items += (
            response.output
        )  # echoes the assistant's function_call item(s) back
        for tc in tool_calls:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": out_content,
                }
            )

    raise RuntimeError("extraction loop exhausted without a final answer")


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
    text = read_text(record.get("plaintext"))
    shot_path = record.get("screenshot")
    screenshot_available = bool(shot_path) and not is_blank(read_blob(shot_path))
    if not text.strip() and not screenshot_available:
        return "no_content"
    return None


async def fetch_snapshot(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    async with sem:
        snapshot = await latest_snapshot(client, url)
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

        text = read_text(record.get("plaintext"))
        shot_path = record.get("screenshot")
        screenshot_blob = read_blob(shot_path) if shot_path else None
        if screenshot_blob is not None and is_blank(screenshot_blob):
            screenshot_blob = None

        log.info("%s → extracting …", record["url"])
        extraction, provenance = await extract(text, screenshot_blob)
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
