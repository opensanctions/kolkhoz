"""Kolkhoz — turn raw web pages into structured data about political position holders.

Orchestrates Pravda web capture and LLM extraction into a single CLI:

- ``snapshot-url``   snapshot a single URL through Pravda
- ``snapshot-csv``   snapshot all URLs from a CSV through Pravda
- ``extract``        read a CSV of URLs, fetch the latest Pravda snapshot for
                      each, run an LLM extraction step, and write the results
                      to data/<csv-stem>.extracted.jsonl
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
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()

log = logging.getLogger("kolkhoz")


# ===========================================================================
# Pravda snapshotting
# ===========================================================================


async def async_snapshot_url(url: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{os.environ['PRAVDA_URL']}/snapshots", json={"url": url}
        )
        return resp.json()


# ===========================================================================
# Pravda read helpers — read captured blobs straight off Pravda's
# content-addressed storage (the API returns file paths; we read them
# directly, as Pravda intends).
# ===========================================================================


def latest_snapshot(client: httpx.Client, url: str) -> dict | None:
    """Return the most recent snapshot for *url*, or None if there are none."""
    resp = client.get(f"{os.environ['PRAVDA_URL']}/snapshots", params={"url": url})
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return items[0] if items else None  # the API returns newest first


def read_blob(path: str) -> bytes:
    """Read a Pravda blob directly from shared storage."""
    return Path(path).read_bytes()


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


# ===========================================================================
# LLM extraction — always sends the rendered page *text* to the model, and
# exposes a ``get_screenshot`` function tool so the model can pull the
# full-page screenshot (tiled into overlapping squares) only when the text is
# insufficient. Uses the OpenAI Responses API with structured outputs and low
# reasoning effort.
# ===========================================================================

client = OpenAI()

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
    position: str | None = Field(
        default=None,
        description=(
            "Specific position title the person holds, e.g. 'Council Member'. "
            "Omit (null) only if the page names the person but states no specific "
            "title for them."
        ),
    )


class Extraction(BaseModel):
    page_type: PageType = Field(description="The kind of page this is.")
    holders: list[Holder] = Field(
        description="Position holders found on the page. Empty if none."
    )


def extract(text: str, screenshot_blob: bytes | None) -> tuple[Extraction, list[dict]]:
    """Extract holders from one page.

    Always sends *text*. The model may pull the *screenshot* via the
    ``get_screenshot`` tool.
    """
    # First turn: send the page text. The Responses API stores the turn
    # server-side (store=True by default), so follow-up turns only need to
    # pass ``previous_response_id`` plus the tool outputs — we never replay
    # the assistant's function_call items ourselves.
    response = client.responses.parse(
        model=os.environ["OPENAI_MODEL"],
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        tools=TOOLS,
        text_format=Extraction,
        reasoning={"effort": REASONING_EFFORT},
    )

    for _ in range(MAX_ROUNDS - 1):
        tool_calls = [o for o in response.output if o.type == "function_call"]
        if not tool_calls:
            # No tool call -> final structured answer.
            if response.output_parsed is None:
                raise ValueError(
                    f"Model returned no parsed output (possible refusal): {response.output}"
                )
            return response.output_parsed

        # Resolve each call by name, then hand the results back to the model.
        outputs = [
            {
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": run_tool(tc.name, screenshot_blob),
            }
            for tc in tool_calls
        ]
        response = client.responses.parse(
            model=os.environ["OPENAI_MODEL"],
            previous_response_id=response.id,
            input=outputs,
            tools=TOOLS,
            text_format=Extraction,
            reasoning={"effort": REASONING_EFFORT},
        )

    raise RuntimeError("extraction loop exhausted without a final answer")


def run_tool(name: str, screenshot_blob: bytes | None):
    """Dispatch a single tool call to its handler by *name*.

    Returns the ``output`` value for a ``function_call_output`` item — either
    a string or a list of input parts (text + images).
    """
    if name == "get_screenshot":
        return screenshot_reply(screenshot_blob)
    raise ValueError(f"unknown tool: {name!r}")


def screenshot_reply(screenshot_blob: bytes | None):
    """Build the model-facing reply for a ``get_screenshot`` call."""
    if screenshot_blob is None:
        # Can't fulfil the request. Tell the model so it can answer from
        # text alone.
        return "No screenshot is available for this page."

    tile = int(os.environ["IMAGE_TILE_SIZE"])
    overlap = float(os.environ["IMAGE_TILE_OVERLAP"])
    tiles = split_image(screenshot_blob, tile, overlap)
    return [
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
                "image_url": "data:image/png;base64," + base64.b64encode(t).decode(),
            }
            for t in tiles
        ],
    ]


# ===========================================================================
# CSV helpers
# ===========================================================================


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class InputRow(BaseModel):
    """One row of the input CSV: a URL plus its known metadata."""

    institute: str
    position: str | None  # fallback when the model extracts no position
    url: str


def default_out_path(csv_path: str, kind: str) -> Path:
    """Default output path derived from the input CSV: ``data/<stem>.<kind>.jsonl``."""
    stem = Path(csv_path).stem
    return Path("data") / f"{stem}.{kind}.jsonl"


def load_input(path: str) -> list[InputRow]:
    """Parse the input CSV into typed rows, dropping rows with a blank URL."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            InputRow(
                institute=row["institute"].strip(),
                position=row["position"].strip() or None,
                url=row["url"].strip(),
            )
            for row in reader
            if row["url"].strip()
        ]


# ===========================================================================
# Core pipelines
# ===========================================================================


async def run_snapshot_csv(
    rows: list[InputRow], out_path: Path, concurrency: int
) -> None:
    urls = [row.url for row in rows]
    log.info("%d unique URL(s) to snapshot", len(urls))

    sem = asyncio.Semaphore(concurrency)

    async def limited_snapshot(url: str) -> dict:
        async with sem:
            return await async_snapshot_url(url)

    tasks = [asyncio.create_task(limited_snapshot(url)) for url in urls]
    results: list[dict] = []
    for task in asyncio.as_completed(tasks):
        data = await task
        results.append(data)
        log.info("snapshotted %s", data.get("url"))
    write_jsonl(out_path, results)
    log.info("wrote %d snapshot(s) → %s", len(results), out_path)


def extract_one(client: httpx.Client, row: InputRow) -> dict | None:
    """Fetch the latest snapshot for *row* and extract holders from it.

    Returns None if Pravda has no snapshot for the URL.

    ``row.position`` fills in any holder whose position the model left blank.
    """
    snapshot = latest_snapshot(client, row.url)
    if snapshot is None:
        log.info("  skip %s — no snapshot", row.url)
        return None

    out = {**snapshot, "model": os.environ["OPENAI_MODEL"]}

    text = read_blob(snapshot.get("plaintext")).decode("utf-8", errors="replace")
    shot_path = snapshot.get("screenshot")
    screenshot_blob = read_blob(shot_path) if shot_path else None
    if screenshot_blob is not None and is_blank(screenshot_blob):
        screenshot_blob = None

    log.info("%s → extracting …", snapshot["url"])
    extraction = extract(text, screenshot_blob)
    holders = [holder.model_dump() for holder in extraction.holders]
    for holder in holders:
        if holder["position"] is None:
            holder["position"] = row.position
    out.update(
        page_type=extraction.page_type.value,
        holders=holders,
    )
    log.info("%s → %d holder(s)", snapshot["url"], len(holders))
    return out


# ===========================================================================
# CLI
# ===========================================================================


@click.group(help=__doc__)
def cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )


@cli.command("snapshot-csv", help="Snapshot all URLs from a CSV through Pravda.")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=5,
    help="Max concurrent requests to Pravda.",
)
def snapshot_csv_cmd(csv_path: str, concurrency: int) -> None:
    out_path = default_out_path(csv_path, "snapshots")
    asyncio.run(run_snapshot_csv(load_input(csv_path), out_path, concurrency))


@cli.command(
    "extract",
    help=(
        "Read a CSV of URLs, fetch the latest Pravda snapshot for each, run an "
        "LLM extraction step, and write the results to "
        "data/<csv-stem>.extracted.jsonl."
    ),
)
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("-n", "--sample", type=int, default=None, help="Randomly sample N URLs.")
def extract_cmd(csv_path: str, sample: int | None) -> None:
    out_path = default_out_path(csv_path, "extracted")
    rows = load_input(csv_path)
    if sample is not None and sample < len(rows):
        rows = random.sample(rows, sample)
    log.info("%d URL(s) to extract", len(rows))

    results: list[dict] = []
    with httpx.Client(timeout=30) as client:
        for row in rows:
            result = extract_one(client, row)
            if result is not None:
                results.append(result)

    write_jsonl(out_path, results)
    log.info("wrote %d record(s) → %s", len(results), out_path)

    hits = sum(1 for r in results if r["holders"])
    misses = len(results) - hits
    log.info("extraction: %d hit, %d miss", hits, misses)

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


if __name__ == "__main__":
    cli()
