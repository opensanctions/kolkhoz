import asyncio
import base64
import csv
import io
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from followthemoney import model as ftm_model
from followthemoney.proxy import EntityProxy
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from models import Base, PageType
from models import Extraction as ExtractionRow
from models import Holder as HolderRow
from models import Page as PageRow

load_dotenv()

engine = create_engine(f"sqlite:///{os.environ['KOLKHOZ_DB']}")
Session = sessionmaker(engine)

log = logging.getLogger("kolkhoz")


async def async_snapshot_url(url: str, sem: asyncio.Semaphore) -> dict:
    async with sem, httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{os.environ['PRAVDA_URL']}/snapshots", json={"url": url}
        )
        return resp.json()


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
    page_type: PageType = Field(
        description=(
            "The kind of page this is. "
            "`roster` lists multiple named position holders. "
            "`profile` is a single person's page about themselves. "
            "`other` is a page that is not in the business of listing "
            "position holders."
        )
    )
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

    for _ in range(MAX_ROUNDS):
        # Each turn: log what the model did, then either settle on its
        # final structured answer or resolve its tool calls and loop.
        tool_calls = [o for o in response.output if o.type == "function_call"]
        for tc in tool_calls:
            log.info("  → tool call: %s", tc.name)
        if not tool_calls:
            log.info(
                "  → final: %d holder(s), page_type=%s",
                len(response.output_parsed.holders),
                response.output_parsed.page_type.value,
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


class InputRow(BaseModel):
    """One row of the input CSV: a URL plus its known metadata."""

    institute: str
    position: str
    url: str


def load_input(path: str) -> list[InputRow]:
    """Parse the input CSV into typed rows, dropping rows with a blank URL."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            InputRow(
                institute=row["institute"].strip(),
                position=row["position"].strip(),
                url=row["url"].strip(),
            )
            for row in reader
            if row["url"].strip() and row["position"].strip()
        ]


async def run_snapshot_csv(
    rows: list[InputRow], dataset: str, concurrency: int
) -> None:
    urls = {row.url for row in rows}
    log.info("%d unique URL(s) to snapshot", len(urls))

    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(async_snapshot_url(url, sem)) for url in urls]
    for task in asyncio.as_completed(tasks):
        data = await task
        log.info("snapshotted %s", data["url"])

    Base.metadata.create_all(engine)
    with Session() as session:
        for row in rows:
            page = session.query(PageRow).filter_by(url=row.url).first()
            if page is None:
                session.add(
                    PageRow(
                        url=row.url,
                        institute=row.institute,
                        position=row.position,
                        dataset=dataset,
                    )
                )
        session.commit()
    log.info("wrote %d page(s) → %s", len(rows), os.environ["KOLKHOZ_DB"])


def build_ftm_entities(session, dataset: str | None = None) -> list[dict]:
    """Build a list of Followthemoney entity dicts from the database.

    Emits Organization, Position, Person, Occupancy, and Document entities,
    deduplicated and merged by id. Only the latest extraction per page is
    exported, so re-running extraction doesn't multiply the output.
    """
    bucket: dict[str, EntityProxy] = {}

    # The latest Extraction per page: we only export the most recent read of
    # each page, so re-running extraction doesn't multiply the output.
    latest = (
        select(
            ExtractionRow.page_id.label("page_id"),
            func.max(ExtractionRow.id).label("extraction_id"),
        )
        .group_by(ExtractionRow.page_id)
        .subquery()
    )
    stmt = (
        select(HolderRow, ExtractionRow, PageRow)
        .join(latest, HolderRow.extraction_id == latest.c.extraction_id)
        .join(ExtractionRow, HolderRow.extraction_id == ExtractionRow.id)
        .join(PageRow, ExtractionRow.page_id == PageRow.id)
    )
    if dataset is not None:
        stmt = stmt.where(PageRow.dataset == dataset)
    rows = session.execute(stmt).all()

    for holder, extraction, page in rows:
        # --- Organization: the institute the page is about. ---
        org = ftm_model.make_entity("Organization")
        org.make_id("org", page.dataset, page.institute)
        org.add("name", page.institute)
        org.add("website", page.url)
        bucket[org.id] = bucket.get(org.id, org).merge(org)

        # --- Document: the Pravda snapshot this holder was read from. ---
        doc = ftm_model.make_entity("Document")
        doc.make_id("pravda", extraction.snapshot_id)
        doc.add("title", page.url)
        doc.add("sourceUrl", page.url)
        doc.add("notes", f"Pravda snapshot {extraction.snapshot_id}")
        doc.add("author", extraction.model)
        bucket[doc.id] = bucket.get(doc.id, doc).merge(doc)

        # --- Person: the named human. ---
        person = ftm_model.make_entity("Person")
        person.make_id("person", page.dataset, holder.human)
        person.add("name", holder.human)
        person.add("sourceUrl", page.url)
        person.add("proof", doc.id)
        bucket[person.id] = bucket.get(person.id, person).merge(person)

        # --- Position: the (institute, title) role. ---
        position_title = holder.position or page.position
        position = ftm_model.make_entity("Position")
        position.make_id("position", org.id, position_title)
        position.add("name", position_title)
        position.add("organization", org.id)
        position.add("sourceUrl", page.url)
        bucket[position.id] = bucket.get(position.id, position).merge(position)

        # --- Occupancy: this person holding this position. ---
        occupancy = ftm_model.make_entity("Occupancy")
        occupancy.make_id("occupancy", person.id, position.id)
        occupancy.add("holder", person.id)
        occupancy.add("post", position.id)
        occupancy.add("status", "current")
        occupancy.add("date", extraction.extracted_at.date().isoformat())
        occupancy.add("sourceUrl", page.url)
        occupancy.add("proof", doc.id)
        bucket[occupancy.id] = bucket.get(occupancy.id, occupancy).merge(occupancy)

    return [entity.to_dict() for entity in bucket.values()]


@click.group()
def cli() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.FileHandler("kolkhoz.log"))
    root.addHandler(logging.StreamHandler())


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
    Path(os.environ["KOLKHOZ_DB"]).parent.mkdir(parents=True, exist_ok=True)
    dataset = Path(csv_path).stem
    asyncio.run(run_snapshot_csv(load_input(csv_path), dataset, concurrency))


@cli.command(
    "extract",
    help=(
        "Read pages from the database, fetch the latest Pravda snapshot for "
        "each, run an LLM extraction step, and store the results in the "
        "database."
    ),
)
@click.option(
    "-d", "--dataset", type=str, default=None, help="Only extract this dataset."
)
@click.option("-n", "--sample", type=int, default=None, help="Randomly sample N pages.")
def extract_cmd(dataset: str | None, sample: int | None) -> None:
    Path(os.environ["KOLKHOZ_DB"]).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)

    with Session() as session:
        query = session.query(PageRow)
        if dataset is not None:
            query = query.filter_by(dataset=dataset)
        pages = query.all()

    if sample is not None and sample < len(pages):
        pages = random.sample(pages, sample)
    log.info("%d page(s) to extract", len(pages))

    n = 0
    hits = 0
    pt_counts: dict[str, int] = {}
    with Session() as session, httpx.Client(timeout=30) as client:
        for page in pages:
            snapshot = latest_snapshot(client, page.url)
            if snapshot is None:
                log.info("  skip %s — no snapshot", page.url)
                continue

            already = (
                session.query(ExtractionRow)
                .filter_by(page_id=page.id, snapshot_id=snapshot["id"])
                .first()
            )
            if already is not None:
                log.info(
                    "  skip %s — snapshot %s already extracted",
                    page.url,
                    snapshot["id"],
                )
                continue

            text = read_blob(snapshot.get("plaintext")).decode(
                "utf-8", errors="replace"
            )
            shot_path = snapshot.get("screenshot")
            screenshot_blob = read_blob(shot_path) if shot_path else None
            if screenshot_blob is not None and is_blank(screenshot_blob):
                screenshot_blob = None

            log.info("%s → extracting …", snapshot["url"])
            extraction = extract(text, screenshot_blob)
            holders = [h.model_dump() for h in extraction.holders]
            for holder in holders:
                if holder["position"] is None:
                    holder["position"] = page.position
            log.info("%s → %d holder(s)", snapshot["url"], len(holders))

            n += 1
            if holders:
                hits += 1
            pt_counts[extraction.page_type.value] = (
                pt_counts.get(extraction.page_type.value, 0) + 1
            )

            # Re-attach the page to this session (it was loaded above in a
            # different, now-closed session).
            page = session.merge(page)

            extraction_row = ExtractionRow(
                page_id=page.id,
                snapshot_id=snapshot["id"],
                model=os.environ["OPENAI_MODEL"],
                extracted_at=datetime.now(),
                page_type=extraction.page_type,
            )
            for h in holders:
                extraction_row.holders.append(
                    HolderRow(human=h["human"], position=h["position"])
                )
            session.add(extraction_row)

        session.commit()

    log.info("wrote %d record(s) → %s", n, os.environ["KOLKHOZ_DB"])
    log.info("extraction: %d hit, %d miss", hits, n - hits)
    log.info(
        "page_type: roster=%d profile=%d other=%d",
        pt_counts.get("roster", 0),
        pt_counts.get("profile", 0),
        pt_counts.get("other", 0),
    )


@cli.command(
    "export-ftm",
    help=(
        "Export extracted holders to a Followthemoney "
        "(followthemoney.tech) entity stream. Writes one JSON entity per "
        "line (the ijson stream format used by the `ftm` CLI) to the "
        "output file, or stdout if none is given."
    ),
)
@click.option(
    "-d", "--dataset", type=str, default=None, help="Only export this dataset."
)
@click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout).",
)
def export_ftm_cmd(dataset: str | None, output) -> None:
    import json

    with Session() as session:
        entities = build_ftm_entities(session, dataset)

    for entity in entities:
        output.write(json.dumps(entity, ensure_ascii=False))
        output.write("\n")
    output.flush()
    log.info("exported %d entit(ies)", len(entities))


if __name__ == "__main__":
    cli()
