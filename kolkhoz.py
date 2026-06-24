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

# Content signals that force the full-page screenshot into the model
# input. Below MIN_TEXT_WORDS the plaintext is too thin to read holders
# from; with at least MIN_IMAGES and an imgs-per-word ratio over
# MAX_IMG_DENSITY the page is image-dominated (org charts, headshot
# rosters) and the names/titles are likely baked into the images rather
# than the text. The decision is made in code, not by the model.
MIN_TEXT_WORDS = 200
MIN_IMAGES = 10
MAX_IMG_DENSITY = 0.1

_INSTRUCTIONS_HEAD = """\
You extract political position holders from the content of a single web page.

A "holder" is a specific, named human who holds a named position (office, seat,
title, or role) at the organisation the page is about — e.g. a council member,
board director, judge, minister, or chair.

Rules:
- Return one entry per (human, position) pair. If one person holds two
  positions, return two entries.
- Only extract humans actually named on the page. Do not invent people.
- Ignore names that are not position holders (authors, contacts, mentions).
"""

_INSTRUCTIONS_WITH_SCREENSHOT = """\
- The full-page screenshot is attached as overlapping image tiles. Read the
  tiles together with the text as one page. Tiles overlap, so the same
  person/position may recur — extract each holder once.
"""


class Holder(BaseModel):
    human: str = Field(
        description="Full name of the person, exactly as written on the page."
    )
    position: str | None = Field(
        default=None,
        description=(
            "Specific position title the person holds, e.g. 'Council Member'. "
            "Use the most specific title shown; if the page is about an "
            "organisation and the title omits it, you may name the body. "
            "Omit (null) only if the page names the person but states no "
            "specific title for them."
        ),
    )


class Extraction(BaseModel):
    page_type: PageType = Field(
        description=(
            "The kind of page this is. "
            "`roster` lists multiple named position holders. "
            "`profile` is a single person's page about themselves. "
            "`other` is for generic pages (about, contact, article, landing) "
            "that are not in the business of listing position holders."
        )
    )
    holders: list[Holder] = Field(
        description="Position holders found on the page. Empty if none."
    )


def extract(text: str, screenshot_blob: bytes | None) -> Extraction:
    """Extract holders from a single page.

    Always sends the page *text*. When the content signals in the caller
    fire, *screenshot_blob* is tiled and attached as overlapping image
    parts alongside the text. There is no model-driven tool call: the
    decision to include the screenshot is made in code from text length
    and image density, not delegated to the model.
    """
    parts: list[dict] = [{"type": "input_text", "text": text}]
    if screenshot_blob is not None:
        parts.extend(screenshot_parts(screenshot_blob))

    response = client.responses.parse(
        model=os.environ["OPENAI_MODEL"],
        instructions=_INSTRUCTIONS_HEAD
        + (_INSTRUCTIONS_WITH_SCREENSHOT if screenshot_blob is not None else ""),
        input=[{"role": "user", "content": parts}],
        text_format=Extraction,
        reasoning={"effort": REASONING_EFFORT},
    )
    log.info(
        "  → final: %d holder(s), page_type=%s",
        len(response.output_parsed.holders),
        response.output_parsed.page_type.value,
    )
    return response.output_parsed


def screenshot_parts(screenshot_blob: bytes) -> list[dict]:
    """Tile a full-page screenshot into overlapping input parts for the model."""
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


def screenshot_reason(text: str, html: str) -> str | None:
    """Return why the screenshot should be attached, or None to skip it.

    Any one signal forces the screenshot into the model input:
    - *thin text*: fewer than ``MIN_TEXT_WORDS`` words of plaintext, i.e.
      the names/titles are unlikely to be in the text at all.
    - *image-dense*: at least ``MIN_IMAGES`` images and an imgs-per-word
      ratio above ``MAX_IMG_DENSITY``, i.e. the page is dominated by
      images that likely carry the names/titles (org charts, headshot
      rosters).
    """
    words = len(text.split())
    imgs = html.count("<img")
    if words < MIN_TEXT_WORDS:
        return f"thin text ({words} words)"
    if imgs >= MIN_IMAGES and imgs / words > MAX_IMG_DENSITY:
        return f"image-dense ({imgs} imgs / {words} words)"
    return None


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

            text = read_blob(snapshot["plaintext"]).decode("utf-8", errors="replace")
            html = read_blob(snapshot["rendered_html"]).decode(
                "utf-8", errors="replace"
            )

            log.info("%s → extracting …", snapshot["url"])
            screenshot_blob = None
            reason = screenshot_reason(text, html)
            if reason is not None:
                log.info("  → %s → including screenshot", reason)
                shot_path = snapshot.get("screenshot")
                if shot_path:
                    blob = read_blob(shot_path)
                    if not is_blank(blob):
                        screenshot_blob = blob
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
