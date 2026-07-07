import asyncio
import base64
import csv
import io
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import click
import fsspec
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from models import Base
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
    # The API returns newest first; ignore snapshots that errored during capture.
    items = [i for i in resp.json().get("items", []) if i.get("error") is None]
    return items[0] if items else None


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

    Solid-colour strips (remainder offcuts, background bands) carry no
    content and are dropped via the same ``getcolors(1)`` check as
    ``is_blank``, so they never reach the model.
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
        crop = image.crop((0, top, width, bottom))
        if crop.getcolors(1) is not None:
            continue
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
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
- Preserve source wording. Do not normalize names, countries, or dates —
  copy them as written. Dates stay as source strings (e.g. '3 May 2022').
- Leave any field blank (null, or empty list for evidence) when the page
  does not state it. Do not infer values from context.
- For evidence_quotes, lift one or more short, verbatim phrases from the
  page that support this (person, position) observation. Prefer the
  sentence that names the person and the role together. If the page only
  states it in a table or list with no prose, return an empty list.
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
    person_dob: str | None = Field(
        default=None,
        description=(
            "The person's date of birth as written on the page (source string, "
            "no normalization). Null if not stated."
        ),
    )
    person_bio: str | None = Field(
        default=None,
        description=(
            "A short biographical note about the person, taken verbatim or "
            "near-verbatim from the page. Null if none is given."
        ),
    )
    person_country: str | None = Field(
        default=None,
        description=(
            "Country associated with the person, as written on the page "
            "(source wording, no territory-code normalization). Null if not stated."
        ),
    )
    position_description: str | None = Field(
        default=None,
        description=(
            "Description of the position as stated on the page, verbatim or "
            "near-verbatim. Null if none is given."
        ),
    )
    position_jurisdiction: str | None = Field(
        default=None,
        description=(
            "Jurisdiction the position operates in, as written on the page "
            "(source wording, no normalization). Null if not stated."
        ),
    )
    position_start_date: str | None = Field(
        default=None,
        description=(
            "When the person started in this position, as written on the page "
            "(source string, no normalization). Null if not stated."
        ),
    )
    position_end_date: str | None = Field(
        default=None,
        description=(
            "When the person left (or will leave) this position, as written on "
            "the page (source string, no normalization). Null if not stated."
        ),
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description=(
            "One or more short quotes lifted verbatim from the page that "
            "support this (person, position) observation. Empty list if the "
            "page states it only in a table or list with no prose."
        ),
    )


class Extraction(BaseModel):
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
        "  → final: %d holder(s)",
        len(response.output_parsed.holders),
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

    organization: str
    url: str


def dataset_name(path: str) -> str:
    """Dataset name derived from the input CSV's filename stem."""
    return os.path.splitext(os.path.basename(path))[0]


def load_inputs(base_path: str) -> list[tuple[str, list[InputRow]]]:
    """Load every CSV under the input directory as a (dataset, rows) pair.

    *base_path* is an fsspec URL (local dir or ``gs://``/``s3://`` prefix).
    Each CSV becomes its own dataset, named after the file's stem; rows with
    a blank URL are dropped. Files are read straight from the bucket without
    a local copy.
    """
    fs, base = fsspec.core.url_to_fs(base_path)
    result: list[tuple[str, list[InputRow]]] = []
    for path in sorted(fs.glob(os.path.join(base, "*.csv"))):
        with fs.open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                InputRow(
                    organization=row["organization"].strip(),
                    url=row["url"].strip(),
                )
                for row in reader
                if row["url"].strip()
            ]
        result.append((dataset_name(path), rows))
    return result


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
                        organization=row.organization,
                        dataset=dataset,
                    )
                )
        session.commit()
    log.info("wrote %d page(s) → %s", len(rows), os.environ["KOLKHOZ_DB"])


# The flat JSONL schema handed to zavod. One record per (person, position)
# observation extracted from one snapshot. Fixed key order, UTF-8, source
# wording preserved (no FtM normalization — that is zavod's job). Every
# field is a scalar — no nested objects — except ``evidence_quotes``, which
# stays a native list[str]; a list of supporting quotes is the reason we
# left CSV behind.
EXPORT_FIELDS = [
    "dataset",
    "source_url",
    "snapshot_id",
    "snapshot_retrieved_at",
    "organisation_name",
    "person_name",
    "person_dob",
    "person_bio",
    "person_country",
    "position_name",
    "position_description",
    "position_jurisdiction",
    "position_start_date",
    "position_end_date",
    "evidence_quotes",
]

# Export files are written to <output-base>/<dataset>/<date>.jsonl, where
# <date> is the export run date.
EXPORT_DATE_FORMAT = "%Y-%m-%d"


def holder_to_record(
    page: PageRow, extraction: ExtractionRow, holder: HolderRow
) -> dict:
    """Flatten one holder observation into a JSONL record.

    The schema is flat — every field is a scalar — except
    ``evidence_quotes``, which stays a native list[str]. All dates are
    plain source strings, copied verbatim with no parsing or reformatting.
    """
    record = {
        "dataset": page.dataset,
        "source_url": page.url,
        "snapshot_id": extraction.snapshot_id,
        "snapshot_retrieved_at": extraction.snapshot_retrieved_at,
        "organisation_name": page.organization,
        "person_name": holder.human,
        "person_dob": holder.person_dob,
        "person_bio": holder.person_bio,
        "person_country": holder.person_country,
        "position_name": holder.position,
        "position_description": holder.position_description,
        "position_jurisdiction": holder.position_jurisdiction,
        "position_start_date": holder.position_start_date,
        "position_end_date": holder.position_end_date,
        "evidence_quotes": holder.evidence_quotes,
    }
    return {name: record[name] for name in EXPORT_FIELDS}


@click.group()
def cli() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.FileHandler("kolkhoz.log"))
    root.addHandler(logging.StreamHandler())


@cli.command(
    "snapshot-csv",
    help="Snapshot all URLs from the CSVs in the input directory through Pravda.",
)
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=5,
    help="Max concurrent requests to Pravda.",
)
def snapshot_csv_cmd(concurrency: int) -> None:
    Path(os.environ["KOLKHOZ_DB"]).parent.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(os.environ["INPUT_BASE_PATH"])
    log.info("%d input CSV(s)", len(inputs))
    for dataset, rows in inputs:
        log.info("dataset %s: %d row(s)", dataset, len(rows))
        asyncio.run(run_snapshot_csv(rows, dataset, concurrency))


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

            prefix = snapshot["prefix"]
            text = (
                Path(prefix, snapshot["plaintext"])
                .read_bytes()
                .decode("utf-8", errors="replace")
            )
            html = (
                Path(prefix, snapshot["rendered_html"])
                .read_bytes()
                .decode("utf-8", errors="replace")
            )

            log.info("%s → extracting …", snapshot["url"])
            screenshot_blob = None
            reason = screenshot_reason(text, html)
            if reason is not None:
                log.info("  → %s → including screenshot", reason)
                if snapshot.get("screenshot") is not None:
                    blob = Path(prefix, snapshot["screenshot"]).read_bytes()
                    if not is_blank(blob):
                        screenshot_blob = blob
            extraction = extract(text, screenshot_blob)
            holders = [h.model_dump() for h in extraction.holders]
            log.info("%s → %d holder(s)", snapshot["url"], len(holders))

            n += 1
            if holders:
                hits += 1

            # Re-attach the page to this session (it was loaded above in a
            # different, now-closed session).
            page = session.merge(page)

            extraction_row = ExtractionRow(
                page_id=page.id,
                snapshot_id=snapshot["id"],
                snapshot_retrieved_at=snapshot["captured_at"],
                model=os.environ["OPENAI_MODEL"],
                extracted_at=datetime.now(),
            )
            for h in holders:
                extraction_row.holders.append(
                    HolderRow(
                        human=h["human"],
                        position=h["position"],
                        person_dob=h["person_dob"],
                        person_bio=h["person_bio"],
                        person_country=h["person_country"],
                        position_description=h["position_description"],
                        position_jurisdiction=h["position_jurisdiction"],
                        position_start_date=h["position_start_date"],
                        position_end_date=h["position_end_date"],
                        evidence_quotes=h["evidence_quotes"],
                    )
                )
            session.add(extraction_row)

        session.commit()

    log.info("wrote %d record(s) → %s", n, os.environ["KOLKHOZ_DB"])
    log.info("extraction: %d hit, %d miss", hits, n - hits)


@cli.command(
    "export",
    help=(
        "Export extracted holders as JSONL (one record per person-position "
        "observation). Writes one .jsonl file per dataset to "
        "<output-base>/<dataset>/<date>.jsonl."
    ),
)
def export_cmd() -> None:
    with Session() as session:
        # Only the most recent extraction per page, so re-running extraction
        # doesn't multiply the output.
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
        groups: dict[str, list[tuple[PageRow, ExtractionRow, HolderRow]]] = {}
        for holder, extraction, page in session.execute(stmt).all():
            groups.setdefault(page.dataset, []).append((page, extraction, holder))

    fs, base = fsspec.core.url_to_fs(os.environ["OUTPUT_BASE_PATH"])
    date = datetime.now().strftime(EXPORT_DATE_FORMAT)
    total = 0
    for group, group_rows in groups.items():
        out_dir = os.path.join(base, group)
        out_file = os.path.join(out_dir, f"{date}.jsonl")
        fs.makedirs(out_dir, exist_ok=True)
        with fs.open(out_file, "wb") as fh:
            for page, extraction, holder in group_rows:
                record = holder_to_record(page, extraction, holder)
                fh.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
                fh.write(b"\n")
        total += len(group_rows)
        log.info("wrote %d record(s) → %s", len(group_rows), out_file)
    log.info("exported %d record(s) across %d dataset(s)", total, len(groups))


if __name__ == "__main__":
    cli()
