"""Pravda integration and screenshot artifacts.

- ``async_snapshot_url`` / ``latest_snapshot`` talk to Pravda's API.
- ``is_blank`` / ``split_image`` are image primitives over a screenshot blob
  (shared by capture and the extraction tiling path).
- ``run_snapshot_csv`` snapshots a dataset's URLs through Pravda and records
  the pages in the database.
"""

import asyncio
import io
import logging

import httpx
from PIL import Image
from sqlalchemy.orm import Session

from kolkhoz.config import PravdaConfig
from kolkhoz.models import Page as PageRow

log = logging.getLogger("kolkhoz")


async def async_snapshot_url(
    pravda: PravdaConfig, url: str, sem: asyncio.Semaphore
) -> dict:
    async with sem, httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{pravda.url}/snapshots", json={"url": url})
        return resp.json()


def latest_snapshot(
    pravda: PravdaConfig, client: httpx.Client, url: str
) -> dict | None:
    """Return the most recent snapshot for *url*, or None if there are none."""
    resp = client.get(f"{pravda.url}/snapshots", params={"url": url})
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


async def run_snapshot_csv(
    rows: list,
    dataset: str,
    concurrency: int,
    pravda: PravdaConfig,
    engine,
) -> None:
    urls = {row.url for row in rows}
    log.info("%d unique URL(s) to snapshot", len(urls))

    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(async_snapshot_url(pravda, url, sem)) for url in urls]
    for task in asyncio.as_completed(tasks):
        data = await task
        log.info("snapshotted %s", data["url"])

    with Session(engine) as session:
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
    log.info("wrote %d page(s) for dataset %s", len(rows), dataset)
