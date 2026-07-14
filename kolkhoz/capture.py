"""Pravda integration and screenshot artifacts.

- ``run_snapshots`` captures every URL through a single long-lived Pravda
  instance (bounded by a semaphore) and records the pages in the database.
- ``latest_snapshot`` queries Pravda's history for the newest successful
  snapshot of a URL.
- ``read_artifact`` reads a snapshot artifact blob from the shared fsspec
  storage backend Kolkhoz and Pravda both use.
- ``is_blank`` / ``split_image`` are image primitives over a screenshot blob
  (shared by capture and the extraction tiling path).

Pravda is an in-process async library. Kolkhoz constructs one ``Pravda``
instance from its environment-backed settings and reuses it across captures;
it never speaks HTTP to Pravda.
"""

import asyncio
import io
import logging
import os

import fsspec
from PIL import Image
from pravda import Pravda, PravdaConfig, Snapshot, migrate
from sqlalchemy.orm import Session

from kolkhoz.config import PravdaSettings
from kolkhoz.models import Page as PageRow

log = logging.getLogger("kolkhoz")


def pravda_client(settings: PravdaSettings) -> Pravda:
    """Construct a Pravda instance from Kolkhoz's environment-backed settings.

    The ``PravdaConfig`` is built here, at the application boundary, from the
    explicit settings Kolkhoz owns; Kolkhoz holds no Pravda URL of its own.
    """
    config = PravdaConfig(
        database_url=settings.database_url,
        browser_ws_url=settings.browser_ws_url,
        storage_base_path=settings.storage_base_path,
    )
    return Pravda(config)


def storage_filesystem(settings: PravdaSettings):
    """The shared fsspec backend Pravda writes artifacts to.

    Pravda resolves each snapshot's ``prefix`` against this same base path, so
    opening ``<prefix>/<filename>`` on this filesystem locates the artifact for
    both local paths and remote (``gs://``/``s3://``) URLs.
    """
    fs, _ = fsspec.core.url_to_fs(settings.storage_base_path)
    return fs


async def latest_snapshot(pravda: Pravda, url: str) -> Snapshot | None:
    """Return the newest successful snapshot of *url*, or None if there are none.

    Pravda returns history newest first and persists capture failures with
    ``error`` set, so the first error-free entry is the newest success.
    """
    snapshots = await pravda.snapshots(url)
    for snapshot in snapshots:
        if snapshot.error is None:
            return snapshot
    return None


def read_artifact(fs, snapshot: Snapshot, filename: str | None) -> bytes:
    """Read a snapshot artifact blob from the shared storage backend.

    ``snapshot.prefix`` is the backend-resolved directory (base path plus the
    normalized host of ``final_url``); *filename* is the bare
    content-addressed name Pravda stored. Both are required for a stored
    artifact: a missing one is a malformed snapshot, so this fails loud rather
    than returning empty bytes.
    """
    if snapshot.prefix is None:
        raise ValueError(f"snapshot {snapshot.id} has no storage prefix")
    if filename is None:
        raise ValueError(f"snapshot {snapshot.id} has no artifact filename")
    path = os.path.join(snapshot.prefix, filename)
    with fs.open(path, "rb") as fh:
        return fh.read()


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


async def run_snapshots(
    inputs: list[tuple[str, list]],
    concurrency: int,
    settings: PravdaSettings,
    engine,
) -> None:
    """Snapshot every URL across all datasets through one Pravda instance.

    Pravda's schema is migrated to head first (idempotently). Then one
    long-lived ``Pravda`` instance captures all unique URLs, bounded by a
    semaphore so at most *concurrency* captures run at once. After capturing,
    each dataset's pages are recorded in the database (a page is created once
    per distinct URL).
    """
    urls = sorted({row.url for _, rows in inputs for row in rows})
    log.info("%d unique URL(s) to snapshot", len(urls))

    await migrate(settings.database_url)

    sem = asyncio.Semaphore(concurrency)
    pravda = pravda_client(settings)
    async with pravda:

        async def snap(url: str) -> None:
            async with sem:
                snapshot = await pravda.snapshot(url)
                log.info("snapshotted %s", snapshot.url)

        await asyncio.gather(*(snap(url) for url in urls))

    with Session(engine) as session:
        for dataset, rows in inputs:
            for row in rows:
                if session.query(PageRow).filter_by(url=row.url).first() is None:
                    session.add(
                        PageRow(
                            url=row.url,
                            organization=row.organization,
                            dataset=dataset,
                        )
                    )
            session.commit()
            log.info("wrote %d page(s) for dataset %s", len(rows), dataset)
