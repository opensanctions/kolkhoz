"""Read helpers for Pravda snapshots.

Look up the latest snapshot for a URL and read captured blobs straight off
Pravda's content-addressed storage (the API returns file paths; we read them
directly, as Pravda intends).
"""

import io
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


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
