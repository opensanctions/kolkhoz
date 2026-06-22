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

# Content types Pravda captures, by their MIME type.
TEXT = "text/plain"  # inner_text("body") — the rendered, tag-stripped page text
SCREENSHOT = "image/png"  # full-page screenshot


async def latest_snapshot(client: httpx.AsyncClient, url: str) -> dict | None:
    """Return the most recent snapshot for *url*, or None if there are none."""
    resp = await client.get(
        f"{os.environ['PRAVDA_URL']}/snapshots", params={"url": url}
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return items[0] if items else None  # the API returns newest first


def content(snapshot: dict, content_type: str) -> dict | None:
    """The captured artifact of *content_type* in *snapshot*, if present."""
    for item in snapshot.get("contents", []):
        if item["content_type"] == content_type:
            return item
    return None


def content_hash(snapshot: dict, content_type: str) -> str | None:
    """The CAS hash of an artifact — the basename of its stored path."""
    item = content(snapshot, content_type)
    return Path(item["path"]).name if item else None


def read_blob(path: str) -> bytes:
    """Read a Pravda blob directly from shared storage."""
    return Path(path).read_bytes()


def read_text(item: dict | None) -> str:
    """Decode a text artifact, or "" if it is absent."""
    if item is None:
        return ""
    return read_blob(item["path"]).decode("utf-8", errors="replace")


def is_blank(blob: bytes) -> bool:
    """True if the image is a single solid colour (a blank or failed render).

    ``getcolors(1)`` returns a list iff the image has at most one distinct
    colour, else None — so a blank white/black/any-colour page reads as blank.
    """
    image = Image.open(io.BytesIO(blob))
    return image.getcolors(1) is not None
