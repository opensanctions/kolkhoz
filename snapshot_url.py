"""Snapshot a single URL through Pravda."""

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

PRAVDA_URL = os.environ["PRAVDA_URL"]


def format_snapshot(data: dict) -> str:
    lines = []
    for key in ["id", "url", "captured_at", "http_status", "error"]:
        if key in data:
            lines.append(f"  {key}: {data[key]}")
    if "contents" in data:
        lines.append("  contents:")
        for c in data["contents"]:
            lines.append(f"    {c['path']}: {c['content_type']}")
    if "headers" in data:
        lines.append("  headers:")
        for h in data["headers"]:
            lines.append(f"    {h['name']}: {h['value']}")
    return "\n".join(lines)


def snapshot_url(url: str) -> dict:
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{PRAVDA_URL}/snapshots", json={"url": url})
        return resp.json()


async def async_snapshot_url(url: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{PRAVDA_URL}/snapshots", json={"url": url})
        return resp.json()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    data = snapshot_url(url)
    print(format_snapshot(data))


if __name__ == "__main__":
    main()
