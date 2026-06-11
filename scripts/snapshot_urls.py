"""Send all pep_url from the CSV through Pravda for snapshotting."""

import csv
import os


import httpx
from dotenv import load_dotenv

load_dotenv()

PRAVDA_URL = os.environ["PRAVDA_URL"]
CSV_PATH = "data/hio_leadership.csv"


def format_snapshot(data: dict) -> str:
    lines = []
    for key in ["id", "url", "captured_at", "http_status", "error"]:
        if key in data:
            lines.append(f"  {key}: {data[key]}")
    if "contents" in data:
        lines.append("  contents:")
        for c in data["contents"]:
            lines.append(f"    {c['content_type']}: {c['path']}")
    if "headers" in data:
        lines.append("  headers:")
        for h in data["headers"]:
            lines.append(f"    {h['name']}: {h['value']}")
    return "\n".join(lines)


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted({row["pep_url"] for row in reader if row["pep_url"].strip()})


def main() -> None:
    urls = load_urls(CSV_PATH)
    print(f"Found {len(urls)} unique URLs")

    with httpx.Client(timeout=120) as client:
        for url in urls:
            resp = client.post(f"{PRAVDA_URL}/snapshots", json={"url": url})
            data = resp.json()
            print(format_snapshot(data))


if __name__ == "__main__":
    main()
