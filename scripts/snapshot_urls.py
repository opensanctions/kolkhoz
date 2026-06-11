"""Send all pep_url from the CSV through Pravda for snapshotting."""

import csv
import os

import json

import httpx
from dotenv import load_dotenv

load_dotenv()

PRAVDA_URL = os.environ["PRAVDA_URL"]
CSV_PATH = "data/hio_leadership.csv"


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
            print(f"  [{resp.status_code}] {url}")
            print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
