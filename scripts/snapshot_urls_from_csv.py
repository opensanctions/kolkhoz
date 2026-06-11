"""Snapshot all pep_urls from a CSV through Pravda."""

import csv
import sys

from snapshot_url import format_snapshot, snapshot_url

DEFAULT_CSV = "data/hio_leadership.csv"


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return sorted({row["pep_url"] for row in reader if row["pep_url"].strip()})


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    urls = load_urls(csv_path)
    print(f"Found {len(urls)} unique URLs in {csv_path}")

    for url in urls:
        data = snapshot_url(url)
        print(format_snapshot(data))


if __name__ == "__main__":
    main()
