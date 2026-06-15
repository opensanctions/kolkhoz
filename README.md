# Kolkhoz

Kolkhoz turns the internet into lists of politicians. It orchestrates web capture (via [Pravda](https://github.com/opensanctions/pravda)), LLM extraction, and structured storage to pull political position holders out of web pages.

## Status

Early R&D. Currently exploring what a viable automated extraction pipeline looks like.

## What it does

1. Sends URLs to Pravda for snapshotting (MHTML + screenshot)
2. Feeds snapshots to an LLM to extract structured "human / position" pairs
3. Stores results in SQLite, linked to Pravda snapshot identifiers
4. (Later) Evaluates extraction quality against manual gold sets

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a running [Pravda](https://github.com/opensanctions/pravda) instance at `http://127.0.0.1:8000`.

```bash
# Install dependencies
uv sync
```

## Usage

```bash
# Snapshot a single URL
uv run python scripts/snapshot_url.py https://example.org

# Snapshot all pep_urls from a CSV
uv run python scripts/snapshot_urls_from_csv.py data/hio_leadership.csv
```
