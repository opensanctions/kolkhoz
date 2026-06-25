# Kolkhoz

Kolkhoz turns the internet into lists of politicians. It orchestrates web capture (via [Pravda](https://github.com/opensanctions/pravda)), LLM extraction, and structured storage to pull political position holders out of web pages.

## Status

Early R&D. Currently exploring what a viable automated extraction pipeline looks like.

## What it does

1. Sends URLs to Pravda for snapshotting (plaintext + rendered HTML + screenshot)
2. Feeds snapshots to an LLM to extract structured "human / position" pairs
3. Stores results in SQLite, linked to Pravda snapshot identifiers
4. Exports the extracted holders as a [FtM](https://followthemoney.tech) entity stream (Organization, Position, Person, Occupancy, Document)

## Setup

Requires uv and a running Pravda instance at `http://127.0.0.1:8000`.

```bash
# Install dependencies
uv sync
```

## Usage

All commands run through a single `kolkhoz.py` CLI:

```bash
# Snapshot all URLs from a CSV through Pravda, recording pages in the DB
uv run python kolkhoz.py snapshot-csv data/hio_leadership.csv

# Extract position holders from the latest snapshot of each page in the DB
uv run python kolkhoz.py extract
uv run python kolkhoz.py extract -d hio_leadership   # one dataset only
uv run python kolkhoz.py extract -n 20               # random sample of 20

# Export extracted holders as a Followthemoney entity stream
uv run python kolkhoz.py export-ftm -o kolkhoz.ftm
uv run python kolkhoz.py export-ftm -d hio_leadership -o hio.ftm
```
