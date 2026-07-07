# Kolkhoz

Kolkhoz turns the internet into lists of politicians. It orchestrates web capture (via [Pravda](https://github.com/opensanctions/pravda)), LLM extraction, and structured storage to pull political position holders out of web pages.

## Status

Early R&D. Currently exploring what a viable automated extraction pipeline looks like.

## What it does

1. Sends URLs to Pravda for snapshotting (plaintext + rendered HTML + screenshot)
2. Feeds snapshots to an LLM to extract structured "human / position" pairs
3. Stores results in SQLite, linked to Pravda snapshot identifiers
4. Exports the extracted holders as JSONL (one record per person/position observation), shaped for ingest by zavod

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

# Export extracted holders as JSONL (one date-prefixed file per dataset)
uv run python kolkhoz.py export -o data/exports
```

## Evaluation

Score the extraction pipeline against hand-authored synthetic fixtures in `evaluate.py`. Each fixture is an authored page with a known set of holders; the harness renders it to HTML, derives the plaintext, runs the real `extract()`, and scores the returned (human, position) pairs by exact string equality.

```bash
uv run python evaluate.py   # run all fixtures
uv run python evaluate.py -v # show expected pairs
```

`data/` is gitignored.
