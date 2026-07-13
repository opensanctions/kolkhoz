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

All commands run through the `kolkhoz` console script (installed by `uv sync`). Input and output
locations are fsspec URLs set via `INPUT_BASE_PATH` and `OUTPUT_BASE_PATH`
in `.env` (a local dir or a `gs://`/`s3://` bucket prefix):

```bash
# Snapshot all URLs from every CSV in the input directory (INPUT_BASE_PATH)
# through Pravda, recording pages in the DB. Each CSV is its own dataset,
# named after the file's stem.
uv run kolkhoz snapshot

# Extract position holders from the latest snapshot of each page in the DB
uv run kolkhoz extract
uv run kolkhoz extract -d hio_leadership   # one dataset only
uv run kolkhoz extract -n 20               # random sample of 20

# Export extracted holders as JSONL, one file per dataset under
# <OUTPUT_BASE_PATH>/<dataset>/<date>.jsonl
uv run kolkhoz export
```

## Evaluation

Score the extraction pipeline against hand-authored fixture pages. Each fixture is a directory under `fixtures/` holding `page.html`, an `expected.json` answer key, and an optional `screenshot.png` that drives the image path. The harness derives the plaintext the model reads, runs the real `extract()`, and scores the returned (human, position) pairs by exact string equality. See `evaluate.py`'s docstring for the fixture layout.

```bash
uv run python evaluate.py   # run all fixtures
uv run python evaluate.py -v # show expected pairs
```
