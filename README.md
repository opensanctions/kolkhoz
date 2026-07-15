# Kolkhoz

Kolkhoz turns the internet into lists of politicians. It orchestrates web capture (via the in-process [Pravda](https://github.com/opensanctions/pravda) async library), LLM extraction, and JSONL output to pull political position holders out of web pages.

## Status

Early R&D. Currently exploring what a viable automated extraction pipeline looks like.

## What it does

1. Captures snapshots (plaintext + rendered HTML + screenshot) via the in-process Pravda library against a remote browser, Postgres, and an artifact store that Kolkhoz owns and runs
2. Feeds snapshots to an LLM to extract structured "human / position" pairs
3. Writes that run's extracted holders as JSONL (one record per person/position observation), shaped for ingest by zavod

Kolkhoz does not persist extraction results in PostgreSQL. Pravda still stores the captured evidence and snapshot metadata there.

## Setup

Requires uv. Kolkhoz embeds Pravda as an async library and
owns the infrastructure Pravda connects to: a headed Chrome browser, a
Postgres database, and an artifact store. Pravda ships on PyPI as
`opensanctions-pravda` (imported as `pravda`); `uv sync` installs it.
Bring the infrastructure up with Docker Compose:

```bash
# Install dependencies
uv sync

# Start the browser (Playwright run-server) and Postgres
docker compose up -d
```

The `run` command applies Pravda's packaged Alembic migrations idempotently before use, so no separate migration step is needed.

## Usage

All commands run through the `kolkhoz` console script (installed by `uv sync`). Input and output
locations are fsspec URLs set via `INPUT_BASE_PATH` and `OUTPUT_BASE_PATH`
in `.env` (a local dir or a `gs://`/`s3://` bucket prefix):

```bash
# Capture every URL from every CSV in the input directory (INPUT_BASE_PATH)
# through Pravda, then extract position holders from each snapshot and write
# this run's JSONL to <OUTPUT_BASE_PATH>/<dataset>/<date>.jsonl. Each CSV is
# its own dataset, named after the file's stem.
uv run kolkhoz run
uv run kolkhoz run -d hio_leadership   # one dataset only
uv run kolkhoz run -n 20               # random sample of 20 page inputs
uv run kolkhoz run -c 10               # up to 10 concurrent captures
```

## Evaluation

Score the extraction pipeline against hand-authored fixture pages. Each fixture is a directory under `fixtures/` holding `page.html`, an `expected.json` answer key, and optional `screenshot.png` and `url.txt` inputs. The harness derives page text and metadata, runs the real `extract()`, and exact-scores the returned holding observations. See `evaluate.py`'s docstring for details.

```bash
uv run python evaluate.py   # run all fixtures
uv run python evaluate.py -v # show expected pairs
```
