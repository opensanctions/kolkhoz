# Kolkhoz

Kolkhoz is an orchestrator that turns raw web pages into structured data about political position holders. It is built on top of Pravda, the evidence layer that captures and stores durable snapshots of web pages.

## Project philosophy

- Early-stage. No backward compatibility. No fallback behaviors. Fail loud: no `try/except` unless there's a specific reason. We want errors to surface immediately.

## Stack

- **Python** 3.13+ managed by **uv**.
- **SQLite** for structured results (extracted humans, positions, links to Pravda snapshots).
- **Pravda** ([github.com/opensanctions/pravda](https://github.com/opensanctions/pravda)) for web page capture and storage. The base URL is set in `PRAVDA_URL` (see `.env`). Kolkhoz hits Pravda's FastAPI and reads returned file paths directly from disk.

## Project structure

```
kolkhoz.py         # the CLI: snapshot, extract, export
models.py          # SQLAlchemy domain (Page, Extraction, Holder)
evaluate.py        # score the extraction pipeline against synthetic fixtures
fixtures/          # one directory per fixture: page.html, expected.json, optional screenshot.png
```

`input/` (gitignored) holds the input CSVs (one dataset per file). `output/`
(gitignored) holds generated exports. Both are fsspec paths set via
`INPUT_BASE_PATH` / `OUTPUT_BASE_PATH` in `.env`.

## Conventions

- Dependencies are added with `uv add`. Don't edit `pyproject.toml` manually.
- Keep imports at the top of each file. No lazy imports unless there's a real cost.
- Environment-specific config goes in `.env`, loaded by `python-dotenv`.
- Read env vars with `os.environ` in the module that needs them.
- True constants (paths, format strings, etc.) live in the module that uses them.
- The user manages git commits, branching, etc.

## Running

```bash
# Install dependencies
uv sync

# Run a script
uv run python some_script.py
```

## Evaluation

Score the extraction pipeline against hand-authored fixture pages. Each
fixture is a directory under `fixtures/` holding `page.html`, an
`expected.json` answer key, and an optional `screenshot.png` that drives the
image path. The harness derives the plaintext the model reads, runs the real
`extract()`, and scores the returned (human, position) pairs by exact string
equality.

```bash
uv run python evaluate.py   # run all fixtures
uv run python evaluate.py -v # show expected pairs
```

See `evaluate.py`'s docstring for the fixture set and options.

## Linting and formatting

Pre-commit hooks run automatically on every commit:

- **ruff check --fix**
- **ruff format**
