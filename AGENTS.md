# Kolkhoz

Kolkhoz is an orchestrator that turns raw web pages into structured data about political position holders. It is built on top of Pravda, the evidence layer that captures and stores durable snapshots of web pages.

## Project philosophy

- Early-stage. No backward compatibility. No fallback behaviors. Fail loud: no `try/except` unless there's a specific reason. We want errors to surface immediately.

## Stack

- **Python** 3.13+ managed by **uv**.
- **SQLite** for structured results (extracted humans, positions, links to Pravda snapshots).
- **[Followthemoney](https://followthemoney.tech)** as the export model: extracted holders are emitted as Organization / Position / Person / Occupancy / Document entities.
- **Pravda** ([github.com/opensanctions/pravda](https://github.com/opensanctions/pravda)) for web page capture and storage. The base URL is set in `PRAVDA_URL` (see `.env`). Kolkhoz hits Pravda's FastAPI and reads returned file paths directly from disk.

## Project structure

```
kolkhoz.py         # the CLI: snapshot-csv, extract, export-ftm
models.py          # SQLAlchemy domain (Page, Extraction, Holder)
```

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

## Linting and formatting

Pre-commit hooks run automatically on every commit:

- **ruff check --fix**
- **ruff format**
