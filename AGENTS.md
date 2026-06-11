# Kolkhoz

Kolkhoz is an orchestrator that turns raw web pages into structured data about political position holders. It is built on top of Pravda, the evidence layer that captures and stores durable snapshots of web pages.

## Project philosophy

- Early-stage. No backward compatibility. No fallback behaviors.
- Quick and dirty scripts over frameworks. Iterate fast, throw things away.
- Shared logic lives in the `kolkhoz` package. Scripts import from it.
- If two approaches exist, prefer the simpler one.

## Stack

- **Python** 3.13+ managed by **uv**.
- **SQLite** for structured results (extracted humans, positions, links to Pravda snapshots).
- **Pravda** ([github.com/opensanctions/pravda](https://github.com/opensanctions/pravda)) for web page capture and storage. Runs locally at `http://127.0.0.1:8000`. Kolkhoz hits Pravda's FastAPI and reads returned file paths directly from disk. OpenAPI spec is at `http://127.0.0.1:8000/openapi.json`.

## Project structure

```
kolkhoz/           # shared library code
scripts/           # standalone scripts for running experiments
```

## Conventions

- Dependencies are added with `uv add`. Don't edit `pyproject.toml` manually.
- Keep imports at the top of each file. No lazy imports unless there's a real cost.
- The user manages git commits, branching, etc.

## Running

```bash
# Install dependencies
uv sync

# Run a script
uv run python scripts/some_experiment.py
```

## Adding dependencies

```bash
uv add <package>
```

## Linting and formatting

Pre-commit hooks run automatically on every commit:

- **ruff check --fix**
- **ruff format**
