"""Generate an input CSV (institute, position, url) from OpenSanctions crawlers.

Reads the crawler definitions in a local OpenSanctions checkout and emits one
row per HTML crawler that targets a PEP / leadership roster page. The output
matches the format consumed by `kolkhoz.py snapshot-csv`, so it can be fed
straight into the pipeline.

    uv run python opensanctions_csv.py -o data/opensanctions_peps.csv
    uv run python opensanctions_csv.py --all          # ignore list.pep tag
"""

import ast
import csv
from pathlib import Path

import click
import yaml
from normality import squash_spaces

# Default location of a local OpenSanctions checkout. Override with --path.
DEFAULT_DATASETS_DIR = Path.home() / "Projects" / "opensanctions" / "datasets"

# Only HTML crawlers are useful to us: Pravda snapshots rendered pages, not
# JSON/CSV/PDF feeds.
HTML_FORMATS = {"HTML"}


def load_crawler(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def position_from_crawler(crawler_path: Path) -> str | None:
    """Return the literal Position name declared in a crawler's crawler.py.

    Every OpenSanctions crawler builds its Position entity with
    `h.make_position(context, name, ...)`. About half the crawlers pass a
    hardcoded string literal as the name ("Member of the Sejm",
    "Member of the European Parliament"); the rest derive it at runtime
    from page content or an LLM translation prompt, so it can't be read
    statically.

    We AST-walk the crawler and collect the string literals passed as the
    name argument. No crawler currently declares more than one distinct
    literal position name; if several ever appear we take the first, since
    the goal is a single fallback role label per page. Returns None when
    the name is derived at runtime — those pages are left for the kolkhoz
    LLM extractor to label correctly on its own.
    """
    try:
        tree = ast.parse(crawler_path.read_text())
    except (OSError, SyntaxError):
        return None

    literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_call = (
            isinstance(func, ast.Attribute) and func.attr == "make_position"
        ) or (isinstance(func, ast.Name) and func.id == "make_position")
        if not is_call:
            continue
        # Position name is the second positional arg, or the `name` keyword.
        value = node.args[1] if len(node.args) >= 2 else None
        if value is None:
            for kw in node.keywords:
                if kw.arg == "name":
                    value = kw.value
                    break
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals.append(value.value)

    if not literals:
        return None
    return squash_spaces(literals[0].strip()) or None


def extract_row(yml_path: Path, yml: dict) -> tuple[str, str, str] | None:
    """Return (institute, position, url) for a crawler, or None to skip it.

    institute = the publishing body (publisher.name_en, falling back to
                publisher.name) — the Organisation the page is about.
    position  = the Position name. Prefer the literal declared in the
                crawler's crawler.py (`h.make_position(...)`), which is
                the authoritative role label for that roster. When the
                crawler derives the name at runtime (scraped from the page
                or LLM-translated) there's no static value to read, so we
                fall back to the dataset title — those pages are expected
                to be labelled correctly by the kolkhoz LLM extractor
                regardless.
    url       = the declared source page Pravda should snapshot.
    """
    data = yml.get("data") or {}
    fmt = str(data.get("format", "")).strip().upper()
    url = (data.get("url") or "").strip()
    if fmt not in HTML_FORMATS or not url:
        return None

    publisher = yml.get("publisher") or {}
    institute = squash_spaces(
        str(publisher.get("name_en") or publisher.get("name") or "").strip()
    )
    position = position_from_crawler(yml_path.parent / "crawler.py")
    if position is None:
        position = squash_spaces(str(yml.get("title", "")).strip())
    if not institute or not position:
        return None
    return institute, position, url


def iter_crawlers(datasets_dir: Path, pep_only: bool) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(datasets_dir.rglob("*.yml")):
        if "_collections" in path.parts:
            continue
        yml = load_crawler(path)
        if pep_only and "list.pep" not in (yml.get("tags") or []):
            continue
        row = extract_row(path, yml)
        if row is not None:
            rows.append(row)
    return rows


@click.command()
@click.option(
    "-p",
    "--path",
    "datasets_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_DATASETS_DIR,
    help="OpenSanctions datasets/ directory.",
)
@click.option(
    "--all",
    "pep_only",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Include every HTML crawler, not just list.pep ones.",
)
@click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="Output CSV file (default: stdout).",
)
def cli(datasets_dir: Path, pep_only: bool, output) -> None:
    rows = iter_crawlers(datasets_dir, pep_only)
    writer = csv.writer(output)
    writer.writerow(["institute", "position", "url"])
    writer.writerows(rows)
    output.flush()
    click.echo(f"wrote {len(rows)} row(s)", err=True)


if __name__ == "__main__":
    cli()
