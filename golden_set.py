"""Build a page-centric golden set of PEP holders from the OpenSanctions export.

The golden set is the ground truth we compare kolkhoz extractions against. It
is sliced straight out of the OpenSanctions PEP collection export (an FTM
entity stream) — no local datasets/ checkout.

kolkhoz snapshots and extracts from rendered HTML pages, so the golden set
keeps only holders whose `sourceUrl` is itself a renderable HTML page. A
dataset can be crawled from an HTML index yet link out to PDFs, spreadsheets,
or other documents (e.g. Bulgaria's judiciary declarations); those leaf
URLs have nothing for kolkhoz to extract and would only pollute the golden
set. We therefore filter at the sourceUrl level by rejecting known non-HTML
file extensions, rather than trusting the dataset's declared format.

The grain of the golden set is the **page**: kolkhoz snapshots a URL and
extracts holders from it, so each golden row attributes a holder to the page
they were scraped from — the holder's `sourceUrl` on their Person entity.
Persons with no `sourceUrl` have nothing page-addressable to test against and
are dropped.

A person may carry several sourceUrls (a list page plus a detail page, or a
second source they were merged with); we emit one row per sourceUrl, recording
exactly what OpenSanctions recorded rather than guessing which page is
"primary".

The export is downloaded on first run and cached locally (~950 MB
uncompressed, far smaller over the wire with gzip). To force a refresh,
delete the cache file.

Output (written under `data/`):

- `golden.csv`     one row per (page, holder). The flat, diffable form of the
                   golden set, keyed by sourceUrl so it can be joined against
                   a kolkhoz extraction per page.

    uv run python golden_set.py        # download (cached) + build
"""

import csv
import json
import sys
import urllib.parse
from pathlib import Path

import click
import httpx

# The OpenSanctions PEP collection export, served as an FTM entity stream.
# See datasets/_collections/peps.yml in the opensanctions repo.
PEPS_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"
# Local cache for the downloaded export, alongside the other data files.
PEPS_CACHE = Path("data/peps.ftm.json")
DEFAULT_OUT_CSV = Path("data/golden.csv")


def fetch_peps(url: str, cache: Path) -> Path:
    """Return a local path to the PEP export, downloading it if absent.

    The export is a single newline-delimited JSON stream (~950 MB). We
    download it once to disk and parse from there (the build needs two passes,
    so holding the live HTTP response open for both isn't an option). gzip is
    negotiated transparently by httpx, so the bytes on the wire are far
    smaller than the stored size. To force a refresh, delete the cache file.
    """
    if cache.exists():
        print(f"using cached peps export: {cache}", file=sys.stderr)
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".part")
    print(f"downloading {url} → {cache} …", file=sys.stderr)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    tmp.replace(cache)
    return cache


def iter_entities(path: Path):
    """Stream FTM entities, one JSON object per line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_prop(entity: dict | None, name: str) -> str | None:
    if entity is None:
        return None
    props = entity.get("properties") or {}
    values = props.get(name) or []
    return values[0] if values else None


# File extensions for documents and binaries kolkhoz cannot render as HTML.
# HTML pages come in endless shapes (no extension, .html, .aspx, .php, ...),
# so we reject a denylist of known non-HTML types rather than allowlisting.
# A URL with no extension is assumed to be a server-rendered page.
NON_HTML_EXTENSIONS = {
    # office documents
    "pdf",
    "doc",
    "docx",
    "rtf",
    "odt",
    "ods",
    "odp",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "csv",
    "tsv",
    # structured data / feeds
    "json",
    "xml",
    "rdf",
    "rss",
    "atom",
    "yaml",
    "yml",
    "txt",
    # archives
    "zip",
    "rar",
    "7z",
    "gz",
    "tar",
    "bz2",
    # images
    "jpg",
    "jpeg",
    "png",
    "gif",
    "webp",
    "svg",
    "bmp",
    "tif",
    "tiff",
    # audio / video
    "mp3",
    "mp4",
    "avi",
    "mov",
    "wmv",
    "flv",
    "webm",
    "m4a",
    "wav",
}


def _is_html_page(url: str) -> bool:
    """True if a sourceUrl points at a renderable HTML page, not a document.

    kolkhoz snapshots rendered HTML; a sourceUrl that resolves to a PDF,
    spreadsheet, image, or other binary has nothing to extract. We parse the
    URL path and reject known non-HTML file extensions. URLs whose final path
    segment has no extension are treated as HTML pages (the common case for
    server-rendered roster and detail pages).
    """
    last = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    if "." not in last:
        return True
    ext = last.rsplit(".", 1)[-1].lower()
    return ext not in NON_HTML_EXTENSIONS


def _source_urls(entity: dict | None) -> list[str]:
    """Return the distinct http(s) sourceUrls on an entity, in stored order."""
    if entity is None:
        return []
    props = entity.get("properties") or {}
    seen: set[str] = set()
    urls: list[str] = []
    for url in props.get("sourceUrl") or []:
        if url and url.startswith(("http://", "https://")) and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def build_golden(peps_ftm: Path) -> list[dict]:
    """Slice the page-centric golden set out of the PEP export.

    Two streaming passes over the export (it is ~950 MB, so we never hold it
    all in memory):

    Pass 1 — collect every Occupancy and accumulate the Person/Position ids
    it references. There is no dataset scoping here; the page-level sourceUrl
    filter in pass 3 is what keeps the golden set to pages kolkhoz can
    actually snapshot.

    Pass 2 — pick up the Person and Position entities needed to name those
    occupancies.

    Each surviving record carries its page (one of the holder's HTML
    sourceUrls), holder name, post name, status, and datasets. Occupancies
    whose Person has no renderable HTML sourceUrl are dropped silently —
    there is no page to snapshot and test against.
    """
    print("pass 1: scanning occupancies…")

    occupancies: list[dict] = []
    needed_persons: set[str] = set()
    needed_positions: set[str] = set()

    for ent in iter_entities(peps_ftm):
        if ent["schema"] != "Occupancy":
            continue
        props = ent.get("properties") or {}
        holder = (props.get("holder") or [None])[0]
        post = (props.get("post") or [None])[0]
        status = (props.get("status") or [None])[0]
        if holder:
            needed_persons.add(holder)
        if post:
            needed_positions.add(post)
        occupancies.append(
            {
                "id": ent["id"],
                "holder": holder,
                "post": post,
                "status": status,
                "datasets": sorted(ent.get("datasets") or []),
            }
        )

    print(
        f"  {len(occupancies)} occupancies, "
        f"{len(needed_persons)} persons, {len(needed_positions)} positions"
    )

    print("pass 2: resolving person/position entities…")
    entities_by_id: dict[str, dict] = {}
    for ent in iter_entities(peps_ftm):
        eid = ent["id"]
        if eid in needed_persons or eid in needed_positions:
            entities_by_id[eid] = ent

    print(
        f"  resolved {len(needed_persons & entities_by_id.keys())}/{len(needed_persons)} "
        f"persons, "
        f"{len(needed_positions & entities_by_id.keys())}/{len(needed_positions)} positions"
    )

    print("pass 3: attributing holders to pages…")
    records: list[dict] = []
    for occ in occupancies:
        person = entities_by_id.get(occ["holder"]) if occ["holder"] else None
        position = entities_by_id.get(occ["post"]) if occ["post"] else None
        pages = [u for u in _source_urls(person) if _is_html_page(u)]
        if not pages:
            continue
        human = _first_prop(person, "name")
        for page in pages:
            records.append(
                {
                    "page": page,
                    "datasets": ";".join(occ["datasets"]),
                    "human": human,
                    "position": _first_prop(position, "name"),
                    "status": occ["status"],
                }
            )
    print(f"  {len(records)} page rows")
    return records


def write_csv(records: list[dict], out: Path) -> None:
    """Write the flat golden CSV, one row per (page, holder)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "datasets", "human", "position", "status"])
        # Stable, diffable order: by page then holder name.
        for rec in sorted(records, key=lambda r: (r["page"], r["human"] or "")):
            writer.writerow(
                [
                    rec["page"],
                    rec["datasets"],
                    rec["human"] or "",
                    rec["position"] or "",
                    rec["status"] or "",
                ]
            )
    pages = {r["page"] for r in records}
    print(f"wrote {len(records)} row(s) across {len(pages)} page(s) → {out}")


@click.command()
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT_CSV,
    help="Output golden CSV.",
)
def cli(out_csv: Path) -> None:
    peps_path = fetch_peps(PEPS_URL, PEPS_CACHE)
    records = build_golden(peps_path)
    write_csv(records, out_csv)


if __name__ == "__main__":
    cli()
