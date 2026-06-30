"""Build a page-centric golden set of PEP holders from the OpenSanctions export.

The golden set is the ground truth we compare kolkhoz extractions against. It
is sliced straight out of the OpenSanctions PEP collection export (an FTM
entity stream) — no local datasets/ checkout.

kolkhoz snapshots and extracts from rendered HTML pages, so the golden set is
scoped to the OpenSanctions datasets that crawl such pages. We read the format
from the OpenSanctions catalog index (a single small JSON) rather than a
local checkout: an Occupancy is in scope only if one of its datasets declares
`data.format: HTML`. This drops Wikidata-derived datasets, JSON/CSV/PDF feeds,
and anything else kolkhoz could never snapshot.

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
- `golden.ftm.json` the Person and Position entities behind those rows, as an
                   FTM entity stream matching the format `kolkhoz.py
                   export-ftm` emits, so the two can be compared at the entity
                   level.

    uv run python golden_set.py        # download (cached) + build
"""

import csv
import json
import sys
from pathlib import Path

import click
import httpx

# The OpenSanctions PEP collection export, served as an FTM entity stream.
# See datasets/_collections/peps.yml in the opensanctions repo.
PEPS_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"
# Local cache for the downloaded export, alongside the other data files.
PEPS_CACHE = Path("data/peps.ftm.json")
# The OpenSanctions catalog index, listing every dataset with its crawler
# config. We use each entry's `data.format` to scope the golden set to
# HTML-page rosters. Small enough to fetch fresh each run.
CATALOG_URL = "https://data.opensanctions.org/datasets/latest/index.json"
DEFAULT_OUT_CSV = Path("data/golden.csv")
DEFAULT_OUT_FTM = Path("data/golden.ftm.json")


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


def load_html_dataset_keys(url: str) -> set[str]:
    """Return the names of OpenSanctions datasets crawled from an HTML page.

    The catalog index lists every dataset alongside its crawler config; we
    keep those whose `data.format` is HTML and that declare a source URL.
    kolkhoz can only snapshot rendered pages, so non-HTML sources (JSON/CSV/
    PDF/XML feeds, Wikidata-derived datasets) are out of scope. The index is
    small, so we fetch it fresh each run.
    """
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    catalog = resp.json()
    keys: set[str] = set()
    for entry in catalog.get("datasets") or []:
        data = entry.get("data") or {}
        if str(data.get("format", "")).upper() == "HTML" and data.get("url"):
            keys.add(entry["name"])
    if not keys:
        raise SystemExit(f"no HTML datasets found in catalog {url}")
    return keys


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


def build_golden(
    peps_ftm: Path,
    html_keys: set[str],
) -> tuple[list[dict], dict[str, dict]]:
    """Slice the page-centric golden set out of the PEP export.

    Two streaming passes over the export (it is ~950 MB, so we never hold it
    all in memory):

    Pass 1 — collect every Occupancy that belongs to at least one HTML-page
    dataset (`html_keys`), and accumulate the Person/Position ids they
    reference. An occupancy can span several datasets (e.g. a person shared
    between a national roster and Wikidata); we keep it as long as one of
    those datasets is an HTML page, and record only the in-scope keys.

    Pass 2 — pick up the Person and Position entities needed to name those
    occupancies.

    Each surviving record carries its page (the holder's sourceUrl), holder
    name, post name, status, and the in-scope datasets. Occupancies whose
    Person has no sourceUrl are dropped — there is no page to snapshot and
    test against.
    """
    print(f"pass 1: scanning occupancies for {len(html_keys)} HTML dataset(s)…")

    occupancies: list[dict] = []
    needed_persons: set[str] = set()
    needed_positions: set[str] = set()

    for ent in iter_entities(peps_ftm):
        if ent["schema"] != "Occupancy":
            continue
        hit_keys = set(ent.get("datasets") or []) & html_keys
        if not hit_keys:
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
                "datasets": sorted(hit_keys),
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
    dropped = 0
    for occ in occupancies:
        person = entities_by_id.get(occ["holder"]) if occ["holder"] else None
        position = entities_by_id.get(occ["post"]) if occ["post"] else None
        pages = _source_urls(person)
        if not pages:
            dropped += 1
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
                    "person_id": occ["holder"],
                    "position_id": occ["post"],
                }
            )
    print(f"  {len(records)} page rows ({dropped} occupancies dropped: no sourceUrl)")
    return records, entities_by_id


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


def write_ftm(records: list[dict], entities_by_id: dict[str, dict], out: Path) -> None:
    """Write the golden FTM stream: the Person and Position behind each row.

    One JSON entity per line, the same stream format `kolkhoz.py export-ftm`
    uses. Only entities that survived into the golden set are emitted, so the
    file is directly comparable to a kolkhoz export.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(out, "w") as f:
        for rec in records:
            for eid in (rec["person_id"], rec["position_id"]):
                if eid and eid in entities_by_id and eid not in seen:
                    seen.add(eid)
                    f.write(json.dumps(entities_by_id[eid], ensure_ascii=False))
                    f.write("\n")
                    n += 1
    print(f"wrote {n} supporting entit(ies) → {out}")


@click.command()
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT_CSV,
    help="Output golden CSV.",
)
@click.option(
    "--out-ftm",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT_FTM,
    help="Output golden FTM stream.",
)
def cli(out_csv: Path, out_ftm: Path) -> None:
    html_keys = load_html_dataset_keys(CATALOG_URL)
    print(f"{len(html_keys)} HTML dataset(s) in the OpenSanctions catalog")
    peps_path = fetch_peps(PEPS_URL, PEPS_CACHE)
    records, entities_by_id = build_golden(peps_path, html_keys)
    write_csv(records, out_csv)
    write_ftm(records, entities_by_id, out_ftm)


if __name__ == "__main__":
    cli()
