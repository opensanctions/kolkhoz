"""Build a golden set of PEP holders from OpenSanctions for our target datasets.

The golden set is the ground truth we compare kolkhoz extractions against. It
is sliced out of an OpenSanctions PEP export (an FTM entity stream) by keeping
only the Occupancy entities that the OpenSanctions crawler for each of our
target datasets emitted, plus the Person and Position entities they link.

The target datasets are the ones in `data/opensanctions_peps.csv` — the same
pages kolkhoz snapshots and extracts. Each institute there maps to an
OpenSanctions dataset key via the local datasets/ checkout used by
`opensanctions_csv.py`.

The PEP export is downloaded from OpenSanctions on first run and cached
locally (~950 MB uncompressed, far smaller over the wire with gzip). To
force a refresh, delete the cache file.

Output (written under `data/`):

- `golden.csv`     one row per (dataset, human, position). The flat, diffable
                   form of the golden set, keyed so it can be joined against a
                   kolkhoz extraction per URL.
- `golden.ftm.json` the same data as an FTM entity stream (Occupancy + Person
                   + Position), matching the format `kolkhoz.py export-ftm`
                   emits, so the two can be compared at the entity level.

    uv run python golden_set.py        # download (cached) + build
"""

import csv
import json
import sys
from pathlib import Path

import click
import httpx
import yaml
from normality import squash_spaces

# The OpenSanctions PEP collection export, served as an FTM entity stream.
# See datasets/_collections/peps.yml in the opensanctions repo.
PEPS_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"
# Local cache for the downloaded export, alongside the other data files.
PEPS_CACHE = Path("data/peps.ftm.json")
DEFAULT_DATASETS_DIR = Path.home() / "Projects" / "opensanctions" / "datasets"
DEFAULT_INPUT_CSV = Path("data/opensanctions_peps.csv")
DEFAULT_OUT_CSV = Path("data/golden.csv")
DEFAULT_OUT_FTM = Path("data/golden.ftm.json")


def load_targets(input_csv: Path) -> list[dict]:
    """Return the target rows (institute, position, url) from the input CSV."""
    with open(input_csv) as f:
        return list(csv.DictReader(f))


def build_name_index(datasets_dir: Path) -> dict[str, str]:
    """Map a human-readable dataset name (title or publisher name) → key.

    The kolkhoz input CSV carries institutes as publisher names; OpenSanctions
    attributes Occupancies by dataset *key*. We bridge the two by indexing
    every dataset's title and publisher name to its key. Titles win over
    publisher names (set first), since the title is the dataset's identity.
    """
    index: dict[str, str] = {}
    for yml in sorted(datasets_dir.rglob("*.yml")):
        if "_collections" in yml.parts:
            continue
        data = yaml.safe_load(yml.read_text()) or {}
        key = yml.stem
        title = squash_spaces(str(data.get("title") or "").strip())
        if title:
            index.setdefault(title, key)
        publisher = data.get("publisher") or {}
        for field in ("name_en", "name"):
            name = squash_spaces(str(publisher.get(field) or "").strip())
            if name:
                index.setdefault(name, key)
    return index


def resolve_dataset_keys(targets: list[dict], index: dict[str, str]) -> dict[str, dict]:
    """Resolve each target row's institute to a dataset key.

    Returns {dataset_key: {institute, position, url}}. Fails loud (raises) if
    any institute can't be resolved — a silent drop here would shrink the
    golden set without warning.
    """
    by_key: dict[str, dict] = {}
    missing = []
    for row in targets:
        key = index.get(row["institute"])
        if key is None:
            missing.append(row["institute"])
            continue
        by_key[key] = row
    if missing:
        raise SystemExit(f"unresolved institutes (no dataset key): {missing}")
    return by_key


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


def build_golden(
    peps_ftm: Path, target_keys: set[str]
) -> tuple[list[dict], dict[str, dict]]:
    """Slice the golden entities out of the PEP export.

    Two streaming passes over the export (it is ~600 MB, so we never hold it
    all in memory):

    Pass 1 — collect every Occupancy whose ``datasets`` includes one of our
    target keys, and accumulate the Person/Position ids they reference.

    Pass 2 — pick up the Person and Position entities needed to name those
    occupancies.

    Returns (occupancy_records, entities_by_id) where each occupancy record
    carries its dataset key, holder/post ids, and status.
    """
    print(f"pass 1: scanning occupancies for {len(target_keys)} dataset(s)…")

    # dataset_key -> list of occupancy records
    by_dataset: dict[str, list[dict]] = {k: [] for k in target_keys}
    needed_persons: set[str] = set()
    needed_positions: set[str] = set()

    for ent in iter_entities(peps_ftm):
        if ent["schema"] != "Occupancy":
            continue
        datasets = set(ent.get("datasets") or [])
        hit_keys = datasets & target_keys
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
        rec = {
            "id": ent["id"],
            "holder": holder,
            "post": post,
            "status": status,
            "datasets": sorted(hit_keys),
        }
        # An occupancy can belong to several of our target datasets at once
        # (e.g. a person shared across two rosters); record it under each.
        for key in hit_keys:
            by_dataset[key].append(rec)

    n_occ = sum(len(v) for v in by_dataset.values())
    print(
        f"  {n_occ} occupancies, "
        f"{len(needed_persons)} persons, {len(needed_positions)} positions"
    )

    print("pass 2: resolving person/position names…")
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

    records = []
    for key, occs in by_dataset.items():
        for occ in occs:
            person = entities_by_id.get(occ["holder"]) if occ["holder"] else None
            position = entities_by_id.get(occ["post"]) if occ["post"] else None
            records.append(
                {
                    "dataset": key,
                    "occupancy_id": occ["id"],
                    "human": _first_prop(person, "name"),
                    "position": _first_prop(position, "name"),
                    "status": occ["status"],
                    "person_id": occ["holder"],
                    "position_id": occ["post"],
                }
            )
    return records, entities_by_id


def _first_prop(entity: dict | None, name: str) -> str | None:
    if entity is None:
        return None
    props = entity.get("properties") or {}
    values = props.get(name) or []
    return values[0] if values else None


def write_csv(records: list[dict], rows_by_key: dict[str, dict], out: Path) -> None:
    """Write the flat golden CSV, joined to institute/url/position per dataset."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "institute", "url", "human", "position", "status"])
        # Stable, diffable order.
        for rec in sorted(records, key=lambda r: (r["dataset"], r["human"] or "")):
            row = rows_by_key[rec["dataset"]]
            writer.writerow(
                [
                    rec["dataset"],
                    row["institute"],
                    row["url"],
                    rec["human"] or "",
                    rec["position"] or "",
                    rec["status"] or "",
                ]
            )
    print(f"wrote {len(records)} row(s) → {out}")


def write_ftm(records: list[dict], entities_by_id: dict[str, dict], out: Path) -> None:
    """Write the golden FTM stream: Occupancy + its Person and Position.

    One JSON entity per line, the same stream format `kolkhoz.py export-ftm`
    uses. Only entities that survive into the golden set are emitted, so the
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
    "-i",
    "--input",
    "input_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_INPUT_CSV,
    help="kolkhoz input CSV (institute, position, url).",
)
@click.option(
    "--datasets-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_DATASETS_DIR,
    help="OpenSanctions datasets/ directory.",
)
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
def cli(
    input_csv: Path,
    datasets_dir: Path,
    out_csv: Path,
    out_ftm: Path,
) -> None:
    targets = load_targets(input_csv)
    index = build_name_index(datasets_dir)
    rows_by_key = resolve_dataset_keys(targets, index)
    target_keys = set(rows_by_key)
    print(f"resolved {len(target_keys)} target dataset(s)")

    peps_path = fetch_peps(PEPS_URL, PEPS_CACHE)
    records, entities_by_id = build_golden(peps_path, target_keys)
    write_csv(records, rows_by_key, out_csv)
    write_ftm(records, entities_by_id, out_ftm)


if __name__ == "__main__":
    cli()
