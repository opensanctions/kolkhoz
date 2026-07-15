"""Write one run's extracted holders as per-dataset JSONL files."""

import json
import logging
import os
from datetime import datetime

import fsspec
from pravda import Snapshot

from kolkhoz.config import PathsConfig

log = logging.getLogger("kolkhoz")

# The flat JSONL schema handed to zavod. One record per (person, position)
# observation extracted from one snapshot. Fixed key order, UTF-8, source
# wording preserved (no FtM normalization — that is zavod's job).
EXPORT_FIELDS = [
    "dataset",
    "source_url",
    "snapshot_id",
    "snapshot_retrieved_at",
    "organisation_name",
    "person_name",
    "person_dob",
    "person_bio",
    "person_countries",
    "position_name",
    "position_organization",
    "position_description",
    "position_jurisdiction",
    "position_start_date",
    "position_end_date",
    "evidence_quotes",
]

EXPORT_DATE_FORMAT = "%Y-%m-%d"


def holder_to_record(
    dataset: str,
    source_url: str,
    organization: str,
    snapshot: Snapshot,
    holder: dict,
) -> dict:
    """Add input and snapshot provenance to one flattened holder."""
    record = {
        "dataset": dataset,
        "source_url": source_url,
        "snapshot_id": str(snapshot.id),
        "snapshot_retrieved_at": snapshot.captured_at.isoformat(),
        "organisation_name": organization,
        **holder,
    }
    return {name: record[name] for name in EXPORT_FIELDS}


def write_outputs(groups: dict[str, list[dict]], paths: PathsConfig) -> None:
    """Write exactly the records produced by the current run."""
    fs, base = fsspec.core.url_to_fs(paths.output_base_path)
    date = datetime.now().strftime(EXPORT_DATE_FORMAT)
    total = 0
    for dataset, records in groups.items():
        out_dir = os.path.join(base, dataset)
        out_file = os.path.join(out_dir, f"{date}.jsonl")
        fs.makedirs(out_dir, exist_ok=True)
        with fs.open(out_file, "wb") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
                fh.write(b"\n")
        total += len(records)
        log.info("wrote %d record(s) → %s", len(records), out_file)
    log.info("wrote %d record(s) across %d dataset(s)", total, len(groups))
