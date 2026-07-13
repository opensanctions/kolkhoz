"""JSONL export of extracted holders.

One flat record per (person, position) observation, taking only the most
recent extraction per page (so re-running extraction does not multiply the
output). Writes one ``.jsonl`` file per dataset under
``<output-base>/<dataset>/<date>.jsonl``.
"""

import json
import logging
import os
from datetime import datetime

import fsspec
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kolkhoz.config import PathsConfig
from kolkhoz.models import Extraction as ExtractionRow
from kolkhoz.models import Holder as HolderRow
from kolkhoz.models import Page as PageRow

log = logging.getLogger("kolkhoz")

# The flat JSONL schema handed to zavod. One record per (person, position)
# observation extracted from one snapshot. Fixed key order, UTF-8, source
# wording preserved (no FtM normalization — that is zavod's job). Every
# field is a scalar — no nested objects — except ``person_countries`` and
# ``evidence_quotes``, which stay native lists.
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

# Export files are written to <output-base>/<dataset>/<date>.jsonl, where
# <date> is the export run date.
EXPORT_DATE_FORMAT = "%Y-%m-%d"


def holder_to_record(
    page: PageRow, extraction: ExtractionRow, holder: HolderRow
) -> dict:
    """Flatten one holder observation into a JSONL record.

    The schema is flat — every field is a scalar — except
    ``person_countries`` and ``evidence_quotes``, which stay native lists.
    All dates are
    plain source strings, copied verbatim with no parsing or reformatting.
    """
    record = {
        "dataset": page.dataset,
        "source_url": page.url,
        "snapshot_id": extraction.snapshot_id,
        "snapshot_retrieved_at": extraction.snapshot_retrieved_at,
        "organisation_name": page.organization,
        "person_name": holder.person_name,
        "person_dob": holder.person_dob,
        "person_bio": holder.person_bio,
        "person_countries": holder.person_countries,
        "position_name": holder.position_name,
        "position_organization": holder.position_organization,
        "position_description": holder.position_description,
        "position_jurisdiction": holder.position_jurisdiction,
        "position_start_date": holder.position_start_date,
        "position_end_date": holder.position_end_date,
        "evidence_quotes": holder.evidence_quotes,
    }
    return {name: record[name] for name in EXPORT_FIELDS}


def run_export(engine, paths: PathsConfig) -> None:
    """Export the latest extraction of every page as per-dataset JSONL."""
    with Session(engine) as session:
        # Only the most recent extraction per page, so re-running extraction
        # doesn't multiply the output.
        latest = (
            select(
                ExtractionRow.page_id.label("page_id"),
                func.max(ExtractionRow.id).label("extraction_id"),
            )
            .group_by(ExtractionRow.page_id)
            .subquery()
        )
        stmt = (
            select(HolderRow, ExtractionRow, PageRow)
            .join(latest, HolderRow.extraction_id == latest.c.extraction_id)
            .join(ExtractionRow, HolderRow.extraction_id == ExtractionRow.id)
            .join(PageRow, ExtractionRow.page_id == PageRow.id)
        )
        groups: dict[str, list[tuple[PageRow, ExtractionRow, HolderRow]]] = {}
        for holder, extraction, page in session.execute(stmt).all():
            groups.setdefault(page.dataset, []).append((page, extraction, holder))

    fs, base = fsspec.core.url_to_fs(paths.output_base_path)
    date = datetime.now().strftime(EXPORT_DATE_FORMAT)
    total = 0
    for group, group_rows in groups.items():
        out_dir = os.path.join(base, group)
        out_file = os.path.join(out_dir, f"{date}.jsonl")
        fs.makedirs(out_dir, exist_ok=True)
        with fs.open(out_file, "wb") as fh:
            for page, extraction, holder in group_rows:
                record = holder_to_record(page, extraction, holder)
                fh.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
                fh.write(b"\n")
        total += len(group_rows)
        log.info("wrote %d record(s) → %s", len(group_rows), out_file)
    log.info("exported %d record(s) across %d dataset(s)", total, len(groups))
