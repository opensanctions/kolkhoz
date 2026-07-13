"""Command-line interface.

A single Click group. Each command loads configuration once via
``load_config()`` and threads it — and the engine / model client it creates
on demand — into the focused handlers. No subsystem reads ``os.environ``
directly.
"""

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path

import click
import httpx
from openai import OpenAI
from sqlalchemy.orm import Session

from kolkhoz.capture import is_blank, latest_snapshot, run_snapshot_csv
from kolkhoz.config import load_config
from kolkhoz.db import init_engine
from kolkhoz.extract import extract, flatten_persons, screenshot_reason
from kolkhoz.models import Extraction as ExtractionRow
from kolkhoz.models import Holder as HolderRow
from kolkhoz.models import Page as PageRow
from kolkhoz.export import run_export
from kolkhoz.sources import load_inputs

log = logging.getLogger("kolkhoz")


@click.group()
def cli() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.FileHandler("kolkhoz.log"))
    root.addHandler(logging.StreamHandler())


@cli.command(
    "snapshot",
    help="Snapshot all URLs from the CSVs in the input directory through Pravda.",
)
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=5,
    help="Max concurrent requests to Pravda.",
)
def snapshot_cmd(concurrency: int) -> None:
    config = load_config()
    engine = init_engine(config.database)
    inputs = load_inputs(config.paths.input_base_path)
    log.info("%d input CSV(s)", len(inputs))
    for dataset, rows in inputs:
        log.info("dataset %s: %d row(s)", dataset, len(rows))
        asyncio.run(run_snapshot_csv(rows, dataset, concurrency, config.pravda, engine))


@cli.command(
    "extract",
    help=(
        "Read pages from the database, fetch the latest Pravda snapshot for "
        "each, run an LLM extraction step, and store the results in the "
        "database."
    ),
)
@click.option(
    "-d", "--dataset", type=str, default=None, help="Only extract this dataset."
)
@click.option("-n", "--sample", type=int, default=None, help="Randomly sample N pages.")
def extract_cmd(dataset: str | None, sample: int | None) -> None:
    config = load_config()
    engine = init_engine(config.database)
    client = OpenAI()

    with Session(engine) as session:
        query = session.query(PageRow)
        if dataset is not None:
            query = query.filter_by(dataset=dataset)
        pages = query.all()

    if sample is not None and sample < len(pages):
        pages = random.sample(pages, sample)
    log.info("%d page(s) to extract", len(pages))

    n = 0
    hits = 0
    with Session(engine) as session, httpx.Client(timeout=30) as client_http:
        for page in pages:
            snapshot = latest_snapshot(config.pravda, client_http, page.url)
            if snapshot is None:
                log.info("  skip %s — no snapshot", page.url)
                continue

            already = (
                session.query(ExtractionRow)
                .filter_by(page_id=page.id, snapshot_id=snapshot["id"])
                .first()
            )
            if already is not None:
                log.info(
                    "  skip %s — snapshot %s already extracted",
                    page.url,
                    snapshot["id"],
                )
                continue

            prefix = snapshot["prefix"]
            text = (
                Path(prefix, snapshot["plaintext"])
                .read_bytes()
                .decode("utf-8", errors="replace")
            )
            html = (
                Path(prefix, snapshot["rendered_html"])
                .read_bytes()
                .decode("utf-8", errors="replace")
            )

            log.info("%s → extracting …", snapshot["url"])
            screenshot_blob = None
            reason = screenshot_reason(text, html)
            if reason is not None:
                log.info("  → %s → including screenshot", reason)
                if snapshot.get("screenshot") is not None:
                    blob = Path(prefix, snapshot["screenshot"]).read_bytes()
                    if not is_blank(blob):
                        screenshot_blob = blob
            extraction = extract(
                client, config.model, config.image, text, screenshot_blob
            )
            holders = flatten_persons(extraction)
            log.info("%s → %d holder(s)", snapshot["url"], len(holders))

            n += 1
            if holders:
                hits += 1

            # Re-attach the page to this session (it was loaded above in a
            # different, now-closed session).
            page = session.merge(page)

            extraction_row = ExtractionRow(
                page_id=page.id,
                snapshot_id=snapshot["id"],
                snapshot_retrieved_at=snapshot["captured_at"],
                model=config.model.name,
                extracted_at=datetime.now(),
            )
            for h in holders:
                extraction_row.holders.append(
                    HolderRow(
                        person_name=h["person_name"],
                        position_name=h["position_name"],
                        person_dob=h["person_dob"],
                        person_bio=h["person_bio"],
                        person_country=h["person_country"],
                        position_description=h["position_description"],
                        position_jurisdiction=h["position_jurisdiction"],
                        position_start_date=h["position_start_date"],
                        position_end_date=h["position_end_date"],
                        evidence_quotes=h["evidence_quotes"],
                    )
                )
            session.add(extraction_row)

        session.commit()

    log.info("wrote %d record(s) → %s", n, config.database.path)
    log.info("extraction: %d hit, %d miss", hits, n - hits)


@cli.command(
    "export",
    help=(
        "Export extracted holders as JSONL (one record per person-position "
        "observation). Writes one .jsonl file per dataset to "
        "<output-base>/<dataset>/<date>.jsonl."
    ),
)
def export_cmd() -> None:
    config = load_config()
    engine = init_engine(config.database)
    run_export(engine, config.paths)


if __name__ == "__main__":
    cli()
