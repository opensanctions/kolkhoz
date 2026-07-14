"""Command-line interface.

A single Click group. Each command loads configuration once via
``load_config()`` and threads it — and the engine / model client it creates
on demand — into the focused handlers. No subsystem reads ``os.environ``
directly. Each command runs one ``asyncio.run`` over a single long-lived
Pravda instance (the snapshot and extract handlers share that instance across
all datasets/pages rather than opening one per item).
"""

import asyncio
import logging
import random
from datetime import datetime

import click
from openai import OpenAI
from pravda import migrate
from sqlalchemy.orm import Session

from kolkhoz.capture import (
    is_blank,
    latest_snapshot,
    pravda_client,
    read_artifact,
    run_snapshots,
    storage_filesystem,
)
from kolkhoz.config import Config, load_config
from kolkhoz.db import init_engine
from kolkhoz.extract import (
    extract,
    flatten_persons,
    metadata_from_html,
    screenshot_reason,
)
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
    help="Max concurrent Pravda captures.",
)
def snapshot_cmd(concurrency: int) -> None:
    config = load_config()
    engine = init_engine(config.database)
    inputs = load_inputs(config.paths.input_base_path)
    log.info("%d input CSV(s)", len(inputs))
    for dataset, rows in inputs:
        log.info("dataset %s: %d row(s)", dataset, len(rows))
    asyncio.run(run_snapshots(inputs, concurrency, config.pravda, engine))


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

    asyncio.run(run_extract(pages, config, engine, client))


async def run_extract(
    pages: list[PageRow], config: Config, engine, client: OpenAI
) -> None:
    """Extract holders from the newest successful snapshot of each page.

    Pravda's schema is migrated to head first (idempotently). Then one
    long-lived Pravda instance answers every page's history query. The
    OpenAI extraction and SQLite writes stay synchronous (the existing
    sequential behavior); only the snapshot lookup goes through Pravda's async
    instance API.
    """
    await migrate(config.pravda.database_url)

    fs = storage_filesystem(config.pravda)
    pravda = pravda_client(config.pravda)
    n = 0
    hits = 0
    async with pravda:
        with Session(engine) as session:
            for page in pages:
                snapshot = await latest_snapshot(pravda, page.url)
                if snapshot is None:
                    log.info("  skip %s — no snapshot", page.url)
                    continue

                snapshot_id = str(snapshot.id)
                already = (
                    session.query(ExtractionRow)
                    .filter_by(page_id=page.id, snapshot_id=snapshot_id)
                    .first()
                )
                if already is not None:
                    log.info(
                        "  skip %s — snapshot %s already extracted",
                        page.url,
                        snapshot_id,
                    )
                    continue

                text = read_artifact(fs, snapshot, snapshot.plaintext).decode(
                    "utf-8", errors="replace"
                )
                html = read_artifact(fs, snapshot, snapshot.rendered_html).decode(
                    "utf-8", errors="replace"
                )

                log.info("%s → extracting …", snapshot.url)
                screenshot_blob = None
                reason = screenshot_reason(text, html)
                if reason is not None:
                    log.info("  → %s → including screenshot", reason)
                    if snapshot.screenshot is not None:
                        blob = read_artifact(fs, snapshot, snapshot.screenshot)
                        if not is_blank(blob):
                            screenshot_blob = blob
                metadata = metadata_from_html(snapshot.url, html)
                extraction = extract(
                    client,
                    config.model,
                    config.image,
                    metadata,
                    text,
                    screenshot_blob,
                )
                holders = flatten_persons(extraction)
                log.info("%s → %d holder(s)", snapshot.url, len(holders))

                n += 1
                if holders:
                    hits += 1

                # Re-attach the page to this session (it was loaded above in a
                # different, now-closed session).
                page = session.merge(page)

                extraction_row = ExtractionRow(
                    page_id=page.id,
                    snapshot_id=snapshot_id,
                    snapshot_retrieved_at=snapshot.captured_at.isoformat(),
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
                            person_countries=h["person_countries"],
                            position_organization=h["position_organization"],
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
