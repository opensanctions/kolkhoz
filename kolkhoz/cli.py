"""Command-line interface.

A single Click group exposing a unified ``run`` pipeline (capture through
Pravda, then LLM extraction) and an ``export`` step. Each command loads
configuration once via ``load_config()`` and threads it — and the engine /
model client it creates on demand — into the focused handlers. No subsystem
reads ``os.environ`` directly. ``run`` opens one long-lived Pravda instance
shared across all datasets and pages rather than one per item.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime

import click
from openai import OpenAI
from pravda import Snapshot, migrate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from kolkhoz.capture import (
    capture_urls,
    is_blank,
    pravda_client,
    read_artifact,
    storage_filesystem,
)
from kolkhoz.config import Config, load_config
from kolkhoz.db import database_engine
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
from kolkhoz.sources import InputRow, load_inputs

log = logging.getLogger("kolkhoz")


@click.group()
def cli() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.FileHandler("kolkhoz.log"))
    root.addHandler(logging.StreamHandler())


@dataclass(frozen=True)
class ExtractionStatus:
    """Outcome of extracting one (page, snapshot) pair.

    ``wrote`` is True when an Extraction row (with its Holder rows) was
    created and staged on the session. ``holders`` is the holder count the
    model returned — meaningful only when ``wrote`` is True; it is 0 when the
    pair was skipped as already extracted.
    """

    wrote: bool
    holders: int


async def extract_snapshot(
    page: PageRow,
    snapshot: Snapshot,
    fs,
    config: Config,
    client: OpenAI,
    session: AsyncSession,
) -> ExtractionStatus:
    """Extract and persist holders from one (page, snapshot) pair.

    The reusable per-snapshot operation: dedup against existing Extraction
    rows, read the snapshot's plaintext and rendered HTML, decide whether the
    full-page screenshot carries content the text lacks, run the LLM
    extraction, and stage a new Extraction row (one Holder per
    person-position) on *session*. The caller owns the commit.

    A pair already extracted — an Extraction row keyed by (page, snapshot) —
    short-circuits before any read or model call and returns ``wrote=False``.
    """
    snapshot_id = str(snapshot.id)
    already = await session.scalar(
        select(ExtractionRow).filter_by(page_id=page.id, snapshot_id=snapshot_id)
    )
    if already is not None:
        log.info(
            "  skip %s — snapshot %s already extracted",
            page.url,
            snapshot_id,
        )
        return ExtractionStatus(wrote=False, holders=0)

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

    return ExtractionStatus(wrote=True, holders=len(holders))


async def run_pipeline(
    inputs: list[tuple[str, list[InputRow]]],
    sample: int | None,
    concurrency: int,
    config: Config,
    client: OpenAI,
) -> None:
    """Run the pipeline with a database engine that is always disposed."""
    await migrate(config.pravda.database_url)
    async with database_engine(config.pravda.database_url) as engine:
        await _run_pipeline(inputs, sample, concurrency, config, client, engine)


async def _run_pipeline(
    inputs: list[tuple[str, list[InputRow]]],
    sample: int | None,
    concurrency: int,
    config: Config,
    client: OpenAI,
    engine: AsyncEngine,
) -> None:
    """Capture and extract the selected page inputs.

    Builds one association per distinct (dataset, URL) across the selected
    datasets: a URL repeated within a dataset collapses to one association,
    while the same URL in several datasets is several associations backed by
    a single capture. Optionally samples *sample* associations before page
    registration and capture — sampling is over these (dataset, URL) page
    inputs, never over old database rows.

    Registers a Page row per association under the (dataset, URL) identity
    (existing rows are kept, so re-running over the same CSVs does no extra
    insert work), captures each unique selected URL once through one
    long-lived Pravda instance, then extracts sequentially from the exact
    Snapshot objects those captures returned — once per association —
    reusing ``extract_snapshot`` and its dedup. Pravda migrations run once;
    the Pravda instance and the shared artifact filesystem are shared across
    the whole operation. Captures Pravda persisted as failures (``error``
    set) are skipped with a clear warning rather than treated as success.
    Writes commit once at the end.
    """
    # One association per distinct (dataset, URL), keeping the first row's
    # organization. A URL repeated within a dataset collapses to one
    # association; the same URL across datasets is several associations.
    associations: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dataset, rows in inputs:
        for row in rows:
            key = (dataset, row.url)
            if key in seen:
                continue
            seen.add(key)
            associations.append((dataset, row.url, row.organization))

    if sample is not None and sample < len(associations):
        associations = random.sample(associations, sample)
    log.info("%d page association(s) selected", len(associations))

    urls = sorted({url for _, url, _ in associations})
    log.info("%d unique URL(s) to snapshot", len(urls))

    # Register Page rows under (dataset, URL) identity; existing rows are
    # left as-is so duplicate input rows do no insert work.
    async with AsyncSession(engine) as session:
        for dataset, url, organization in associations:
            page = await session.scalar(
                select(PageRow).filter_by(dataset=dataset, url=url)
            )
            if page is None:
                session.add(
                    PageRow(
                        url=url,
                        organization=organization,
                        dataset=dataset,
                    )
                )
        await session.commit()

    fs = storage_filesystem(config.pravda)
    pravda = pravda_client(config.pravda)
    async with pravda:
        # Capture each unique URL once; extract from the exact Snapshot
        # objects returned, never re-querying Pravda's history.
        captures = await capture_urls(pravda, urls, concurrency)

        async with AsyncSession(engine) as session:
            wrote = 0
            hits = 0
            for dataset, url, _ in associations:
                snapshot = captures[url]
                if snapshot.error is not None:
                    log.warning("  skip %s — capture failed: %s", url, snapshot.error)
                    continue
                page = await session.scalar(
                    select(PageRow).filter_by(dataset=dataset, url=url)
                )
                if page is None:
                    raise RuntimeError(f"page was not registered: {dataset} {url}")
                status = await extract_snapshot(
                    page, snapshot, fs, config, client, session
                )
                if status.wrote:
                    wrote += 1
                    if status.holders > 0:
                        hits += 1
            await session.commit()

    log.info("wrote %d extraction record(s) to Postgres", wrote)
    log.info("extraction: %d hit, %d miss", hits, wrote - hits)


@cli.command(
    "run",
    help=(
        "Unified pipeline: capture every URL from the CSVs in the input "
        "directory through Pravda, then extract position holders from each "
        "captured snapshot. Dataset filtering and random sampling happen "
        "before capture; each unique URL is captured once, concurrently."
    ),
)
@click.option("-d", "--dataset", type=str, default=None, help="Only run this dataset.")
@click.option(
    "-n",
    "--sample",
    type=click.IntRange(min=0),
    default=None,
    help="Randomly sample N page inputs.",
)
@click.option(
    "-c",
    "--concurrency",
    type=click.IntRange(min=1),
    default=5,
    help="Max concurrent Pravda captures.",
)
def run_cmd(dataset: str | None, sample: int | None, concurrency: int) -> None:
    config = load_config()
    client = OpenAI()
    inputs = load_inputs(config.paths.input_base_path)
    if dataset is not None:
        inputs = [(d, rows) for d, rows in inputs if d == dataset]
    log.info("%d input CSV(s)", len(inputs))
    for dataset_name, rows in inputs:
        log.info("dataset %s: %d row(s)", dataset_name, len(rows))
    asyncio.run(run_pipeline(inputs, sample, concurrency, config, client))


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

    async def export() -> None:
        async with database_engine(config.pravda.database_url) as engine:
            await run_export(engine, config.paths)

    asyncio.run(export())


if __name__ == "__main__":
    cli()
