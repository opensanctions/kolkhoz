"""Command-line interface for the capture, extraction, and output pipeline."""

import asyncio
import logging
import random

import click
from openai import OpenAI
from pravda import Snapshot, migrate

from kolkhoz.capture import (
    capture_urls,
    is_blank,
    pravda_client,
    read_artifact,
    storage_filesystem,
)
from kolkhoz.config import Config, load_config
from kolkhoz.export import holder_to_record, write_outputs
from kolkhoz.extract import (
    extract,
    flatten_persons,
    metadata_from_html,
    screenshot_reason,
)
from kolkhoz.sources import InputRow, load_inputs

log = logging.getLogger("kolkhoz")


@click.group()
def cli() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.FileHandler("kolkhoz.log"))
    root.addHandler(logging.StreamHandler())


async def extract_snapshot(
    snapshot: Snapshot,
    fs,
    config: Config,
    client: OpenAI,
) -> list[dict]:
    """Extract flattened holder observations from one captured snapshot."""
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
    return holders


async def run_pipeline(
    inputs: list[tuple[str, list[InputRow]]],
    sample: int | None,
    concurrency: int,
    config: Config,
    client: OpenAI,
) -> None:
    """Apply Pravda migrations, then capture, extract, and write this run."""
    await migrate(config.pravda.database_url)
    await _run_pipeline(inputs, sample, concurrency, config, client)


async def _run_pipeline(
    inputs: list[tuple[str, list[InputRow]]],
    sample: int | None,
    concurrency: int,
    config: Config,
    client: OpenAI,
) -> None:
    """Capture and extract selected inputs, then write only these results.

    A URL repeated within a dataset collapses to one page association. The
    same URL in several datasets remains several associations, backed by one
    Pravda capture. Sampling happens over associations before capture.
    Kolkhoz keeps no database state of its own; Pravda alone persists the
    captured evidence.
    """
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

    # Replace the output for every loaded dataset, including those with no
    # sampled pages, successful captures, or extracted holders.
    groups: dict[str, list[dict]] = {dataset: [] for dataset, _ in inputs}
    fs = storage_filesystem(config.pravda)
    pravda = pravda_client(config.pravda)
    async with pravda:
        captures = await capture_urls(pravda, urls, concurrency)

        extracted = 0
        hits = 0
        for dataset, url, organization in associations:
            snapshot = captures[url]
            if snapshot.error is not None:
                log.warning("  skip %s — capture failed: %s", url, snapshot.error)
                continue
            missing = [
                name
                for name, value in (
                    ("storage prefix", snapshot.prefix),
                    ("plaintext", snapshot.plaintext),
                    ("rendered HTML", snapshot.rendered_html),
                )
                if value is None
            ]
            if missing:
                log.warning(
                    "  skip %s — capture missing required artifact metadata: %s",
                    url,
                    ", ".join(missing),
                )
                continue

            holders = await extract_snapshot(snapshot, fs, config, client)
            groups[dataset].extend(
                holder_to_record(dataset, url, organization, snapshot, holder)
                for holder in holders
            )
            extracted += 1
            if holders:
                hits += 1

    write_outputs(groups, config.paths)
    log.info("extraction: %d hit, %d miss", hits, extracted - hits)


@cli.command(
    "run",
    help=(
        "Capture URLs from the input CSVs through Pravda, extract position "
        "holders, and write this run's records as per-dataset JSONL. Dataset "
        "filtering and sampling happen before capture; each unique URL is "
        "captured once, concurrently."
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


if __name__ == "__main__":
    cli()
