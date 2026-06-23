"""Shared driver for the staged extraction scripts.

Both stages do the same dance per record — check the needed artifact is present,
call the LLM, write the result back — and differ only in which artifact they
require and how they extract. `process_url` layers extraction fields onto a
stage-0 record (so stage 1/2 output is stage-0 output plus extraction); each stage
supplies a `requires` check and an `extract` coroutine. `run_batch` fans out
over records not already in the output and merges the results back.
"""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from kolkhoz.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

# requires(record) -> miss reason if a needed artifact is absent, else None.
Requires = Callable[[dict], str | None]
# extract(record) -> {status, reason, holders, usage}.
Extract = Callable[[dict], Awaitable[dict]]


async def process_url(
    record: dict,
    sem: asyncio.Semaphore,
    *,
    requires: Requires,
    extract: Extract,
) -> dict:
    async with sem:
        out = {**record, "model": os.environ["OPENAI_MODEL"]}

        missing = requires(record)
        if missing is not None:
            log.info("%s → miss (%s)", record["url"], missing)
            out.update(status="miss", reason=missing, holders=[], usage=None)
            return out

        log.info("%s → extracting …", record["url"])
        result = await extract(record)
        out.update(result)
        log.info(
            "%s → %s (%d holder(s))",
            record["url"],
            result["status"],
            len(result.get("holders", [])),
        )
        return out


async def run_batch(
    records: list[dict],
    out_path: Path,
    *,
    requires: Requires,
    extract: Extract,
    concurrency: int,
) -> list[dict]:
    """Process *records*, merge into *out_path*, and return this run's records.
    URLs already present in *out_path* are skipped."""
    by_url = {record["url"]: record for record in read_jsonl(out_path)}
    pending = [record for record in records if record["url"] not in by_url]
    log.info(
        "%d record(s) queued, %d already done, %d to process",
        len(records),
        len(by_url),
        len(pending),
    )
    if not pending:
        return []

    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *[
            process_url(record, sem, requires=requires, extract=extract)
            for record in pending
        ]
    )

    by_url.update({record["url"]: record for record in results})
    write_jsonl(out_path, by_url.values())
    log.info("Persisted %d record(s) to %s", len(results), out_path)
    return results
