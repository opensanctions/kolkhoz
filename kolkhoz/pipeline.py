"""Shared driver for the tiered extraction scripts.

Both tiers do the same dance per URL — fetch the latest snapshot, check the
text-hash cache, call the LLM, write the result back — and differ only in which
artifact they require and how they extract. `process_url` holds the skeleton;
each tier supplies a `requires` check and an `extract` coroutine. `run_batch`
wraps the whole run: seed the cache from the existing output, fan out, merge,
and write.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from kolkhoz import pravda
from kolkhoz.extract import PROMPT_VERSION
from kolkhoz.utils import build_content_cache, read_jsonl, write_jsonl

# requires(snapshot) -> miss reason if a needed artifact is absent, else None.
Requires = Callable[[dict], str | None]
# extract(snapshot) -> {status, reason, holders, usage} for a cache miss.
Extract = Callable[[dict], Awaitable[dict]]


async def process_url(
    client: httpx.AsyncClient,
    url: str,
    content_cache: dict[str, dict],
    sem: asyncio.Semaphore,
    *,
    requires: Requires,
    extract: Extract,
    snapshot: dict,
) -> dict:
    async with sem:
        text_hash = pravda.content_hash(snapshot, pravda.TEXT)
        base = {
            "url": url,
            "snapshot_id": snapshot["id"],
            "text_hash": text_hash,
            "model": os.environ["OPENAI_MODEL"],
            "prompt_version": PROMPT_VERSION,
        }

        missing = requires(snapshot)
        if missing is not None:
            return {
                **base,
                "status": "miss",
                "reason": missing,
                "holders": [],
                "usage": None,
            }

        cached = content_cache.get(text_hash)
        if cached is not None:
            return {**base, **cached}

        result = await extract(snapshot)
        content_cache[text_hash] = result
        return {**base, **result}


PersistFilter = Callable[[dict], bool]


async def run_batch(
    urls: list[str],
    out_path: Path,
    *,
    requires: Requires,
    extract: Extract,
    persist_if: PersistFilter | None = None,
    concurrency: int,
    timeout: float,
    snapshot_by_url: dict[str, dict],
) -> list[dict]:
    """Process *urls*, merge into *out_path* (which doubles as the cache), and
    return this run's records."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_url = {record["url"]: record for record in read_jsonl(out_path)}
    content_cache = build_content_cache(list(by_url.values()))

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *[
                process_url(
                    client,
                    url,
                    content_cache,
                    sem,
                    requires=requires,
                    extract=extract,
                    snapshot=snapshot_by_url[url],
                )
                for url in urls
            ]
        )

    for record in results:
        if persist_if is None or persist_if(record):
            by_url[record["url"]] = {
                k: v for k, v in record.items() if k not in ("status", "reason")
            }
    write_jsonl(out_path, by_url.values())
    return results
