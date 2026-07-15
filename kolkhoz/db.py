"""Async database lifecycle for the Postgres database shared with Pravda."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from kolkhoz.models import Base

__all__ = ["database_engine"]


@asynccontextmanager
async def database_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """Create the Kolkhoz tables and yield an engine that is always disposed."""
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
