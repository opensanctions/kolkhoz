"""Database initialization.

Engine and schema creation happen on demand via ``init_engine``, never at
import time. Callers open sessions with SQLAlchemy's own ``Session`` bound to
the returned engine.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from kolkhoz.config import DatabaseConfig
from kolkhoz.models import Base

__all__ = ["Base", "Session", "init_engine"]


def init_engine(config: DatabaseConfig):
    """Create the parent directory, the SQLite file, and all tables.

    Returns an engine bound to the configured database path.
    """
    Path(config.path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(config.url)
    Base.metadata.create_all(engine)
    return engine
