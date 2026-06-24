"""SQLAlchemy models — the Kolkhoz domain.

Kolkhoz keeps its own data in SQLite. Pravda owns snapshots and their
content hashes; we only store the provenance that links a Page to the
Pravda snapshot we extracted from.

Three tables:

- ``Page``      one row per distinct URL. Carries the CSV metadata
                (institute, fallback position, dataset) and is the unit
                re-extraction is driven from.
- ``Extraction`` one per (Page, snapshot). Records which Pravda snapshot we
                read, which model extracted from it, when, and the resulting
                ``page_type``. Re-running extraction on a new snapshot makes
                a new row; Pravda tells us when content changed.
- ``Holder``    a named (human, position) pair found in one Extraction.
                Many per Extraction.
"""

import enum
from datetime import datetime

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class PageType(str, enum.Enum):
    roster = "roster"  # page that lists named position holders (board, council, staff, directory)
    profile = "profile"  # a single person's bio / CV / appointment page
    other = "other"  # about/contact/landing/article — not expected to list holders


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(unique=True, index=True)
    institute: Mapped[str]
    # Fallback position from the dataset, applied to holders the model left
    # without a title.
    position: Mapped[str]
    # Which CSV / dataset this page was sourced from.
    dataset: Mapped[str]

    extractions: Mapped[list["Extraction"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )


class Extraction(Base):
    __tablename__ = "extractions"
    __table_args__ = (UniqueConstraint("page_id", "snapshot_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), index=True
    )
    # Pravda snapshot this extraction read from. The evidence.
    snapshot_id: Mapped[str]
    model: Mapped[str]
    extracted_at: Mapped[datetime]
    page_type: Mapped[PageType]

    page: Mapped[Page] = relationship(back_populates="extractions")
    holders: Mapped[list["Holder"]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan"
    )


class Holder(Base):
    __tablename__ = "holders"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(
        ForeignKey("extractions.id", ondelete="CASCADE"), index=True
    )
    human: Mapped[str]
    # May be null on the page; filled from Page.position at write time.
    position: Mapped[str | None]

    extraction: Mapped[Extraction] = relationship(back_populates="holders")
