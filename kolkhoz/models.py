"""SQLAlchemy models — the Kolkhoz domain.

Kolkhoz keeps its own data in SQLite. Pravda owns snapshots and their
content hashes; we only store the provenance that links a Page to the
Pravda snapshot we extracted from.

Three tables:

- ``Page``      one row per distinct URL. Carries the CSV metadata
                (organization, dataset) and is the unit
                re-extraction is driven from.
- ``Extraction`` one per (Page, snapshot). Records which Pravda snapshot we
                read, which model extracted from it, and when. Re-running
                extraction on a new snapshot makes a new row; Pravda tells
                us when content changed.
- ``Holder``    a named (person, position) pair found in one Extraction.
                Many per Extraction.
"""

from datetime import datetime

from sqlalchemy import ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(unique=True, index=True)
    organization: Mapped[str]
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
    # When Pravda captured the snapshot we extracted from, stored verbatim
    # as the raw snapshot.captured_at string — no parsing or normalization.
    snapshot_retrieved_at: Mapped[str]
    model: Mapped[str]
    extracted_at: Mapped[datetime]

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
    person_name: Mapped[str]
    position_name: Mapped[str]
    # All fields below are source wording, verbatim. Dates are source strings.
    # Blank (None) when the page does not state them.
    person_dob: Mapped[str | None]
    person_bio: Mapped[str | None]
    person_countries: Mapped[list[str]] = mapped_column(JSON)
    position_organization: Mapped[str | None]
    position_description: Mapped[str | None]
    position_jurisdiction: Mapped[str | None]
    position_start_date: Mapped[str | None]
    position_end_date: Mapped[str | None]
    # One or more supporting quotes lifted from the page, verbatim.
    evidence_quotes: Mapped[list[str]] = mapped_column(JSON)

    extraction: Mapped[Extraction] = relationship(back_populates="holders")
