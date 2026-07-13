"""Source ingestion: read input CSVs into typed rows.

Each CSV under the input directory is its own dataset, named after the file
stem. Rows with a blank URL are dropped.
"""

import csv
import os

import fsspec
from pydantic import BaseModel


class InputRow(BaseModel):
    """One row of the input CSV: a URL plus its known metadata."""

    organization: str
    url: str


def dataset_name(path: str) -> str:
    """Dataset name derived from the input CSV's filename stem."""
    return os.path.splitext(os.path.basename(path))[0]


def load_inputs(base_path: str) -> list[tuple[str, list[InputRow]]]:
    """Load every CSV under the input directory as a (dataset, rows) pair.

    *base_path* is an fsspec URL (local dir or ``gs://``/``s3://`` prefix).
    Each CSV becomes its own dataset, named after the file's stem; rows with
    a blank URL are dropped. Files are read straight from the bucket without
    a local copy.
    """
    fs, base = fsspec.core.url_to_fs(base_path)
    result: list[tuple[str, list[InputRow]]] = []
    for path in sorted(fs.glob(os.path.join(base, "*.csv"))):
        with fs.open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                InputRow(
                    organization=row["organization"].strip(),
                    url=row["url"].strip(),
                )
                for row in reader
                if row["url"].strip()
            ]
        result.append((dataset_name(path), rows))
    return result
