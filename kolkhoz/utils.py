import json
import os
from pathlib import Path

RESULT_FIELDS = ("status", "reason", "holders", "usage")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_content_cache(records: list[dict]) -> dict[str, dict]:
    """Seed a text-hash cache from prior results, filtering by current model."""
    return {
        record["text_hash"]: {field: record[field] for field in RESULT_FIELDS}
        for record in records
        if record.get("text_hash") and record.get("model") == os.environ["OPENAI_MODEL"]
    }
