"""Stable, dependency-light serialization for experiment artifacts."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Iterable


def write_predictions(records: Iterable[object], path: str | Path) -> None:
    rows = [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("\n", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
