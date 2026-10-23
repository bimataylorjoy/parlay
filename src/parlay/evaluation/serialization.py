"""Stable, dependency-light serialization for experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write dict rows to CSV; the header is the union of keys in first-seen order."""
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(records: Iterable[Any], path: str | Path) -> None:
    write_rows([asdict(r) if is_dataclass(r) else dict(r) for r in records], path)  # type: ignore[arg-type]


def write_json(value: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
