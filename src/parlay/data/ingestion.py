"""Reproducible ingestion orchestration and dataset manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from parlay.features.historical import build_pre_match_features, features_json

from .database import ResearchDatabase
from .loaders import load_many
from .normalization import TeamRegistry


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_csv_files(
    paths: Sequence[str | Path],
    database: ResearchDatabase,
    *,
    registry: TeamRegistry | None = None,
    competition: str = "unknown",
    season: str | None = None,
    feature_set: str = "rolling_v1",
    feature_window: int = 5,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Load CSVs, persist facts/features, and optionally write a manifest."""
    matches, odds = load_many(paths, registry, competition, season)
    database.insert_matches(matches)
    database.insert_odds(odds)
    features = build_pre_match_features(matches, window=feature_window)
    for match in matches:
        database.insert_feature_snapshot(
            match.match_id,
            datetime.combine(match.date, datetime.min.time(), tzinfo=timezone.utc),
            feature_set,
            features_json(features[match.match_id]),
        )
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "competition": competition,
        "season": season or "auto",
        "feature_set": feature_set,
        "feature_window": feature_window,
        "match_count": len(matches),
        "odds_count": len(odds),
    }
    if manifest_path is not None:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
