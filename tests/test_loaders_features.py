from datetime import date

import pytest

from parlay.data.loaders import load_football_data_csv
from parlay.data.database import ResearchDatabase
from parlay.data.ingestion import ingest_csv_files
from parlay.features.historical import build_pre_match_features, build_pre_match_feature_sets


def test_loader_reads_football_data_format(tmp_path):
    path = tmp_path / "E0.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "01/01/24,A,B,2,1,1.80,3.50,4.50\n",
        encoding="utf-8",
    )
    matches, odds = load_football_data_csv(path, competition="EPL", season="2023-24")
    assert matches[0].home_team == "A"
    assert matches[0].home_goals == 2
    assert matches[0].kickoff_at.hour == 0 if matches[0].kickoff_at else True
    assert len(odds) == 3


def test_features_do_not_include_same_date_match():
    from parlay.data.schemas import Match
    rows = [
        Match("a", date(2024, 1, 1), "l", "s", "A", "B", 2, 0),
        Match("b", date(2024, 1, 1), "l", "s", "A", "C", 0, 1),
        Match("c", date(2024, 1, 2), "l", "s", "A", "D", 1, 0),
    ]
    features = build_pre_match_features(rows)
    assert features["a"]["home_matches"] == 0
    assert features["b"]["home_matches"] == 0
    assert features["c"]["home_matches"] == 2


def test_loader_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,HomeTeam,AwayTeam\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_football_data_csv(path)


def test_feature_sets_carry_temporal_provenance():
    from datetime import datetime, timezone
    from parlay.data.schemas import Match
    rows = [Match("a", date(2024, 1, 1), "l", "s", "A", "B", 2, 0, kickoff_at=datetime(2024, 1, 1, 15, tzinfo=timezone.utc))]
    feature_sets = build_pre_match_feature_sets(rows)
    feature_set = feature_sets["a"]
    assert feature_set.as_of.hour == 14
    assert feature_set.is_knowable(feature_set.as_of)
    assert not feature_set.is_knowable(feature_set.as_of - __import__("datetime").timedelta(minutes=1))


def test_ingestion_writes_database_and_manifest(tmp_path):
    path = tmp_path / "E0.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "01/01/24,A,B,2,1,1.80,3.50,4.50\n",
        encoding="utf-8",
    )
    database = ResearchDatabase()
    manifest_path = tmp_path / "manifest.json"
    manifest = ingest_csv_files([path], database, competition="EPL", season="2023-24", manifest_path=manifest_path)
    assert manifest["match_count"] == 1
    assert manifest["odds_count"] == 3
    assert manifest_path.exists()
    assert database.feature_as_of("bad", "rolling_v1", __import__("datetime").datetime.now()) is None
