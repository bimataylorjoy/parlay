import json

import pytest

from parlay.data.sources import (
    FOOTBALL_DATA_EPL_SEASONS,
    FOOTBALL_DATA_LEAGUES,
    acquire_csv,
    detect_season,
    football_data_url,
)


def test_acquire_local_file_creates_content_addressed_snapshot(tmp_path):
    source = tmp_path / "E0.csv"
    source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n", encoding="utf-8")
    snapshot, metadata = acquire_csv(source, tmp_path / "snapshots")
    assert snapshot.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert metadata["sha256"]
    assert json.loads(snapshot.with_suffix(".csv.json").read_text())["source"] == str(source)


def test_acquire_missing_local_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        acquire_csv(tmp_path / "missing.csv", tmp_path / "snapshots")


def test_detect_season_from_url_or_path():
    assert detect_season("https://www.football-data.co.uk/mmz4281/2324/E0.csv") == "2023-2024"
    assert detect_season("/data/mmz4281/2122/E0.csv") == "2021-2022"
    assert detect_season("/local/path/E0.csv") == "unknown"


def test_football_data_urls_and_catalog():
    assert football_data_url("E0", 2025) == "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    assert football_data_url("E1", 1999) == "https://www.football-data.co.uk/mmz4281/9900/E1.csv"
    assert FOOTBALL_DATA_LEAGUES["E1"] == "Championship"
    assert "2023-24" in FOOTBALL_DATA_EPL_SEASONS
