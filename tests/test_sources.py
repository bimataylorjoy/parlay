import json

import pytest

from parlay.data.sources import acquire_csv, detect_season, FOOTBALL_DATA_EPL_SEASONS


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


def test_detect_season_from_url():
    assert detect_season("https://www.football-data.co.uk/mmz4281/2324/E0.csv") == "2023-2024"
    assert detect_season("https://www.football-data.co.uk/mmz4281/1718/E0.csv") == "2017-2018"
    assert detect_season("/local/path/E0.csv") == "unknown"


def test_detect_season_in_local_path():
    assert detect_season("/data/mmz4281/2122/E0.csv") == "2021-2022"


def test_acquire_stores_detected_season(tmp_path):
    # Simulate a file whose "source" path contains a season pattern
    season_dir = tmp_path / "mmz4281" / "2223"
    season_dir.mkdir(parents=True)
    source = season_dir / "E0.csv"
    source.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n", encoding="utf-8")
    _, metadata = acquire_csv(source, tmp_path / "snapshots")
    assert metadata["season"] == "2022-2023"


def test_epl_season_catalog_covers_recent():
    assert "2023-24" in FOOTBALL_DATA_EPL_SEASONS
    assert all(url.startswith("https://") for url in FOOTBALL_DATA_EPL_SEASONS.values())
