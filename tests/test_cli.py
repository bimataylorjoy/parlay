import json
from datetime import date, datetime, timezone

import pytest

from conftest import synthetic_league
from parlay.cli import build_parser, main
from parlay.data.database import ResearchDatabase
from parlay.data.schemas import Match, OddsSnapshot


def _league_csv(path, rounds=6):
    """football-data style CSV of the synthetic league, with Bet365 odds."""
    lines = ["Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,B365H,B365D,B365A,PSCH,PSCD,PSCA"]
    for m in synthetic_league(rounds=rounds, corners=True):
        lines.append(
            f"E0,{m.date:%d/%m/%Y},15:00,{m.home_team},{m.away_team},{m.home_goals},{m.away_goals},"
            f"{m.home_corners},{m.away_corners},2.1,3.4,3.6,2.0,3.5,3.8"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def database(tmp_path):
    csv_path = _league_csv(tmp_path / "E0.csv")
    db = tmp_path / "data.sqlite"
    assert main(["ingest", str(csv_path), "--database", str(db), "--manifest", str(tmp_path / "m.json"),
                 "--snapshot-dir", str(tmp_path / "snap")]) == 0  # fmt: skip
    return db


WALK = ["--initial-train-days", "60", "--test-days", "14"]


def test_parser_lists_every_command():
    help_text = build_parser().format_help()
    for command in (
        "ingest",
        "fetch",
        "backtest",
        "compare",
        "tune",
        "calibrate",
        "predict",
        "sync-sportmonks",
    ):
        assert command in help_text


def test_ingest_writes_manifest_and_snapshot(tmp_path, database):
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert manifest["match_count"] == 6 * 20
    assert list((tmp_path / "snap").glob("*.csv"))


def test_backtest_writes_artifacts_and_reports_market(tmp_path, database, capsys):
    out = tmp_path / "bt"
    assert (
        main(["backtest", "--database", str(database), *WALK, "--model", "poisson", "--output-dir", str(out)])
        == 0
    )
    for name in ("predictions.csv", "metrics.json", "calibration.json"):
        assert (out / name).exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["metrics"]["n"] > 0 and "closing_log_loss" in metrics["metrics"]
    assert "bookmaker de-vig" in capsys.readouterr().out


def test_backtest_from_csv_without_database(tmp_path):
    csv_path = _league_csv(tmp_path / "E0.csv", rounds=4)
    out = tmp_path / "bt"
    assert main(["backtest", str(csv_path), "--database", str(tmp_path / "none.sqlite"), *WALK,
                 "--output-dir", str(out)]) == 0  # fmt: skip
    assert (out / "predictions.csv").exists()


def test_missing_database_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["backtest", "--database", str(tmp_path / "missing.sqlite")])


def test_compare_and_tune(tmp_path, database):
    assert main(["compare", "--database", str(database), *WALK, "--output-dir", str(tmp_path / "cmp")]) == 0
    summary = json.loads((tmp_path / "cmp" / "model_comparison.json").read_text())
    assert {row["model"] for row in summary} >= {"poisson", "dixon_coles", "negative_binomial"}
    tune = ["tune", "--database", str(database), *WALK, "--model", "poisson", "--half-lives", "180", "none"]
    assert (
        main([*tune, "--l2-grid", "1", "--holdout-days", "20", "--output-dir", str(tmp_path / "tune")]) == 0
    )
    report = json.loads((tmp_path / "tune" / "tune_poisson.json").read_text())
    assert report["holdout_metrics"]["n"] > 0


def test_calibrate_reads_backtest_predictions(tmp_path, database, capsys):
    out = tmp_path / "bt"
    main(["backtest", "--database", str(database), *WALK, "--output-dir", str(out)])
    assert main(["calibrate", str(out / "predictions.csv")]) == 0
    assert "Optimal temperature" in capsys.readouterr().out


def test_predict_prints_and_writes_every_market(tmp_path, database, capsys):
    fixtures = tmp_path / "fixtures.csv"
    fixtures.write_text(
        "# comment lines are ignored\n"
        "Date,HomeTeam,AwayTeam,B365H,B365D,B365A,B365>2.5,B365<2.5,B365CRN>9.5,B365CRN<9.5\n"
        "2030-01-05,Lions,Foxes,3.0,3.5,2.5,1.9,1.9,1.9,1.9\n"
        "05/01/2030,Newcomers,Lions,,,,,,,\n",
        encoding="utf-8",
    )
    output = tmp_path / "pred.csv"
    assert main(["predict", str(fixtures), "--database", str(database), "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "Lions vs Foxes" in printed and "Priced selections" in printed
    assert "no pre-match history for Newcomers" in printed
    text = output.read_text(encoding="utf-8")
    assert "home_prob" in text and "corners_over_9_5_prob" in text and "unknown_team" in text


def test_predict_rejects_bad_fixture_files(tmp_path, database):
    bad = tmp_path / "bad.csv"
    bad.write_text("Date,Home\n2030-01-01,A\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing columns"):
        main(["predict", str(bad), "--database", str(database)])
    bad.write_text("Date,HomeTeam,AwayTeam\n31/31/2030,A,B\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Unsupported match date"):
        main(["predict", str(bad), "--database", str(database)])


def test_predict_sportmonks_uses_completed_history_only(tmp_path):
    path = tmp_path / "data.sqlite"
    db = ResearchDatabase(str(path))
    history = synthetic_league(rounds=3)
    future = Match("sportmonks:1", date(2030, 9, 5), "EPL", "2030-31", "Lions", "Foxes", None, None,
                   kickoff_at=datetime(2030, 9, 5, 14, tzinfo=timezone.utc))  # fmt: skip
    db.insert_matches([*history, future])
    now = datetime(2030, 9, 1, tzinfo=timezone.utc)
    db.insert_odds(
        [
            OddsSnapshot("sportmonks:1", "sportmonks:2", "totals_2.5", side, 2.0, now)
            for side in ("over", "under")
        ]
    )
    db.close()
    output = tmp_path / "predictions.csv"
    assert main(["predict-sportmonks", "--database", str(path), "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "Lions" in text and "over_2_5_prob" in text


def test_fetch_downloads_each_season(tmp_path, monkeypatch):
    import parlay.cli.commands.data as data_command

    source = _league_csv(tmp_path / "source.csv", rounds=1)
    requested = []

    def fake_acquire(url, snapshot_dir):
        requested.append(url)
        return source, {}

    monkeypatch.setattr(data_command, "acquire_csv", fake_acquire)
    db = tmp_path / "fetched.sqlite"
    fetch = ["fetch", "--league", "E0", "--from-season", "2024", "--to-season", "2025"]
    assert main([*fetch, "--database", str(db), "--manifest-dir", str(tmp_path / "man")]) == 0
    assert requested == [
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    ]
    assert len(ResearchDatabase(str(db)).load_matches()) == 20
