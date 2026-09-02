from pathlib import Path

from parlay.cli import main


_TEAMS = [("Arsenal", "Chelsea"), ("Liverpool", "Man Utd"), ("Tottenham", "Everton"),
          ("Man City", "West Ham"), ("Leicester", "Wolves")]


def _make_csv(with_odds: bool = False) -> str:
    header = "Date,HomeTeam,AwayTeam,FTHG,FTAG"
    if with_odds:
        header += ",B365H,B365D,B365A"
    rows = [header]
    for i in range(20):
        home, away = _TEAMS[i % len(_TEAMS)]
        d = f"{(i % 28) + 1:02d}/01/24"
        hg, ag = i % 3, (i + 1) % 3
        line = f"{d},{home},{away},{hg},{ag}"
        if with_odds:
            line += ",2.0,3.5,4.0"
        rows.append(line)
    return "\n".join(rows) + "\n"


def test_backtest_cli_writes_artifacts(tmp_path):
    csv_path = tmp_path / "E0.csv"
    csv_path.write_text(_make_csv(with_odds=True), encoding="utf-8")
    output = tmp_path / "output"
    assert main(["backtest", str(csv_path), "--database", str(tmp_path / "empty.sqlite"), "--initial-train-days", "3", "--test-days", "2", "--output-dir", str(output)]) == 0
    assert (output / "predictions.csv").exists()
    assert (output / "metrics.json").exists()
    assert (output / "calibration.json").exists()


def test_ingest_cli_writes_manifest(tmp_path):
    csv_path = tmp_path / "E0.csv"
    csv_path.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n01/01/24,A,B,1,0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    snapshot_dir = tmp_path / "snapshots"
    assert main(["ingest", str(csv_path), "--manifest", str(manifest), "--database", str(tmp_path / "data.sqlite"), "--snapshot-dir", str(snapshot_dir)]) == 0
    assert manifest.exists()
    assert list(snapshot_dir.glob("*.csv"))


def test_backtest_cli_reads_existing_database(tmp_path):
    csv_path = tmp_path / "E0.csv"
    csv_path.write_text(_make_csv(with_odds=True), encoding="utf-8")
    database = tmp_path / "data.sqlite"
    manifest = tmp_path / "manifest.json"
    assert main(["ingest", str(csv_path), "--manifest", str(manifest), "--database", str(database)]) == 0
    output = tmp_path / "db-output"
    assert main(["backtest", "--database", str(database), "--initial-train-days", "3", "--test-days", "2", "--output-dir", str(output)]) == 0
    assert (output / "predictions.csv").exists()


def test_compare_cli_writes_summary(tmp_path):
    csv_path = tmp_path / "E0.csv"
    csv_path.write_text(_make_csv(), encoding="utf-8")
    output = tmp_path / "comparison"
    assert main(["compare", str(csv_path), "--database", str(tmp_path / "missing.sqlite"), "--initial-train-days", "3", "--test-days", "2", "--output-dir", str(output)]) == 0
    assert (output / "model_comparison.json").exists()
