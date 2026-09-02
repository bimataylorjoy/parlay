import json
from pathlib import Path

from parlay.cli import main

def test_predict_cli_outputs_recommendations(tmp_path):
    # Buat dummy database CSV
    db_csv = tmp_path / "E0.csv"
    db_csv.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
        + "\n".join(f"{i + 1:02d}/01/24,TeamA,TeamB,{i % 3},{(i + 1) % 2}" for i in range(10))
        + "\n",
        encoding="utf-8",
    )
    
    # Ingest
    db_path = tmp_path / "data.sqlite"
    manifest_path = tmp_path / "manifest.json"
    assert main(["ingest", str(db_csv), "--database", str(db_path), "--manifest", str(manifest_path)]) == 0
    
    # Buat fixtures CSV
    fixtures = tmp_path / "fixtures.csv"
    fixtures.write_text(
        "Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "25/01/24,TeamA,TeamB,2.0,3.5,4.0\n"
        "26/01/24,TeamC,TeamA,3.0,3.5,2.0\n",
        encoding="utf-8"
    )
    
    # Predict (we use heuristic here for speed in tests, mle takes slightly longer)
    assert main(["predict", str(fixtures), "--database", str(db_path), "--estimator", "heuristic"]) == 0
