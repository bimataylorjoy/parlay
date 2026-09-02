from datetime import date, datetime, timezone

from parlay.cli import main
from parlay.data.database import ResearchDatabase
from parlay.data.schemas import Match, OddsSnapshot


def test_predict_sportmonks_uses_completed_history_only(tmp_path):
    database_path = tmp_path / "data.sqlite"
    db = ResearchDatabase(str(database_path))
    history = [Match(str(i), date(2025, 1, 1 + i), "EPL", "2024-25", "A", "B", i % 2, (i + 1) % 2) for i in range(10)]
    future = Match("sportmonks:1", date(2026, 9, 5), "EPL", "2026-27", "A", "B", None, None, kickoff_at=datetime(2026, 9, 5, 14, tzinfo=timezone.utc))
    db.insert_matches(history + [future])
    db.insert_odds([
        OddsSnapshot("sportmonks:1", "sportmonks:2", "totals_2.5", "over", 2.0, datetime.now(timezone.utc)),
        OddsSnapshot("sportmonks:1", "sportmonks:2", "totals_2.5", "under", 2.0, datetime.now(timezone.utc)),
    ])
    output = tmp_path / "predictions.csv"
    assert main(["predict-sportmonks", "--database", str(database_path), "--estimator", "heuristic", "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "sportmonks:1" in text
    assert "over_probability" in text
