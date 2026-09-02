from datetime import date, timedelta

import numpy as np

from parlay.data.database import ResearchDatabase
from parlay.data.schemas import Match
from parlay.evaluation.backtest import run_backtest, run_backtest_full
from parlay.evaluation.metrics import brier_score, log_loss, outcome_label


def test_metrics_are_properly_defined():
    probabilities = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}
    assert outcome_label(2, 0) == "home_win"
    assert log_loss(probabilities, "home_win") == np.log(2)
    assert brier_score(probabilities, "home_win") == 0.375


def test_backtest_generates_scored_records_and_persists_them():
    start = date(2020, 1, 1)
    rows = [
        Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2)
        for i in range(12)
    ]
    records, metrics = run_backtest(rows, initial_train_days=5, test_days=2)
    assert records
    assert metrics["n"] == len(records)
    assert all(np.isfinite(record.log_loss) for record in records)
    db = ResearchDatabase()
    db.insert_matches(rows)
    db.insert_predictions(records)
    count = db.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert count == len(records)


def test_backtest_full_returns_per_fold_metrics():
    start = date(2020, 1, 1)
    rows = [
        Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2)
        for i in range(12)
    ]
    result = run_backtest_full(rows, initial_train_days=5, test_days=2)
    assert result.records
    assert result.metrics["n"] == len(result.records)
    assert len(result.fold_metrics) >= 1
    total_from_folds = sum(fm.n for fm in result.fold_metrics)
    assert total_from_folds == len(result.records)
    for fm in result.fold_metrics:
        assert fm.n > 0
        assert fm.log_loss > 0
        assert fm.brier_score > 0
        assert fm.train_end < fm.test_start
