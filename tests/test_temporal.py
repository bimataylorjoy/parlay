from datetime import date, timedelta

from parlay.data.schemas import Match
from parlay.evaluation.temporal import expanding_window


def test_expanding_window_never_trains_on_test_dates():
    start = date(2024, 1, 1)
    rows = [Match(str(i), start + timedelta(days=i), "league", "2024", "A", "B", 1, 0) for i in range(10)]
    folds = expanding_window(rows, initial_train_days=3, test_days=2)
    assert folds
    for fold in folds:
        assert max(row.date for row in fold.train) < min(row.date for row in fold.test)
        assert fold.train_end < fold.test_start


def test_fixed_origin_keeps_training_set_frozen():
    start = date(2024, 1, 1)
    rows = [Match(str(i), start + timedelta(days=i), "league", "2024", "A", "B", 1, 0) for i in range(12)]
    folds = expanding_window(rows, initial_train_days=3, test_days=2, mode="fixed_origin")
    assert len(folds) > 1
    assert [m.match_id for m in folds[0].train] == [m.match_id for m in folds[1].train]


def test_rolling_origin_expands_training_set():
    start = date(2024, 1, 1)
    rows = [Match(str(i), start + timedelta(days=i), "league", "2024", "A", "B", 1, 0) for i in range(12)]
    folds = expanding_window(rows, initial_train_days=3, test_days=2, mode="rolling_origin")
    assert len(folds) > 1
    assert len(folds[1].train) > len(folds[0].train)
