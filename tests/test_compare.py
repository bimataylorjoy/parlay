from datetime import date, timedelta

from parlay.data.schemas import Match
from parlay.evaluation.compare import compare_models


def _make_matches(n: int = 10):
    start = date(2020, 1, 1)
    return [Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2) for i in range(n)]


def test_compare_models_returns_all_baselines():
    matches = _make_matches()
    records, summary = compare_models(matches, initial_train_days=3, test_days=2)
    assert set(records) == {"poisson", "dixon_coles", "negative_binomial"}
    # Model rows exist (without market since odds are None)
    assert all(row["n"] > 0 for row in summary)


def test_compare_includes_per_fold_breakdown():
    matches = _make_matches(12)
    _, summary = compare_models(matches, initial_train_days=3, test_days=2)
    for row in summary:
        assert "per_fold" in row
        assert len(row["per_fold"]) >= 1
        fold = row["per_fold"][0]
        assert "fold_index" in fold
        assert "log_loss" in fold
        assert "brier_score" in fold
        assert "n" in fold
        assert "train_end" in fold
        assert "test_start" in fold
        assert "test_end" in fold


def test_compare_includes_stability_stats():
    matches = _make_matches(12)
    _, summary = compare_models(matches, initial_train_days=3, test_days=2)
    for row in summary:
        for key in ("stability_log_loss", "stability_brier"):
            assert key in row
            stats = row[key]
            assert set(stats) == {"mean", "std", "min", "max"}
            assert stats["std"] >= 0
            assert stats["min"] <= stats["mean"] <= stats["max"]
