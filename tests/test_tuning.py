from datetime import date, timedelta
from parlay.data.schemas import Match
from parlay.evaluation.tuning import tune_half_life, tune_half_life_nested

def _make_matches(n: int = 15):
    start = date(2020, 1, 1)
    return [Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2) for i in range(n)]

def test_tune_half_life_returns_sorted_results():
    matches = _make_matches()
    # Test a small set of candidates for speed
    candidates = [1.0, 5.0, None]
    results = tune_half_life(
        matches, model="poisson", estimator="mle", 
        half_life_candidates=candidates,
        initial_train_days=3, test_days=2
    )
    
    assert len(results) == len(candidates)
    
    # Results should be sorted by log_loss
    assert results[0]["log_loss"] <= results[-1]["log_loss"]
    
    # Check structure
    for r in results:
        assert "half_life_days" in r
        assert "log_loss" in r
        assert "brier_score" in r
        assert "folds" in r


def test_nested_tuning_reports_holdout_only():
    matches = _make_matches(30)
    result = tune_half_life_nested(
        matches, half_life_candidates=[1.0, None], initial_train_days=3,
        test_days=2, holdout_days=8,
    )
    assert "tuned_on" in result
    assert result["holdout_metrics"]["n"] > 0
    assert result["evaluated_on"].startswith("holdout >")
