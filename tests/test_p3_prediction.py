"""P3 tests: uncertainty-first API, anomaly, regime, diagnostics, performance."""

from datetime import date, datetime, timezone
import numpy as np
import pytest
from parlay.data.schemas import Match
from parlay.models.team_strength import fit_team_strength
from parlay.prediction.api import predict_with_uncertainty, PredictionResult
from parlay.evaluation.anomaly import diagnose
from parlay.evaluation.regime import analyze_by_regime
from parlay.evaluation.diagnostics import frequentist_diagnostics


def _matches(n=20):
    return [
        Match(match_id=f"m{i}", date=date(2024,1,1), competition="EPL", season="test", home_team="A", away_team="B", home_goals=1, away_goals=0, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc))
        for i in range(n)
    ]


def test_prediction_result_uncertainty_api():
    ms = _matches(20)
    model = fit_team_strength(ms, half_life_days=None)
    res = predict_with_uncertainty(model, "A", "B", n_historical=20)
    assert isinstance(res, PredictionResult)
    assert "home_win" in res.probabilities
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-9
    assert res.expected_rates.lambda_home_mean > 0
    assert res.uncertainty.decision in ("PASS","WATCH","RESEARCH_SIGNAL","REJECT")
    # Backward compat dict-like
    assert res["home_win"] == res.probabilities["home_win"]


def test_anomaly_extreme_disagreement():
    d = diagnose(0.79, 0.51, n_historical=20)
    assert "extreme_model_market_disagreement" in d.anomaly_flags
    # With high uncertainty -> REJECT
    d2 = diagnose(0.79, 0.51, posterior_std=0.2, n_historical=5, is_promoted=True)
    assert d2.decision == "REJECT"
    assert "wide_posterior_uncertainty" in d2.anomaly_flags


def test_regime_analysis():
    class R:
        def __init__(self, team, prob, actual):
            self.home_team=team; self.away_team="B"; self.home_win=prob; self.draw=(1-prob)/2; self.away_win=(1-prob)/2; self.actual=actual
            self.forecast_timestamp="2024-01-01T00:00:00+00:00"
    records = [R("Coventry", 0.6, "home_win") for _ in range(35)] + [R("Arsenal", 0.6, "home_win") for _ in range(40)]
    out = analyze_by_regime(records, lambda r: "promoted" if r.home_team=="Coventry" else "established", min_n=30)
    assert "promoted" in out and "established" in out
    assert out["promoted"]["n"] == 35
    # insufficient sample
    out2 = analyze_by_regime(records[:10], lambda r: "tiny", min_n=30)
    assert "insufficient_sample" in out2["tiny"]["note"]


def test_performance_caching():
    ms = _matches(30)
    m1 = fit_team_strength(ms, half_life_days=365)
    m2 = fit_team_strength(ms, half_life_days=365)  # should hit cache
    assert m1 is m2  # same object from cache
    # Different cutoff should miss
    ms2 = ms + [Match(match_id="extra", date=date(2024,1,2), competition="EPL", season="test", home_team="A", away_team="B", home_goals=2, away_goals=1, kickoff_at=datetime(2024,1,2,15,0,tzinfo=timezone.utc))]
    m3 = fit_team_strength(ms2, half_life_days=365)
    assert m3 is not m1


def test_diagnostics_frequentist():
    # Simulate MLE result tuple
    import numpy as np
    att = np.zeros(1)
    res = (att, att, 0.5, 0.2, 0.0, True, 123.0)
    d = frequentist_diagnostics(res)
    assert d["converged"] is True
    res_fail = (att, att, 0.5, 0.2, 0.0, False, 999)
    d2 = frequentist_diagnostics(res_fail)
    assert "failed" in d2["warning"]
