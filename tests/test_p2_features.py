"""P2 feature registry and dynamic model tests."""

from datetime import date, datetime, timezone
from parlay.data.schemas import Match
from parlay.features.registry import FeatureValue, SOT_BETA_LEAGUE, get_sot_beta
from parlay.models.dynamic import fit_dynamic_strength, DynamicStrengthConfig
from parlay.evaluation.regime import analyze_by_regime
from parlay.evaluation.anomaly import diagnose
from parlay.evaluation.diagnostics import frequentist_diagnostics, distribution_diagnostics


def test_feature_value_provenance():
    fv = FeatureValue(value=1.5, source="football-data HST", computed_at=datetime(2024,1,1, tzinfo=timezone.utc), available_at=datetime(2024,1,1,15,0, tzinfo=timezone.utc))
    assert fv.source == "football-data HST"
    assert fv.available_at.tzinfo is not None


def test_sot_beta_league_specific():
    assert get_sot_beta("EPL") == SOT_BETA_LEAGUE["EPL"]
    assert get_sot_beta("Championship") == SOT_BETA_LEAGUE["Championship"]
    assert get_sot_beta("Unknown") == SOT_BETA_LEAGUE["default"]


def test_dynamic_model_benchmark():
    matches = [
        Match(match_id=f"m{i}", date=date(2024,1,1), competition="EPL", season="test", home_team="A", away_team="B", home_goals=1, away_goals=0, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc))
        for i in range(20)
    ]
    base = fit_dynamic_strength(matches, config=DynamicStrengthConfig(sigma_attack=0.05))
    assert base.model.startswith("dynamic_")
    assert "A" in base.teams


def test_regime_and_anomaly():
    class R:
        home_win=0.6; draw=0.2; away_win=0.2; actual="home_win"
        home_team="Coventry"; away_team="Hull"; forecast_timestamp="2024-01-02T15:00:00+00:00"
    # promoted regime
    from parlay.evaluation.regime import promoted_regime
    assert promoted_regime(R()) == "promoted_involved"
    d = diagnose(0.79, 0.51, posterior_std=0.2, n_historical=5, is_promoted=True)
    assert "wide_posterior_uncertainty" in d.anomaly_flags
    assert d.decision in ("REJECT","WATCH","RESEARCH_SIGNAL","PASS")


def test_diagnostics():
    diag = frequentist_diagnostics((None, None, None, None, None, True, 123.4))
    assert diag["converged"] is True
    dist = distribution_diagnostics([1,2,0], [1.2,1.8,0.5])
    assert "overdispersion" in dist
