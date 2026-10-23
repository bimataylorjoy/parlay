from datetime import date

import numpy as np
import pytest

from conftest import (
    TRUE_ATTACK,
    TRUE_DEFENSE,
    TRUE_HOME_ADVANTAGE,
    make_match,
    simple_series,
    synthetic_league,
)
from parlay.models.team_strength import GOAL_MODELS, fit_team_strength, time_decay_weights
from parlay.prediction.api import PredictionResult, predict_with_uncertainty


@pytest.mark.parametrize("model", GOAL_MODELS)
def test_all_models_share_prediction_interface(league, model):
    fitted = fit_team_strength(league, model=model)
    matrix = fitted.score_matrix("Lions", "Foxes")
    assert np.isclose(matrix.sum(), 1.0)
    home, away = fitted.expected_goals("Lions", "Foxes")
    assert home > away > 0
    assert fitted.fit_converged
    assert "Lions" in fitted and "Nobody" not in fitted


def test_recovers_team_ordering_from_simulated_league(league):
    fitted = fit_team_strength(league, model="poisson", half_life_days=None, l2_reg=0.0)
    by_attack = sorted(fitted.attack, key=fitted.attack.get, reverse=True)
    assert by_attack == sorted(TRUE_ATTACK, key=TRUE_ATTACK.get, reverse=True)
    assert fitted.defense["Lions"] > fitted.defense["Foxes"]
    assert fitted.home_advantage == pytest.approx(TRUE_HOME_ADVANTAGE, abs=0.12)
    assert max(abs(fitted.defense[t] - TRUE_DEFENSE[t]) for t in TRUE_DEFENSE) < 0.25


def test_dixon_coles_and_negative_binomial_parameters_are_estimated(league):
    assert -0.3 <= fit_team_strength(league, model="dixon_coles").rho <= 0.3
    nb = fit_team_strength(league, model="negative_binomial")
    assert nb.home_dispersion >= 0.5
    fixed = fit_team_strength(league, model="negative_binomial", dispersion=5.0)
    assert fixed.home_dispersion == 5.0


def test_neutral_venue_drops_home_advantage(league):
    fitted = fit_team_strength(league)
    home_venue = fitted.expected_goals("Bears", "Wolves")
    neutral = fitted.expected_goals("Bears", "Wolves", neutral=True)
    assert neutral[0] < home_venue[0]
    assert neutral[1] == pytest.approx(home_venue[1])


def test_unknown_team_and_future_as_of_are_rejected(league):
    fitted = fit_team_strength(league)
    with pytest.raises(KeyError):
        fitted.expected_goals("Unknown", "Lions")
    with pytest.raises(ValueError, match="before"):
        fit_team_strength(league, as_of=date(2000, 1, 1))


@pytest.mark.parametrize(
    "kwargs",
    [{"model": "elo"}, {"estimator": "heuristic"}, {"half_life_days": 0}, {"l2_reg": -1}, {"dispersion": 0}],
)
def test_invalid_arguments_are_rejected(league, kwargs):
    with pytest.raises(ValueError):
        fit_team_strength(league, **kwargs)


def test_rates_stay_inside_shared_log_domain():
    rows = [make_match(f"m{i}", date(2024, 1, 1), "Strong", "Weak", 9, 0) for i in range(20)]
    rows += [make_match(f"n{i}", date(2024, 1, 2), "Weak", "Strong", 0, 9) for i in range(20)]
    fitted = fit_team_strength(rows, model="poisson", half_life_days=None, l2_reg=0.0)
    home, away = fitted.expected_goals("Strong", "Weak")
    assert np.exp(-3) - 1e-9 <= away and home <= np.exp(3) + 1e-9


def test_fit_cache_hits_for_identical_inputs_only():
    rows = simple_series(20)
    first = fit_team_strength(rows)
    assert fit_team_strength(rows) is first
    assert fit_team_strength(rows, l2_reg=1.0) is not first
    assert fit_team_strength([*rows, make_match("x", date(2020, 2, 1), "A", "B", 3, 3)]) is not first
    assert fit_team_strength(rows, use_cache=False) is not first


def test_time_decay_weights_halve_every_half_life():
    weights = time_decay_weights([date(2024, 1, 1), date(2023, 1, 1)], date(2024, 1, 1), 365.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, rel=1e-2)
    assert np.all(time_decay_weights([date(2020, 1, 1)], date(2024, 1, 1), None) == 1.0)


def test_history_counts_are_exposed(league):
    fitted = fit_team_strength(league)
    assert fitted.historical_matches["Lions"] == 12 * 8
    assert 0 < fitted.effective_matches["Lions"] <= fitted.historical_matches["Lions"]


def test_uncertainty_api_returns_probabilities_and_diagnostics(league):
    fitted = fit_team_strength(league)
    result = predict_with_uncertainty(
        fitted, "Lions", "Foxes", market_odds={"home": 1.4, "draw": 4.8, "away": 8.0}
    )
    assert isinstance(result, PredictionResult)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result["home_win"] == result.probabilities["home_win"]
    assert result.expected_rates.lambda_home_mean > result.expected_rates.lambda_away_mean
    assert result.uncertainty.market_disagreement is not None
    assert result.model_metadata["uncertainty_source"] == "none"
    assert set(result.totals) == {1.5, 2.5, 2.75, 3.5}
    assert result.to_dict()["uncertainty"]["decision"] in {"PASS", "WATCH", "RESEARCH_SIGNAL", "REJECT"}


def test_sot_blend_mixes_shot_based_strengths():
    from dataclasses import replace

    rows = synthetic_league(rounds=4)
    # Shots on target say Foxes are the best attack even though goals say otherwise.
    rows = [
        replace(m, home_sot=12 if m.home_team == "Foxes" else 3, away_sot=12 if m.away_team == "Foxes" else 3)
        for m in rows
    ]
    goals_only = fit_team_strength(rows, sot_blend=0.0)
    blended = fit_team_strength(rows, sot_blend=0.5)
    assert blended.attack["Foxes"] > goals_only.attack["Foxes"]
    without_sot = [replace(m, home_sot=None, away_sot=None) for m in rows]
    assert fit_team_strength(without_sot, sot_blend=0.5).attack == pytest.approx(goals_only.attack)
    with pytest.raises(ValueError, match="sot_blend"):
        fit_team_strength(rows, sot_blend=1.5)
