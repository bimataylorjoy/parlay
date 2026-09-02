from datetime import date, timedelta

import numpy as np
import pytest

from parlay.data.schemas import Match
from parlay.models.team_strength import fit_team_strength


def matches():
    start = date(2024, 1, 1)
    rows = []
    for i in range(12):
        rows.append(Match(
            f"m{i}", start + timedelta(days=i), "league", "2024",
            "Strong" if i % 2 == 0 else "Weak",
            "Weak" if i % 2 == 0 else "Strong",
            3 if i % 2 == 0 else 0,
            0 if i % 2 == 0 else 1,
        ))
    return rows


@pytest.mark.parametrize("model", ["poisson", "dixon_coles", "negative_binomial"])
@pytest.mark.parametrize("estimator", ["mle", "heuristic"])
def test_all_models_share_prediction_interface(model, estimator):
    fitted = fit_team_strength(matches(), model=model, estimator=estimator, rho=0.01, dispersion=5.0)
    matrix = fitted.score_matrix("Strong", "Weak")
    assert matrix.shape == (11, 11)
    assert np.isclose(matrix.sum(), 1.0)
    assert fitted.expected_goals("Strong", "Weak")[0] > 0


def test_fit_auto_estimates_dixon_coles_rho():
    fitted = fit_team_strength(matches(), model="dixon_coles")
    assert -0.3 <= fitted.rho <= 0.3


def test_fit_auto_estimates_nb_dispersion():
    fitted = fit_team_strength(matches(), model="negative_binomial")
    assert fitted.home_dispersion >= 1.0
    assert fitted.away_dispersion >= 1.0


def test_unknown_team_is_rejected():
    fitted = fit_team_strength(matches())
    with pytest.raises(KeyError):
        fitted.expected_goals("Unknown", "Weak")


def test_future_as_of_is_rejected():
    with pytest.raises(ValueError, match="before"):
        fit_team_strength(matches(), as_of=date(2023, 12, 31))
