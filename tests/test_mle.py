import numpy as np
import pytest

from parlay.models.mle import fit_poisson_mle


def test_fit_poisson_mle_recovers_known_hierarchy():
    # 3 teams: Team 0 beats everyone, Team 1 average, Team 2 loses to everyone
    home_idx = np.array([0, 0, 1, 1, 2, 2])
    away_idx = np.array([1, 2, 0, 2, 0, 1])
    home_goals = np.array([3, 4, 1, 2, 0, 1])
    away_goals = np.array([0, 0, 3, 1, 4, 2])
    weights = np.ones(6)

    att, defense, mu, gamma, rho, conv, nll = fit_poisson_mle(
        home_idx, away_idx, home_goals, away_goals, weights, n_teams=3
    )

    assert conv
    assert att[0] > att[1] > att[2]  # Team 0 attack strongest
    assert defense[0] > defense[1] > defense[2]  # Team 0 defense best (higher is better)
    assert np.isclose(np.sum(att), 0.0, atol=1e-5)
    assert np.isclose(np.sum(defense), 0.0, atol=1e-5)
    assert gamma > 0.0  # Home advantage should be positive


def test_fit_poisson_mle_with_rho():
    home_idx = np.array([0, 1, 0, 1])
    away_idx = np.array([1, 0, 1, 0])
    home_goals = np.array([0, 1, 0, 2])
    away_goals = np.array([0, 0, 1, 1])
    weights = np.ones(4)

    att, defense, mu, gamma, rho, conv, nll = fit_poisson_mle(
        home_idx, away_idx, home_goals, away_goals, weights, n_teams=2,
        include_rho=True,
    )

    assert conv
    assert -0.3 <= rho <= 0.3
