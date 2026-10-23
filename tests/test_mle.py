import numpy as np
import pytest
from scipy.optimize import check_grad

import parlay.models.mle as mle
from parlay.models.mle import fit_rate_model


def _random_design(seed=0, n_teams=6, n_matches=300):
    rng = np.random.default_rng(seed)
    home = rng.integers(0, n_teams, n_matches)
    away = (home + rng.integers(1, n_teams, n_matches)) % n_teams
    return (
        home,
        away,
        rng.poisson(1.5, n_matches).astype(float),
        rng.poisson(1.1, n_matches).astype(float),
        rng.uniform(0.2, 1.0, n_matches),
        (rng.uniform(size=n_matches) > 0.1).astype(float),
    )


@pytest.mark.parametrize(
    "family,dixon_coles", [("poisson", False), ("poisson", True), ("negative_binomial", False)]
)
def test_analytic_gradient_matches_finite_differences(monkeypatch, family, dixon_coles):
    captured = {}
    original = mle.minimize

    def spy(fun, x0, **kwargs):
        captured["fun"], captured["x0"] = fun, x0
        return original(fun, x0, **kwargs)

    monkeypatch.setattr(mle, "minimize", spy)
    home, away, yh, ya, w, venue = _random_design()
    fit_rate_model(home, away, yh, ya, w, 6, family=family, dixon_coles=dixon_coles, l2_reg=2.0, venue=venue)
    fun = captured["fun"]
    rng = np.random.default_rng(1)
    for _ in range(3):
        point = captured["x0"] + rng.normal(0, 0.1, len(captured["x0"]))
        if dixon_coles:
            point[2 * 5 + 2] = np.clip(point[2 * 5 + 2], -0.2, 0.2)
        error = check_grad(lambda t: fun(t)[0], lambda t: fun(t)[1], point, epsilon=1e-6)
        assert error / max(1.0, np.linalg.norm(fun(point)[1])) < 1e-4


def test_recovers_known_hierarchy():
    home = np.array([0, 0, 1, 1, 2, 2])
    away = np.array([1, 2, 0, 2, 0, 1])
    fit = fit_rate_model(
        home, away, np.array([3, 4, 1, 2, 0, 1]), np.array([0, 0, 3, 1, 4, 2]), np.ones(6), 3
    )
    assert fit.converged
    assert fit.attack[0] > fit.attack[1] > fit.attack[2]
    assert fit.defense[0] > fit.defense[1] > fit.defense[2]
    assert fit.attack.sum() == pytest.approx(0.0, abs=1e-9)
    assert fit.defense.sum() == pytest.approx(0.0, abs=1e-9)


def test_recovers_simulated_parameters():
    rng = np.random.default_rng(3)
    n_teams, n = 8, 4000
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defense = rng.normal(0, 0.3, n_teams)
    defense -= defense.mean()
    home = rng.integers(0, n_teams, n)
    away = (home + rng.integers(1, n_teams, n)) % n_teams
    yh = rng.poisson(np.exp(0.2 + 0.3 + attack[home] - defense[away]))
    ya = rng.poisson(np.exp(0.2 + attack[away] - defense[home]))
    fit = fit_rate_model(home, away, yh, ya, np.ones(n), n_teams)
    assert fit.converged
    assert fit.intercept == pytest.approx(0.2, abs=0.05)
    assert fit.home_advantage == pytest.approx(0.3, abs=0.05)
    assert np.max(np.abs(fit.attack - attack)) < 0.12
    assert np.max(np.abs(fit.defense - defense)) < 0.12


def test_dixon_coles_rho_estimated_in_domain_and_fixed_rho_respected():
    home, away, yh, ya, w, _ = _random_design(seed=4)
    fit = fit_rate_model(home, away, yh, ya, w, 6, dixon_coles=True)
    assert fit.converged
    assert -0.3 <= fit.rho <= 0.3
    fixed = fit_rate_model(home, away, yh, ya, w, 6, dixon_coles=True, fixed_rho=-0.05)
    assert fixed.rho == -0.05


def test_negative_binomial_detects_overdispersion():
    rng = np.random.default_rng(0)
    n = 600
    home = np.zeros(n, dtype=int)
    away = np.ones(n, dtype=int)
    yh = rng.negative_binomial(4, 4 / (4 + 1.6), n)
    ya = rng.negative_binomial(4, 4 / (4 + 1.1), n)
    fit = fit_rate_model(home, away, yh, ya, np.ones(n), 2, family="negative_binomial")
    assert fit.converged
    assert 1.5 < fit.dispersion < 15


def test_l2_prior_shrinks_team_parameters():
    home, away, yh, ya, w, _ = _random_design(seed=5, n_matches=120)
    loose = fit_rate_model(home, away, yh, ya, w, 6, l2_reg=0.0)
    tight = fit_rate_model(home, away, yh, ya, w, 6, l2_reg=50.0)
    assert np.abs(tight.attack).sum() < np.abs(loose.attack).sum()


def test_neutral_venue_removes_home_advantage_from_likelihood():
    rng = np.random.default_rng(6)
    n = 2000
    home = rng.integers(0, 4, n)
    away = (home + rng.integers(1, 4, n)) % 4
    yh = rng.poisson(1.3, n)  # no home effect in the data
    ya = rng.poisson(1.3, n)
    fit = fit_rate_model(home, away, yh, ya, np.ones(n), 4, venue=np.zeros(n))
    assert fit.home_advantage == pytest.approx(0.2, abs=1e-6)  # untouched initial value: no gradient


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"family": "negative_binomial", "dixon_coles": True}, "Dixon-Coles"),
        ({"family": "skellam"}, "family"),
        ({"l2_reg": -1.0}, "l2_reg"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_rate_model(
            np.array([0]), np.array([1]), np.array([1.0]), np.array([0.0]), np.ones(1), 2, **kwargs
        )
