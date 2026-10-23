import numpy as np
import pytest

from parlay.models.dixon_coles import estimate_rho, global_rho_bounds, tau, valid_rho_bounds
from parlay.models.dixon_coles import score_matrix as dc_matrix
from parlay.models.negative_binomial import estimate_dispersion
from parlay.models.negative_binomial import score_matrix as nb_matrix
from parlay.models.numerics import (
    MIN_GRID,
    adaptive_support,
    negative_binomial_pmf,
    poisson_pmf,
    poisson_tail,
    sample_scores,
)
from parlay.models.poisson import score_matrix, score_matrix_with_metadata


def test_poisson_pmf_matches_closed_form():
    k = np.arange(6)
    expected = np.array([np.exp(-1.5) * 1.5**i / np.prod(range(1, i + 1)) for i in k])
    assert np.allclose(poisson_pmf(k, 1.5), expected)


@pytest.mark.parametrize("mean,dispersion", [(0, 1), (1, 0), (1, -1), (float("inf"), 1)])
def test_negative_binomial_rejects_invalid_parameters(mean, dispersion):
    with pytest.raises(ValueError):
        negative_binomial_pmf(np.array([0]), mean, dispersion)


def test_negative_binomial_variance_matches_nb2():
    k = np.arange(200)
    p = negative_binomial_pmf(k, 2.0, 4.0)
    mean = float((k * p).sum())
    var = float((k**2 * p).sum() - mean**2)
    assert mean == pytest.approx(2.0, rel=1e-6)
    assert var == pytest.approx(2.0 + 2.0**2 / 4.0, rel=1e-6)


def test_adaptive_support_is_smallest_valid_cutoff():
    cutoff, capped = adaptive_support(1.2, epsilon=1e-6, cap=100)
    assert not capped
    assert poisson_tail(1.2, cutoff) < 1e-6
    assert poisson_tail(1.2, cutoff - 1) >= 1e-6


def test_matrices_are_normalized_square_distributions():
    for matrix in (score_matrix(1.4, 1.0), dc_matrix(1.4, 1.0, -0.08), nb_matrix(1.4, 1.0, 8.0, 8.0)):
        assert matrix.shape[0] == matrix.shape[1] >= MIN_GRID + 1
        assert np.isclose(matrix.sum(), 1.0)
        assert np.all(matrix >= 0)


def test_adaptive_grid_grows_for_high_rates_and_reports_tail():
    matrix, meta = score_matrix_with_metadata(3.5, 3.5, max_goals=None, epsilon=1e-6)
    assert meta["joint_tail_mass_estimate"] < 1e-5
    assert matrix.shape[0] >= 14
    _, fixed = score_matrix_with_metadata(3.5, 3.5, max_goals=10)
    assert fixed["tail_mass_home"] > 1e-4


def test_zero_rho_dixon_coles_equals_poisson():
    assert np.allclose(dc_matrix(1.3, 0.9, rho=0.0), score_matrix(1.3, 0.9))


def test_positive_rho_moves_mass_from_draws_to_one_nil():
    baseline = score_matrix(1.2, 1.1)
    corrected = dc_matrix(1.2, 1.1, rho=0.1)
    assert corrected[0, 0] < baseline[0, 0]
    assert corrected[1, 1] < baseline[1, 1]
    assert corrected[0, 1] > baseline[0, 1]


def test_tau_is_clamped_non_negative():
    assert np.all(tau(np.array([[0]]), np.array([[0]]), 2.0, 2.0, 1.0) >= 0)


def test_global_rho_bounds_are_intersection_of_per_match_bounds():
    home = np.array([3.5, 0.8])
    away = np.array([3.5, 0.8])
    lower, upper = valid_rho_bounds(home, away)
    g_lower, g_upper = global_rho_bounds(home, away)
    assert g_lower >= np.max(lower) - 1e-12
    assert g_upper <= np.min(upper) + 1e-12


def test_estimate_rho_stays_inside_global_domain():
    home_rates = np.array([3.5, 3.0, 0.8])
    away_rates = np.array([3.5, 2.5, 0.8])
    lower, upper = global_rho_bounds(home_rates, away_rates)
    value = estimate_rho(np.array([0, 1, 0]), np.array([0, 0, 1]), home_rates, away_rates)
    assert lower <= value <= upper


def test_large_dispersion_nb_approaches_poisson():
    assert np.allclose(
        nb_matrix(1.4, 1.0, 1e6, 1e6, max_goals=10), score_matrix(1.4, 1.0, max_goals=10), atol=1e-6
    )


def test_small_dispersion_has_heavier_zero_and_tail():
    concentrated = negative_binomial_pmf(np.arange(21), 1.5, 100.0)
    dispersed = negative_binomial_pmf(np.arange(21), 1.5, 1.0)
    assert dispersed[0] > concentrated[0]
    assert dispersed[-1] > concentrated[-1]


def test_estimate_dispersion_bounds_and_underdispersion():
    goals = np.array([0, 1, 2, 0, 3, 1, 4, 0, 2, 1] * 5)
    assert 1.0 <= estimate_dispersion(goals, np.full(len(goals), 1.5)) <= 200.0
    flat = np.array([1, 1, 1, 1, 1, 2, 1, 1, 1, 2])
    assert estimate_dispersion(flat, np.full(len(flat), 1.2), ceiling=150.0) == 150.0


def test_sampling_is_reproducible_and_validated():
    matrix = dc_matrix(1.1, 1.2, rho=0.02)
    left = sample_scores(matrix, 100, np.random.default_rng(42))
    right = sample_scores(matrix, 100, np.random.default_rng(42))
    assert np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])
    with pytest.raises(ValueError):
        sample_scores(matrix * 2, 10)
    with pytest.raises(ValueError):
        sample_scores(matrix, 0)
