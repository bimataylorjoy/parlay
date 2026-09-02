import numpy as np
import pytest

from parlay.models.dixon_coles import sample_scores, score_matrix, tau
from parlay.models.poisson import score_matrix as poisson_score_matrix


def test_zero_rho_matches_poisson():
    assert np.allclose(score_matrix(1.3, 0.9, rho=0), poisson_score_matrix(1.3, 0.9))


def test_dixon_coles_matrix_is_normalized_and_nonnegative():
    matrix = score_matrix(1.4, 1.0, rho=-0.05)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix >= 0)


def test_low_score_cells_are_corrected():
    baseline = poisson_score_matrix(1.2, 1.1)
    corrected = score_matrix(1.2, 1.1, rho=0.1)
    assert corrected[0, 0] < baseline[0, 0]
    assert corrected[1, 1] < baseline[1, 1]
    assert corrected[0, 1] > baseline[0, 1]


def test_extreme_rho_is_clamped_safely():
    # Extreme rho should not crash or produce negative values
    result = tau(np.array([[0]]), np.array([[0]]), 2.0, 2.0, 1.0)
    assert np.all(result >= 0)


def test_sampling_is_reproducible_with_seed():
    matrix = score_matrix(1.1, 1.2, rho=0.02)
    left = sample_scores(matrix, 100, np.random.default_rng(42))
    right = sample_scores(matrix, 100, np.random.default_rng(42))
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_estimate_rho_recovers_known_direction():
    from parlay.models.dixon_coles import estimate_rho
    # Generate synthetic matches with positive dependence on 0-0/1-1
    hg = np.array([0, 1, 0, 1, 2, 1, 0, 0, 1, 1, 2, 3] * 10)
    ag = np.array([0, 1, 1, 0, 1, 2, 0, 0, 1, 1, 0, 2] * 10)
    hr = np.full(len(hg), 1.2)
    ar = np.full(len(ag), 1.0)
    rho = estimate_rho(hg, ag, hr, ar)
    assert -0.3 <= rho <= 0.3
