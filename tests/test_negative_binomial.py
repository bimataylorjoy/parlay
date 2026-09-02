import numpy as np
import pytest

from parlay.models.negative_binomial import (
    negative_binomial_pmf,
    sample_scores,
    score_matrix,
)
from parlay.models.poisson import score_matrix as poisson_score_matrix


def test_negative_binomial_matrix_is_normalized():
    matrix = score_matrix(1.4, 1.0, 8.0, 8.0)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.all(matrix >= 0)


def test_large_dispersion_approaches_poisson():
    nb = score_matrix(1.4, 1.0, 1_000_000.0, 1_000_000.0)
    poisson = poisson_score_matrix(1.4, 1.0)
    assert np.allclose(nb, poisson, atol=1e-6)


def test_small_dispersion_has_heavier_zero_and_tail_mass():
    concentrated = negative_binomial_pmf(np.arange(21), 1.5, 100.0)
    dispersed = negative_binomial_pmf(np.arange(21), 1.5, 1.0)
    assert dispersed[0] > concentrated[0]
    assert dispersed[-1] > concentrated[-1]


def test_sampler_is_reproducible():
    matrix = score_matrix(1.2, 0.8, 5.0, 7.0)
    left = sample_scores(matrix, 100, np.random.default_rng(12))
    right = sample_scores(matrix, 100, np.random.default_rng(12))
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


@pytest.mark.parametrize("mean,dispersion", [(0, 1), (1, 0), (1, -1)])
def test_invalid_parameters_are_rejected(mean, dispersion):
    with pytest.raises(ValueError):
        negative_binomial_pmf(np.array([0]), mean, dispersion)


def test_estimate_dispersion_bounds():
    from parlay.models.negative_binomial import estimate_dispersion
    goals = np.array([0, 1, 2, 0, 3, 1, 4, 0, 2, 1] * 5)
    expected = np.full(len(goals), 1.5)
    disp = estimate_dispersion(goals, expected)
    assert 1.0 <= disp <= 200.0


def test_estimate_dispersion_underdispersed_clamps_to_ceiling():
    from parlay.models.negative_binomial import estimate_dispersion
    # All values very close to mean -> variance < mean (underdispersed)
    goals = np.array([1, 1, 1, 1, 1, 2, 1, 1, 1, 2])
    expected = np.full(len(goals), 1.2)
    disp = estimate_dispersion(goals, expected, ceiling=150.0)
    assert disp == 150.0
