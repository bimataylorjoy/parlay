"""Shared numerical policy: rate bounds, count PMFs, finite score grids."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

LOG_RATE_MIN = -3.0
LOG_RATE_MAX = 3.0
DEFAULT_TRUNCATION_EPSILON = 1e-6
DEFAULT_SCORE_CAP = 100
MIN_GRID = 10


def bounded_log_rate(value: float) -> float:
    """Apply the model's shared finite log-rate domain."""
    return max(min(float(value), LOG_RATE_MAX), LOG_RATE_MIN)


def _check_rate(rate: float) -> None:
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and positive")


def poisson_pmf(k: np.ndarray, rate: float) -> np.ndarray:
    """Poisson probabilities for integer counts ``k``."""
    _check_rate(rate)
    values = np.asarray(k, dtype=float)
    return np.exp(values * math.log(rate) - rate - gammaln(values + 1.0))


def negative_binomial_pmf(k: np.ndarray, mean: float, dispersion: float) -> np.ndarray:
    """NB2 probabilities where ``Var(Y) = mean + mean**2 / dispersion``."""
    if mean <= 0 or not math.isfinite(mean):
        raise ValueError("mean must be finite and positive")
    if dispersion <= 0 or not math.isfinite(dispersion):
        raise ValueError("dispersion must be finite and positive")
    values = np.asarray(k, dtype=float)
    log_p = (
        gammaln(values + dispersion)
        - gammaln(dispersion)
        - gammaln(values + 1.0)
        + dispersion * math.log(dispersion / (dispersion + mean))
        + values * math.log(mean / (dispersion + mean))
    )
    return np.exp(log_p)


def poisson_tail(rate: float, cutoff: int) -> float:
    """Return ``P(X > cutoff)`` for ``X ~ Poisson(rate)``."""
    _check_rate(rate)
    if cutoff < 0:
        return 1.0
    cumulative = float(poisson_pmf(np.arange(cutoff + 1), rate).sum())
    return max(0.0, min(1.0, 1.0 - cumulative))


def adaptive_support(
    rate: float,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
    cap: int = DEFAULT_SCORE_CAP,
    dispersion: float | None = None,
) -> tuple[int, bool]:
    """Smallest cutoff whose upper tail mass is below ``epsilon``.

    Uses the Poisson tail, or the NB2 tail when ``dispersion`` is given. The
    boolean reports whether ``cap`` was reached before the tail was small enough.
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")
    _check_rate(rate)
    if cap < 0:
        raise ValueError("cap must be non-negative")
    grid = np.arange(cap + 1)
    pmf = poisson_pmf(grid, rate) if dispersion is None else negative_binomial_pmf(grid, rate, dispersion)
    tails = 1.0 - np.cumsum(pmf)
    below = np.nonzero(tails < epsilon)[0]
    if below.size:
        return int(below[0]), False
    return cap, True


def grid_size(
    home_rate: float,
    away_rate: float,
    max_goals: int | None,
    *,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
    home_dispersion: float | None = None,
    away_dispersion: float | None = None,
) -> int:
    """Resolve the score grid: fixed ``max_goals`` or adaptive (never below ``MIN_GRID``)."""
    if max_goals is not None:
        if max_goals < 1:
            raise ValueError("max_goals must be at least 1")
        return max_goals
    home_k, _ = adaptive_support(home_rate, epsilon, dispersion=home_dispersion)
    away_k, _ = adaptive_support(away_rate, epsilon, dispersion=away_dispersion)
    return max(home_k, away_k, MIN_GRID)


def normalized(matrix: np.ndarray, label: str = "score matrix") -> np.ndarray:
    total = matrix.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"{label} has invalid probability mass")
    return matrix / total


def validated_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` as float array after checking it is a square distribution."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score matrix must be square")
    if np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0, atol=1e-10):
        raise ValueError("score matrix must be non-negative and sum to one")
    return matrix


def sample_scores(
    matrix: np.ndarray, size: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (home, away) score pairs from a normalized score matrix."""
    if size < 1:
        raise ValueError("size must be positive")
    matrix = validated_matrix(matrix)
    generator = rng or np.random.default_rng()
    flat = generator.choice(matrix.size, size=size, p=matrix.ravel())
    home, away = np.unravel_index(flat, matrix.shape)
    return home, away
