"""Negative Binomial2 score model with mean/dispersion parameterization."""

import math

import numpy as np


def negative_binomial_pmf(k: np.ndarray, mean: float, dispersion: float) -> np.ndarray:
    """Return NB2 probabilities where Var(Y) = mean + mean**2 / dispersion."""
    if mean <= 0 or not math.isfinite(mean):
        raise ValueError("mean must be finite and positive")
    if dispersion <= 0 or not math.isfinite(dispersion):
        raise ValueError("dispersion must be finite and positive")
    values = np.asarray(k, dtype=int)
    log_p = (
        np.vectorize(math.lgamma)(values + dispersion)
        - math.lgamma(dispersion)
        - np.vectorize(math.lgamma)(values + 1)
        + dispersion * math.log(dispersion / (dispersion + mean))
        + values * math.log(mean / (dispersion + mean))
    )
    return np.exp(log_p)


def estimate_dispersion(
    goals: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    floor: float = 1.0,
    ceiling: float = 200.0,
) -> float:
    """Estimate NB2 dispersion via method of moments.

    For NB2: Var(Y) = mu + mu^2 / phi  =>  phi = mu^2 / (Var(Y) - mu).
    When data is equi- or under-dispersed relative to Poisson, phi is clamped
    to ``ceiling`` (effectively recovering Poisson).
    """
    g = np.asarray(goals, dtype=float)
    mu = np.asarray(expected, dtype=float)
    w = np.ones(len(g), dtype=float) if weights is None else np.asarray(weights, dtype=float)

    # Weighted residual variance
    residuals = g - mu
    weighted_mean_sq = np.average(residuals ** 2, weights=w)
    weighted_mean_mu = np.average(mu, weights=w)

    excess = weighted_mean_sq - weighted_mean_mu
    if excess <= 0:
        # Data is under-dispersed or equi-dispersed => large phi (Poisson-like)
        return ceiling

    phi = weighted_mean_mu ** 2 / excess
    return float(max(floor, min(phi, ceiling)))


def score_matrix(home_mean: float, away_mean: float,
                 home_dispersion: float, away_dispersion: float,
                 max_goals: int = 10) -> np.ndarray:
    """Return normalized P(home goals, away goals) on a finite score grid."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    goals = np.arange(max_goals + 1)
    home = negative_binomial_pmf(goals, home_mean, home_dispersion)
    away = negative_binomial_pmf(goals, away_mean, away_dispersion)
    matrix = np.outer(home, away)
    total = matrix.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("negative-binomial score matrix has invalid probability")
    return matrix / total


def sample_scores(matrix: np.ndarray, size: int,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample score pairs from a normalized score matrix."""
    if size < 1:
        raise ValueError("size must be positive")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score matrix must be square")
    if np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0, atol=1e-10):
        raise ValueError("score matrix must be non-negative and sum to one")
    generator = rng or np.random.default_rng()
    flat = generator.choice(matrix.size, size=size, p=matrix.ravel())
    return np.unravel_index(flat, matrix.shape)
