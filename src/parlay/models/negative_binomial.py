"""Negative Binomial (NB2) score model with mean/dispersion parameterization."""

from __future__ import annotations

import numpy as np

from .numerics import (
    DEFAULT_TRUNCATION_EPSILON,
    grid_size,
    negative_binomial_pmf,
    normalized,
)

__all__ = ["estimate_dispersion", "negative_binomial_pmf", "score_matrix"]


def estimate_dispersion(
    goals: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    floor: float = 1.0,
    ceiling: float = 200.0,
) -> float:
    """Method-of-moments NB2 dispersion: ``phi = mu**2 / (Var(Y) - mu)``.

    Equi- or under-dispersed data returns ``ceiling`` (effectively Poisson).
    """
    g = np.asarray(goals, dtype=float)
    mu = np.asarray(expected, dtype=float)
    w = np.ones(len(g), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    weighted_mean_sq = np.average((g - mu) ** 2, weights=w)
    weighted_mean_mu = np.average(mu, weights=w)
    excess = weighted_mean_sq - weighted_mean_mu
    if excess <= 0:
        return ceiling
    return float(max(floor, min(weighted_mean_mu**2 / excess, ceiling)))


def score_matrix(
    home_mean: float,
    away_mean: float,
    home_dispersion: float,
    away_dispersion: float,
    max_goals: int | None = None,
    *,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
) -> np.ndarray:
    """Return normalized ``P(home=i, away=j)`` on a finite grid."""
    size = grid_size(
        home_mean,
        away_mean,
        max_goals,
        epsilon=epsilon,
        home_dispersion=home_dispersion,
        away_dispersion=away_dispersion,
    )
    goals = np.arange(size + 1)
    home = negative_binomial_pmf(goals, home_mean, home_dispersion)
    away = negative_binomial_pmf(goals, away_mean, away_dispersion)
    return normalized(np.outer(home, away), "negative-binomial score matrix")
