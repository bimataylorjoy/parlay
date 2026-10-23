"""Independent-Poisson score model."""

from __future__ import annotations

import numpy as np

from .numerics import (
    DEFAULT_TRUNCATION_EPSILON,
    grid_size,
    normalized,
    poisson_pmf,
    poisson_tail,
)

__all__ = ["poisson_pmf", "score_matrix", "score_matrix_with_metadata"]


def score_matrix(
    home_rate: float,
    away_rate: float,
    max_goals: int | None = None,
    *,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
) -> np.ndarray:
    """Return normalized ``P(home=i, away=j)``.

    ``max_goals=None`` picks the smallest grid whose per-team tail is below
    ``epsilon`` (never smaller than 10x10). The truncated grid is renormalized.
    """
    size = grid_size(home_rate, away_rate, max_goals, epsilon=epsilon)
    goals = np.arange(size + 1)
    return normalized(np.outer(poisson_pmf(goals, home_rate), poisson_pmf(goals, away_rate)))


def score_matrix_with_metadata(
    home_rate: float,
    away_rate: float,
    max_goals: int | None = None,
    *,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return ``(matrix, metadata)`` with the tail mass dropped by truncation."""
    size = grid_size(home_rate, away_rate, max_goals, epsilon=epsilon)
    tail_home = poisson_tail(home_rate, size)
    tail_away = poisson_tail(away_rate, size)
    matrix = score_matrix(home_rate, away_rate, max_goals=size)
    return matrix, {
        "max_goals": float(size),
        "tail_mass_home": tail_home,
        "tail_mass_away": tail_away,
        "joint_tail_mass_estimate": 1.0 - (1.0 - tail_home) * (1.0 - tail_away),
        "truncation_epsilon": float(epsilon),
    }
