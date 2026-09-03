"""Independent-Poisson score model and posterior-draw-friendly outputs."""

import math

import numpy as np

from .numerics import (
    DEFAULT_SCORE_CAP,
    DEFAULT_TRUNCATION_EPSILON,
    adaptive_support,
    poisson_tail,
)


def poisson_pmf(k: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and positive")
    values = np.asarray(k, dtype=int)
    log_p = values * math.log(rate) - rate - np.vectorize(math.lgamma)(values + 1)
    return np.exp(log_p)


def _adaptive_max_goals(rate: float, epsilon: float = 1e-6, cap: int = DEFAULT_SCORE_CAP) -> int:
    """Smallest k with P(X>k) < epsilon for Poisson(rate). Cap for safety."""
    if rate <= 0 or not math.isfinite(rate):
        return 10
    return adaptive_support(rate, epsilon, cap)[0]


def score_matrix(
    home_rate: float, away_rate: float, max_goals: int | None = 10, *, epsilon: float = DEFAULT_TRUNCATION_EPSILON
) -> np.ndarray:
    """Return P(home goals, away goals), with adaptive truncation (ε=1e-6).

    If max_goals is None, adaptively choose smallest k where tail < epsilon for
    max(home_rate, away_rate). Otherwise use fixed k but still track tail mass.
    Finite-grid tail is renormalized; caller can inspect tail via score_matrix_with_metadata.
    """
    if max_goals is not None and max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    if max_goals is None:
        # Adaptive: use max of both rates
        k_home = _adaptive_max_goals(home_rate, epsilon)
        k_away = _adaptive_max_goals(away_rate, epsilon)
        max_goals = min(max(k_home, k_away), DEFAULT_SCORE_CAP)
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson_pmf(goals, home_rate), poisson_pmf(goals, away_rate))
    total = matrix.sum()
    if total <= 0:
        raise ValueError("score matrix has zero probability")
    return matrix / total


def score_matrix_with_metadata(
    home_rate: float, away_rate: float, max_goals: int | None = 10, *, epsilon: float = DEFAULT_TRUNCATION_EPSILON
) -> tuple[np.ndarray, dict[str, float]]:
    """Return (matrix, metadata) with tail_mass diagnostics."""
    if max_goals is None:
        k_home = _adaptive_max_goals(home_rate, epsilon)
        k_away = _adaptive_max_goals(away_rate, epsilon)
        max_goals = min(max(k_home, k_away), DEFAULT_SCORE_CAP)
    # Compute tail mass before truncation
    # Poisson tail P(X > max_goals)
    def tail(rate: float, k: int) -> float:
        return poisson_tail(rate, k)

    tail_home = tail(home_rate, max_goals) if max_goals is not None else 0.0
    tail_away = tail(away_rate, max_goals) if max_goals is not None else 0.0
    # Joint tail approx (independent)
    joint_tail = 1.0 - (1.0 - tail_home) * (1.0 - tail_away)
    M = score_matrix(home_rate, away_rate, max_goals=max_goals, epsilon=epsilon)
    meta = {
        "max_goals": float(max_goals) if max_goals is not None else 0.0,
        "tail_mass_home": float(tail_home),
        "tail_mass_away": float(tail_away),
        "joint_tail_mass_estimate": float(joint_tail),
        "truncation_epsilon": float(epsilon),
    }
    return M, meta


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    matrix = _validated_matrix(matrix)
    home = np.tril(matrix, -1).sum()
    away = np.triu(matrix, 1).sum()
    draw = np.trace(matrix)
    return {"home_win": float(home), "draw": float(draw), "away_win": float(away)}


def totals_probability(matrix: np.ndarray, line: float) -> dict[str, float]:
    matrix = _validated_matrix(matrix)
    if not math.isfinite(line):
        raise ValueError("line must be finite")
    home, away = np.indices(matrix.shape)
    over = float(matrix[(home + away) > line].sum())
    return {"over": over, "under": float(1.0 - over)}


def btts_probability(matrix: np.ndarray) -> dict[str, float]:
    matrix = _validated_matrix(matrix)
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": float(1.0 - yes)}


def _validated_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score matrix must be square")
    if np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0, atol=1e-10):
        raise ValueError("score matrix must be non-negative and sum to one")
    return matrix
