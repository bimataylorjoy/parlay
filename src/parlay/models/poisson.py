"""Independent-Poisson score model and posterior-draw-friendly outputs."""

import math

import numpy as np


def poisson_pmf(k: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and positive")
    values = np.asarray(k, dtype=int)
    log_p = values * math.log(rate) - rate - np.vectorize(math.lgamma)(values + 1)
    return np.exp(log_p)


def score_matrix(home_rate: float, away_rate: float, max_goals: int = 10) -> np.ndarray:
    """Return P(home goals, away goals), with the finite-grid tail normalized."""
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson_pmf(goals, home_rate), poisson_pmf(goals, away_rate))
    total = matrix.sum()
    if total <= 0:
        raise ValueError("score matrix has zero probability")
    return matrix / total


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
