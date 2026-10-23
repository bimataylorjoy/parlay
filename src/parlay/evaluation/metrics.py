"""Proper scoring rules for football outcomes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

OUTCOMES = ("home_win", "draw", "away_win")
PROBABILITY_FLOOR = 1e-15


def _check(probabilities: Mapping[str, float]) -> None:
    if set(probabilities) != set(OUTCOMES):
        raise ValueError("probabilities must contain home_win, draw, and away_win")
    if any(value < 0 for value in probabilities.values()) or not math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-8
    ):
        raise ValueError("probabilities must be non-negative and sum to one")


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def log_loss(probabilities: Mapping[str, float], actual: str) -> float:
    _check(probabilities)
    if actual not in probabilities:
        raise ValueError(f"unknown outcome: {actual}")
    return -math.log(max(probabilities[actual], PROBABILITY_FLOOR))


def brier_score(probabilities: Mapping[str, float], actual: str) -> float:
    """Multiclass Brier score (sum over the three outcomes, range 0-2)."""
    _check(probabilities)
    if actual not in probabilities:
        raise ValueError(f"unknown outcome: {actual}")
    return sum((probabilities[key] - float(key == actual)) ** 2 for key in probabilities)


def ranked_probability_score(probabilities: Mapping[str, float], actual: str) -> float:
    """RPS for the ordered outcomes home < draw < away (Constantinou & Fenton, 2012).

    Unlike log loss and Brier, RPS rewards putting mass *near* the realized
    outcome: predicting a draw when the home side wins is penalized less than
    predicting an away win.
    """
    _check(probabilities)
    if actual not in probabilities:
        raise ValueError(f"unknown outcome: {actual}")
    cumulative_p = 0.0
    cumulative_o = 0.0
    total = 0.0
    for outcome in OUTCOMES[:-1]:
        cumulative_p += probabilities[outcome]
        cumulative_o += float(outcome == actual)
        total += (cumulative_p - cumulative_o) ** 2
    return total / (len(OUTCOMES) - 1)


def binary_log_loss(probability: float, happened: bool) -> float:
    p = probability if happened else 1.0 - probability
    return -math.log(max(p, PROBABILITY_FLOOR))


def aggregate_scores(records: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Mean of every scoring field present in the records."""
    rows = list(records)
    if not rows:
        raise ValueError("at least one prediction record is required")
    result = {"n": float(len(rows))}
    for key in ("log_loss", "brier_score", "rps"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]  # type: ignore[arg-type]
        if values:
            result[key] = sum(values) / len(values)
    return result
