"""Proper scoring rules for multiclass football outcomes."""

import math
from typing import Iterable


def _check(probabilities: dict[str, float]) -> None:
    if set(probabilities) != {"home_win", "draw", "away_win"}:
        raise ValueError("probabilities must contain home_win, draw, and away_win")
    if any(value < 0 for value in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities must be non-negative and sum to one")


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def log_loss(probabilities: dict[str, float], actual: str) -> float:
    _check(probabilities)
    if actual not in probabilities:
        raise ValueError(f"unknown outcome: {actual}")
    return -math.log(max(probabilities[actual], 1e-15))


def brier_score(probabilities: dict[str, float], actual: str) -> float:
    _check(probabilities)
    if actual not in probabilities:
        raise ValueError(f"unknown outcome: {actual}")
    return sum((probabilities[key] - float(key == actual)) ** 2 for key in probabilities)


def aggregate_scores(records: Iterable[dict[str, object]]) -> dict[str, float]:
    rows = list(records)
    if not rows:
        raise ValueError("at least one prediction record is required")
    return {
        "n": float(len(rows)),
        "log_loss": sum(float(row["log_loss"]) for row in rows) / len(rows),
        "brier_score": sum(float(row["brier_score"]) for row in rows) / len(rows),
    }
