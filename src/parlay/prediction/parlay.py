"""Joint probability and risk calculations for correlated football legs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from parlay.prediction.markets import fair_odds


@dataclass(frozen=True, slots=True)
class ScoreLeg:
    """A leg represented as a predicate over (home_goals, away_goals)."""
    name: str
    predicate: object

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        home, away = np.indices(shape)
        return np.asarray(self.predicate(home, away), dtype=bool)


@dataclass(frozen=True, slots=True)
class ParlayResult:
    leg_probabilities: dict[str, float]
    joint_probability: float
    independence_probability: float
    dependence_adjustment: float
    odds: float
    expected_value: float
    covariance_matrix: tuple[tuple[float, ...], ...]
    diagnostics: tuple[str, ...] = ()


def joint_score_probability(matrix: np.ndarray, legs: list[ScoreLeg]) -> float:
    """Return P(all legs) under the supplied joint score distribution."""
    if not legs:
        raise ValueError("at least one leg is required")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0):
        raise ValueError("matrix must be a normalized non-negative square matrix")
    mask = np.ones(matrix.shape, dtype=bool)
    for leg in legs:
        mask &= leg.mask(matrix.shape)
    return float(matrix[mask].sum())


def parlay_from_score_matrix(
    matrix: np.ndarray, legs: list[ScoreLeg], decimal_odds: float,
) -> ParlayResult:
    """Calculate joint EV and dependence diagnostics from one score matrix."""
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1")
    matrix = np.asarray(matrix, dtype=float)
    masks = [leg.mask(matrix.shape) for leg in legs]
    probabilities = {leg.name: float(matrix[mask].sum()) for leg, mask in zip(legs, masks)}
    joint = float(matrix[np.logical_and.reduce(masks)].sum())
    independent = math.prod(probabilities.values())
    indicators = np.stack([mask.ravel().astype(float) for mask in masks])
    covariance = np.cov(indicators, aweights=matrix.ravel(), bias=True) if len(legs) > 1 else np.array([[0.0]])
    covariance = np.atleast_2d(covariance)
    diagnostics = []
    if len(legs) > 1 and not math.isclose(joint, independent, rel_tol=1e-6, abs_tol=1e-9):
        diagnostics.append("leg_dependence_present")
    if joint <= 0:
        diagnostics.append("joint_probability_zero")
    return ParlayResult(
        leg_probabilities=probabilities,
        joint_probability=joint,
        independence_probability=independent,
        dependence_adjustment=joint - independent,
        odds=float(decimal_odds),
        expected_value=joint * decimal_odds - 1.0,
        covariance_matrix=tuple(tuple(float(value) for value in row) for row in covariance),
        diagnostics=tuple(diagnostics),
    )


def score_leg(name: str, predicate) -> ScoreLeg:
    return ScoreLeg(name=name, predicate=predicate)


def btts_leg(name: str = "btts_yes") -> ScoreLeg:
    return score_leg(name, lambda home, away: (home > 0) & (away > 0))


def over_leg(line: float, name: str | None = None) -> ScoreLeg:
    return score_leg(name or f"over_{line:g}", lambda home, away: (home + away) > line)


def home_win_leg(name: str = "home_win") -> ScoreLeg:
    return score_leg(name, lambda home, away: home > away)


def correct_score_leg(home_goals: int, away_goals: int, name: str | None = None) -> ScoreLeg:
    return score_leg(name or f"correct_score_{home_goals}_{away_goals}", lambda home, away: (home == home_goals) & (away == away_goals))
