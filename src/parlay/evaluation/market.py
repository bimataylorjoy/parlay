"""Out-of-sample market scoring and transparent staking-strategy evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from parlay.evaluation.metrics import PROBABILITY_FLOOR, brier_score, log_loss

SELECTIONS = ("home", "draw", "away")
_OUTCOME = {"home": "home_win", "draw": "draw", "away": "away_win"}


def _multiclass_scores(probabilities: dict[str, float], actual: str) -> tuple[float, float]:
    if not probabilities or actual not in probabilities:
        raise ValueError("actual must be one of the supplied market outcomes")
    if any(value < 0 for value in probabilities.values()) or abs(sum(probabilities.values()) - 1.0) > 1e-8:
        raise ValueError("market probabilities must be non-negative and sum to one")
    ll = -math.log(max(probabilities[actual], PROBABILITY_FLOOR))
    brier = sum((value - float(key == actual)) ** 2 for key, value in probabilities.items())
    return ll, brier


def evaluate_market_predictions(
    records: Iterable[Any], *, probabilities_attr: str, actual_attr: str = "actual_market"
) -> dict[str, float]:
    """Score any mutually exclusive market given a probability mapping and realized selection."""
    scored = [
        _multiclass_scores(getattr(record, probabilities_attr), getattr(record, actual_attr))
        for record in records
        if getattr(record, probabilities_attr, None) is not None
        and getattr(record, actual_attr, None) is not None
    ]
    if not scored:
        return {"n": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    return {
        "n": float(len(scored)),
        "log_loss": sum(item[0] for item in scored) / len(scored),
        "brier_score": sum(item[1] for item in scored) / len(scored),
    }


def evaluate_binary_market(
    records: Iterable[Any], *, probability_attr: str, actual_attr: str
) -> dict[str, float]:
    """Score a binary market (over/under, yes/no); explicit pushes are excluded.

    When a record has ``push_probability``, the probability is renormalized
    to the decisive outcomes before scoring.
    """
    rows = []
    pushes = 0
    for record in records:
        probability = getattr(record, probability_attr, None)
        actual = getattr(record, actual_attr, None)
        if probability is None or actual is None:
            continue
        if actual == "push":
            pushes += 1
            continue
        if actual not in {"over", "under", "yes", "no"}:
            raise ValueError(f"unsupported binary market outcome: {actual}")
        p = float(probability)
        push_probability = float(getattr(record, "push_probability", 0.0) or 0.0)
        if push_probability:
            decisive = 1.0 - push_probability
            if decisive <= 0:
                pushes += 1
                continue
            p = (p - 0.5 * push_probability) / decisive
        positive = actual in {"over", "yes"}
        p_actual = p if positive else 1.0 - p
        rows.append((-math.log(max(p_actual, PROBABILITY_FLOOR)), (p - float(positive)) ** 2))
    if not rows:
        return {"n": 0.0, "pushes": float(pushes), "log_loss": 0.0, "brier_score": 0.0}
    return {
        "n": float(len(rows)),
        "pushes": float(pushes),
        "log_loss": sum(row[0] for row in rows) / len(rows),
        "brier_score": sum(row[1] for row in rows) / len(rows),
    }


def _reliability(values: list[tuple[float, float]], bins: int) -> list[dict[str, float]]:
    output = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [v for v in values if lower <= v[0] < upper or (index == bins - 1 and v[0] == upper)]
        if selected:
            output.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": float(len(selected)),
                    "mean_probability": sum(v[0] for v in selected) / len(selected),
                    "empirical_frequency": sum(v[1] for v in selected) / len(selected),
                }
            )
    return output


def market_calibration_bins(
    records: Iterable[Any], *, probability_attr: str, actual_attr: str, bins: int = 10
) -> list[dict[str, float]]:
    """Reliability bins for binary over/yes probabilities."""
    if bins < 1:
        raise ValueError("bins must be positive")
    values = [
        (float(getattr(r, probability_attr)), float(getattr(r, actual_attr) in {"over", "yes"}))
        for r in records
        if getattr(r, probability_attr, None) is not None
        and getattr(r, actual_attr, None) in {"over", "under", "yes", "no"}
    ]
    return _reliability(values, bins)


def calibration_bins(records: Iterable[Any], bins: int = 10) -> list[dict[str, float]]:
    """Reliability bins for each 1X2 outcome (mean forecast vs empirical frequency)."""
    if bins < 1:
        raise ValueError("bins must be positive")
    rows = list(records)
    output: list[dict[str, Any]] = []
    for selection in SELECTIONS:
        outcome = _OUTCOME[selection]
        values = [(float(getattr(r, outcome)), float(r.actual == outcome)) for r in rows]
        output.extend({"selection": selection, **row} for row in _reliability(values, bins))
    return output


def evaluate_market_benchmark(records: Iterable[Any]) -> dict[str, float]:
    """Log loss / Brier of de-vigged market probabilities on the records that have them."""
    pairs = [
        (
            {
                "home_win": float(r.market_home),
                "draw": float(r.market_draw),
                "away_win": float(r.market_away),
            },
            r.actual,
        )
        for r in records
        if getattr(r, "market_home", None) is not None
        and getattr(r, "market_draw", None) is not None
        and getattr(r, "market_away", None) is not None
    ]
    if not pairs:
        return {"n": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    n = float(len(pairs))
    return {
        "n": n,
        "log_loss": sum(log_loss(p, a) for p, a in pairs) / n,
        "brier_score": sum(brier_score(p, a) for p, a in pairs) / n,
    }


def evaluate_flat_stake(
    records: Iterable[Any], *, min_edge: float = 0.03, min_ev: float = 0.02, stake: float = 1.0
) -> dict[str, float]:
    """One-selection-per-match flat-stake strategy on 1X2 prices.

    A selection qualifies when both its probability edge over the de-vigged
    market and its expected value clear the thresholds; the highest-EV
    qualifying selection is bet. When closing probabilities are available the
    result also reports closing-line value (CLV): the taken price relative to
    the fair closing price, ``odds * p_closing - 1``. Positive mean CLV is the
    most reliable evidence of genuine edge.
    """
    if min_edge < 0 or min_ev < 0 or stake <= 0:
        raise ValueError("thresholds must be non-negative and stake must be positive")
    profits: list[float] = []
    clv: list[float] = []
    for record in records:
        candidates = []
        for selection in SELECTIONS:
            edge = getattr(record, f"edge_{selection}", None)
            ev = getattr(record, f"ev_{selection}", None)
            odds = getattr(record, f"odds_{selection}", None)
            if edge is None or ev is None or odds is None or edge < min_edge or ev < min_ev:
                continue
            candidates.append((float(ev), selection, float(odds)))
        if not candidates:
            continue
        _, selection, odds = max(candidates)
        won = record.actual == _OUTCOME[selection]
        profits.append(stake * (odds - 1.0) if won else -stake)
        closing = getattr(record, f"closing_{selection}", None)
        if closing is not None:
            clv.append(odds * float(closing) - 1.0)
    if not profits:
        return {
            "bets": 0.0,
            "turnover": 0.0,
            "profit": 0.0,
            "yield": 0.0,
            "yield_se": 0.0,
            "max_drawdown": 0.0,
        }
    cumulative = peak = max_drawdown = 0.0
    for profit in profits:
        cumulative += profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    turnover = stake * len(profits)
    mean = cumulative / len(profits)
    variance = sum((p - mean) ** 2 for p in profits) / max(1, len(profits) - 1)
    result = {
        "bets": float(len(profits)),
        "turnover": turnover,
        "profit": cumulative,
        "yield": cumulative / turnover,
        "yield_se": math.sqrt(variance / len(profits)) / stake,
        "max_drawdown": max_drawdown,
    }
    if clv:
        result["clv_n"] = float(len(clv))
        result["mean_clv"] = sum(clv) / len(clv)
    return result
