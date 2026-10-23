"""Out-of-sample scores broken down by regime (league, team history, ...)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from parlay.evaluation.metrics import brier_score, log_loss, ranked_probability_score


def analyze_by_regime(
    records: Iterable[Any], regime_fn: Callable[[Any], str], *, min_n: int = 30
) -> dict[str, dict[str, float | str]]:
    """Per-regime 1X2 scores; regimes with fewer than ``min_n`` records are not scored."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        groups[regime_fn(record)].append(record)
    output: dict[str, dict[str, float | str]] = {}
    for label, rows in sorted(groups.items()):
        if len(rows) < min_n:
            output[label] = {"n": float(len(rows)), "note": f"insufficient_sample (<{min_n})"}
            continue
        probabilities = [
            ({"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual) for r in rows
        ]
        output[label] = {
            "n": float(len(rows)),
            "log_loss": sum(log_loss(p, a) for p, a in probabilities) / len(rows),
            "brier_score": sum(brier_score(p, a) for p, a in probabilities) / len(rows),
            "rps": sum(ranked_probability_score(p, a) for p, a in probabilities) / len(rows),
        }
    return output


def competition_regime(record: Any) -> str:
    return getattr(record, "competition", None) or "unknown"


def team_history_regime(record: Any, thresholds: tuple[int, int] = (10, 38)) -> str:
    """Bucket by the smaller number of training matches of the two teams."""
    count = getattr(record, "min_team_matches", None)
    if count is None:
        return "unknown"
    low, high = thresholds
    if count < low:
        return f"new_team (<{low} matches)"
    if count < high:
        return f"limited_history ({low}-{high - 1})"
    return f"established (>={high})"
