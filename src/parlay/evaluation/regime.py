"""Regime analysis infrastructure (§18)."""

from collections import defaultdict
from typing import Callable
from parlay.evaluation.metrics import aggregate_scores, log_loss, brier_score


def analyze_by_regime(
    records,
    regime_fn: Callable[[object], str],
    *,
    min_n: int = 30,
) -> dict[str, dict]:
    """Group records by regime and report per-regime metrics with sample threshold.

    regime_fn: maps record -> regime label (e.g., "early_season", "promoted")
    """
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        label = regime_fn(r)
        groups[label].append(r)
    out = {}
    for label, rows in groups.items():
        if len(rows) < min_n:
            out[label] = {"n": len(rows), "note": f"insufficient_sample (<{min_n}), not reporting"}
            continue
        agg = aggregate_scores({"log_loss": log_loss({"home_win": x.home_win, "draw": x.draw, "away_win": x.away_win}, x.actual), "brier_score": brier_score({"home_win": x.home_win, "draw": x.draw, "away_win": x.away_win}, x.actual)} for x in rows)
        out[label] = {"n": len(rows), "log_loss": agg["log_loss"], "brier_score": agg["brier_score"]}
    return out


# Common regime mappers

def league_regime(record) -> str:
    return getattr(record, "competition", "unknown") or "unknown"


def early_season_regime(record, early_days: int = 30) -> str:
    # Requires record to have match date; fallback
    return "early" if "early" in str(getattr(record, "forecast_timestamp", "")) else "regular"


def promoted_regime(record, promoted_teams: set[str] | None = None) -> str:
    promoted = promoted_teams or {"Coventry", "Hull", "Wrexham", "Charlton"}
    if getattr(record, "home_team", "") in promoted or getattr(record, "away_team", "") in promoted:
        return "promoted_involved"
    return "established"
