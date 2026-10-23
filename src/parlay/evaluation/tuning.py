"""Walk-forward hyperparameter search with an untouched holdout (nested tuning)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

from parlay.data.schemas import Match, OddsSnapshot
from parlay.evaluation.backtest import BacktestConfig, run_backtest
from parlay.models.team_strength import DEFAULT_L2_REG

DEFAULT_HALF_LIVES: tuple[float | None, ...] = (90.0, 180.0, 270.0, 365.0, 540.0, 730.0, None)
DEFAULT_L2_GRID: tuple[float, ...] = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0)


def grid_search(
    matches: list[Match],
    config: BacktestConfig | None = None,
    *,
    half_lives: tuple[float | None, ...] = DEFAULT_HALF_LIVES,
    l2_grid: tuple[float, ...] = (DEFAULT_L2_REG,),
    odds: list[OddsSnapshot] | None = None,
) -> list[dict[str, Any]]:
    """Backtest every (half-life, L2) combination; rows sorted by log loss."""
    base = config or BacktestConfig()
    rows: list[dict[str, Any]] = []
    for half_life, l2 in itertools.product(half_lives, l2_grid):
        result = run_backtest(
            matches, odds, replace(base, half_life_days=half_life, l2_reg=l2, model_version=None)
        )
        if not result.records:
            continue
        rows.append(
            {
                "half_life_days": half_life,
                "l2_reg": l2,
                "n": result.metrics["n"],
                "log_loss": result.metrics["log_loss"],
                "brier_score": result.metrics["brier_score"],
                "rps": result.metrics["rps"],
                "folds": result.metrics["folds"],
            }
        )
    rows.sort(key=lambda row: float(row["log_loss"] or 0.0))
    return rows


def tune_half_life(
    matches: list[Match], config: BacktestConfig | None = None, **kwargs: Any
) -> list[dict[str, Any]]:
    """Grid search over half-life only (L2 held at the configured value)."""
    return grid_search(matches, config, **kwargs)


def nested_tuning(
    matches: list[Match],
    config: BacktestConfig | None = None,
    *,
    half_lives: tuple[float | None, ...] = DEFAULT_HALF_LIVES,
    l2_grid: tuple[float, ...] = DEFAULT_L2_GRID,
    development_end: date | None = None,
    holdout_days: int = 365,
    odds: list[OddsSnapshot] | None = None,
) -> dict[str, Any]:
    """Select hyperparameters on the development period only, then score them once on the holdout.

    The holdout score is the honest estimate of out-of-sample performance;
    the inner grid scores are optimistic by construction.
    """
    base = config or BacktestConfig()
    completed = sorted((m for m in matches if m.home_goals is not None), key=lambda m: m.date)
    if not completed:
        raise ValueError("no completed matches to tune on")
    dev_end = development_end or completed[-1].date - timedelta(days=holdout_days)
    development = [m for m in completed if m.date <= dev_end]
    holdout = [m for m in completed if m.date > dev_end]
    if not development or not holdout:
        raise ValueError("development/holdout split is empty; check the dates")
    inner = grid_search(development, base, half_lives=half_lives, l2_grid=l2_grid, odds=odds)
    if not inner:
        raise ValueError("no hyperparameter combination produced predictions on the development period")
    best = inner[0]
    locked = replace(
        base,
        half_life_days=best["half_life_days"],
        l2_reg=best["l2_reg"],
        initial_train_days=max(base.initial_train_days, (dev_end - completed[0].date).days + 1),
        evaluation_start=dev_end + timedelta(days=1),
        model_version=None,
    )
    holdout_result = run_backtest(completed, odds, locked)
    if not holdout_result.records:
        raise ValueError("holdout evaluation produced no predictions")
    return {
        "tuned_on": f"development <= {dev_end.isoformat()} ({len(development)} matches)",
        "evaluated_on": f"holdout > {dev_end.isoformat()} ({len(holdout)} matches)",
        "hyperparameters": {"half_life_days": best["half_life_days"], "l2_reg": best["l2_reg"]},
        "selection_metric": "log_loss",
        "holdout_metrics": holdout_result.metrics,
        "inner_results": inner,
    }
