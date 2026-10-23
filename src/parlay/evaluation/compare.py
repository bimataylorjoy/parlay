"""Run several goal models on identical walk-forward folds."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

from parlay.data.schemas import Match, OddsSnapshot
from parlay.evaluation.backtest import BacktestConfig, BacktestResult, run_backtest
from parlay.evaluation.market import evaluate_market_benchmark
from parlay.models.team_strength import GOAL_MODELS


def _stability(values: list[float]) -> dict[str, float]:
    """Mean, (population) std, min and max of per-fold scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
        "min": min(values),
        "max": max(values),
    }


def compare_models(
    matches: list[Match],
    odds: list[OddsSnapshot] | None = None,
    config: BacktestConfig | None = None,
    *,
    models: tuple[str, ...] = GOAL_MODELS,
    include_market_benchmark: bool = True,
) -> tuple[dict[str, BacktestResult], list[dict[str, object]]]:
    """Backtest every model with the same configuration.

    Returns ``(results_by_model, summary_rows)``; the summary includes per-fold
    stability and, when prices exist, the de-vigged market on the same sample.
    """
    base = config or BacktestConfig()
    results: dict[str, BacktestResult] = {}
    summary: list[dict[str, object]] = []
    for model in models:
        result = run_backtest(matches, odds, replace(base, model=model, model_version=None))
        results[model] = result
        summary.append(
            {
                "model": model,
                **result.metrics,
                "stability_log_loss": _stability([fm.log_loss for fm in result.fold_metrics]),
                "stability_brier": _stability([fm.brier_score for fm in result.fold_metrics]),
                "per_fold": [asdict(fm) for fm in result.fold_metrics],
            }
        )
    if include_market_benchmark and models:
        market = evaluate_market_benchmark(results[models[0]].records)
        if market["n"] > 0:
            summary.append(
                {
                    "model": f"market_{base.bookmaker or 'de_vig'}",
                    "n": market["n"],
                    "log_loss": market["log_loss"],
                    "brier_score": market["brier_score"],
                    "folds": summary[0].get("folds", 0.0),
                }
            )
    return results, summary
