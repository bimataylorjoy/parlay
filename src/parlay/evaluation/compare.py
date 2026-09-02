"""Run identical temporal folds across all supported baseline models."""

import math
from dataclasses import asdict

from parlay.data.schemas import Match, OddsSnapshot
from parlay.evaluation.backtest import BacktestResult, run_backtest_full
from parlay.evaluation.market import evaluate_market_benchmark


def _stability(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of per-fold scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def compare_models(
    matches: list[Match], *, odds: list[OddsSnapshot] | None = None,
    bookmaker: str | None = None,
    models: tuple[str, ...] = ("poisson", "dixon_coles", "negative_binomial"),
    estimator: str = "mle",
    include_market_benchmark: bool = True,
    initial_train_days: int = 730, test_days: int = 30,
    step_days: int | None = None, half_life_days: float | None = 730.0,
    max_goals: int = 10,
    sot_weight: float = 0.0,
) -> tuple[dict[str, list[object]], list[dict[str, float]]]:
    """Run every model against the same temporal configuration.

    Returns (all_records, summary) where summary includes per-fold breakdown,
    stability statistics for each model, and (if odds are available) the market
    efficiency benchmark.
    """
    all_records: dict[str, list[object]] = {}
    summary: list[dict[str, float]] = []
    for model in models:
        result = run_backtest_full(
            matches, model=model, estimator=estimator, odds=odds, bookmaker=bookmaker,
            initial_train_days=initial_train_days, test_days=test_days,
            step_days=step_days, half_life_days=half_life_days,
            max_goals=max_goals, sot_weight=sot_weight,
        )
        all_records[model] = result.records
        fold_log_losses = [fm.log_loss for fm in result.fold_metrics]
        fold_briers = [fm.brier_score for fm in result.fold_metrics]
        entry: dict[str, object] = {
            "model": model,
            **result.metrics,
            "stability_log_loss": _stability(fold_log_losses),
            "stability_brier": _stability(fold_briers),
            "per_fold": [asdict(fm) for fm in result.fold_metrics],
        }
        summary.append(entry)

    if include_market_benchmark and models:
        first_records = all_records[models[0]]
        mkt = evaluate_market_benchmark(first_records)
        if mkt["n"] > 0:
            summary.append({
                "model": f"market_{bookmaker or 'de_vig'}",
                "n": mkt["n"],
                "log_loss": mkt["log_loss"],
                "brier_score": mkt["brier_score"],
                "folds": float(summary[0].get("folds", 0)),
            })

    return all_records, summary
