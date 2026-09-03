"""Hyperparameter optimization grid search."""

from typing import Any
from parlay.data.schemas import Match
from parlay.evaluation.backtest import run_backtest_full


def tune_half_life(
    matches: list[Match],
    *,
    model: str = "poisson",
    estimator: str = "mle",
    half_life_candidates: list[float | None] | None = None,
    initial_train_days: int = 730,
    test_days: int = 30,
    step_days: int | None = None,
    max_goals: int = 10,
) -> list[dict[str, Any]]:
    """Evaluate backtest performance across different half_life_days values.
    
    Returns a list of dictionaries containing metrics for each candidate.
    """
    if half_life_candidates is None:
        half_life_candidates = [90.0, 180.0, 270.0, 365.0, 540.0, 730.0, 1095.0, None]
        
    results = []
    
    for hl in half_life_candidates:
        result = run_backtest_full(
            matches, model=model, estimator=estimator,
            initial_train_days=initial_train_days, test_days=test_days,
            step_days=step_days, half_life_days=hl,
            max_goals=max_goals,
        )
        
        # Calculate stability
        fold_log_losses = [fm.log_loss for fm in result.fold_metrics]
        mean_ll = sum(fold_log_losses) / len(fold_log_losses) if fold_log_losses else 0.0
        
        results.append({
            "half_life_days": hl,
            "n": result.metrics["n"],
            "log_loss": result.metrics["log_loss"],
            "brier_score": result.metrics["brier_score"],
            "folds": result.metrics["folds"],
            "mean_fold_log_loss": mean_ll,
        })
        
    # Sort by log loss (ascending)
    results.sort(key=lambda x: x["log_loss"] if x["log_loss"] > 0 else float('inf'))
    
    return results


def tune_half_life_nested(
    matches: list[Match],
    *,
    model: str = "poisson",
    estimator: str = "mle",
    half_life_candidates: list[float | None] | None = None,
    development_end: str | None = None,
    holdout_days: int = 90,
    initial_train_days: int = 730,
    test_days: int = 30,
) -> dict[str, Any]:
    """Nested temporal tuning (§14): tune on development, evaluate on untouched holdout.

    Returns {tuned_on, evaluated_on, best_half_life, holdout_metrics}
    """
    from datetime import date, timedelta
    if half_life_candidates is None:
        half_life_candidates = [180.0, 365.0, 730.0, None]
    # Split development vs holdout by date
    sorted_matches = sorted(matches, key=lambda m: m.date)
    last_date = sorted_matches[-1].date
    if development_end:
        dev_end = date.fromisoformat(development_end)
    else:
        dev_end = last_date - timedelta(days=holdout_days)
    development = [m for m in sorted_matches if m.date <= dev_end]
    holdout = [m for m in sorted_matches if m.date > dev_end]
    if not development or not holdout:
        raise ValueError("development/holdout split empty — check dates")
    # Inner tuning on development
    inner_results = tune_half_life(
        development, model=model, estimator=estimator, half_life_candidates=half_life_candidates,
        initial_train_days=initial_train_days, test_days=test_days,
    )
    best = min(inner_results, key=lambda x: x["log_loss"])
    best_hl = best["half_life_days"]
    # Final evaluation on holdout with locked config
    from parlay.evaluation.backtest import run_backtest_full
    holdout_res = run_backtest_full(
        holdout + development[-initial_train_days:],  # ensure train has history
        model=model, estimator=estimator, half_life_days=best_hl,
        initial_train_days=initial_train_days, test_days=test_days,
    )
    return {
        "tuned_on": f"development <= {dev_end.isoformat()} ({len(development)} matches)",
        "evaluated_on": f"holdout > {dev_end.isoformat()} ({len(holdout)} matches)",
        "hyperparameters": {"half_life_days": best_hl},
        "selection_metric": "log_loss",
        "holdout_metrics": holdout_res.metrics,
        "inner_results": inner_results,
    }
