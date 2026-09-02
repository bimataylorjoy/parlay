"""Calibration and strictly out-of-sample market strategy evaluation."""

from dataclasses import asdict
from typing import Iterable


from parlay.evaluation.metrics import brier_score, log_loss

SELECTIONS = ("home", "draw", "away")


def evaluate_market_benchmark(records: Iterable[object]) -> dict[str, float]:
    """Compute benchmark Log Loss and Brier Score from de-vigged market prices.

    Filters for records where market probabilities are available, providing
    the gold-standard market efficiency baseline for the exact same sample.
    """
    market_log_losses = []
    market_briers = []
    for record in records:
        mh = getattr(record, "market_home", None)
        md = getattr(record, "market_draw", None)
        ma = getattr(record, "market_away", None)
        actual = getattr(record, "actual", None)
        if mh is not None and md is not None and ma is not None and actual is not None:
            probs = {"home_win": float(mh), "draw": float(md), "away_win": float(ma)}
            market_log_losses.append(log_loss(probs, actual))
            market_briers.append(brier_score(probs, actual))
    if not market_log_losses:
        return {"n": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    n = float(len(market_log_losses))
    return {
        "n": n,
        "log_loss": sum(market_log_losses) / n,
        "brier_score": sum(market_briers) / n,
    }


def calibration_bins(records: Iterable[object], bins: int = 10) -> list[dict[str, float]]:
    """Group each model outcome probability and report empirical frequency."""
    if bins < 1:
        raise ValueError("bins must be positive")
    rows: list[dict[str, float]] = []
    for selection, actual_name in (("home", "home_win"), ("draw", "draw"), ("away", "away_win")):
        values = []
        for record in records:
            probability = getattr(record, f"{selection}_win" if selection != "draw" else "draw")
            values.append((float(probability), float(getattr(record, "actual") == actual_name)))
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            selected = [item for item in values if lower <= item[0] < upper or (index == bins - 1 and item[0] == upper)]
            if selected:
                rows.append({
                    "selection": selection,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": float(len(selected)),
                    "mean_probability": sum(item[0] for item in selected) / len(selected),
                    "empirical_frequency": sum(item[1] for item in selected) / len(selected),
                })
    return rows


def evaluate_flat_stake(records: Iterable[object], *, min_edge: float = 0.03,
                        min_ev: float = 0.02, stake: float = 1.0,
                        max_stake_fraction: float | None = None,
                        use_pinnacle_filter: bool = False,
                        min_pinnacle_edge: float = -0.01) -> dict[str, float]:
    """Evaluate a transparent one-selection-per-match flat-stake strategy.
    
    If use_pinnacle_filter is True, bets are skipped if the model's probability 
    is significantly higher than Pinnacle's efficient probability (i.e. model 
    probability - pinnacle probability < min_pinnacle_edge). This avoids betting
    into sharp line movements.
    """
    if min_edge < 0 or min_ev < 0 or stake <= 0:
        raise ValueError("thresholds must be non-negative and stake must be positive")
    if max_stake_fraction is not None and not 0 < max_stake_fraction <= 1:
        raise ValueError("max_stake_fraction must be in (0, 1]")
    if max_stake_fraction is not None:
        stake = min(stake, max_stake_fraction)
    bets: list[tuple[float, float]] = []
    for record in records:
        candidates = []
        for selection in SELECTIONS:
            edge = getattr(record, f"edge_{selection}", None)
            ev = getattr(record, f"ev_{selection}", None)
            odds = getattr(record, f"odds_{selection}", None)
            
            if edge is not None and ev is not None and odds is not None and edge >= min_edge and ev >= min_ev:
                # Pinnacle Filter
                if use_pinnacle_filter:
                    pin_prob = getattr(record, f"pin_{selection}", None)
                    model_prob = getattr(record, f"{selection}_win" if selection != "draw" else "draw", None)
                    if pin_prob is not None and model_prob is not None:
                        pin_edge = model_prob - pin_prob
                        if pin_edge < min_pinnacle_edge:
                            continue # Skip this bet, Pinnacle disagrees too much
                
                candidates.append((float(ev), selection, float(odds)))
                
        if not candidates:
            continue
        _, selection, odds = max(candidates)
        actual = getattr(record, "actual")
        won = actual == {"home": "home_win", "draw": "draw", "away": "away_win"}[selection]
        bets.append((stake * (odds - 1.0) if won else -stake, stake))
    if not bets:
        return {"bets": 0.0, "turnover": 0.0, "profit": 0.0, "yield": 0.0, "max_drawdown": 0.0}
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for profit, _ in bets:
        cumulative += profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    turnover = sum(item[1] for item in bets)
    return {
        "bets": float(len(bets)),
        "turnover": turnover,
        "profit": cumulative,
        "yield": cumulative / turnover,
        "max_drawdown": max_drawdown,
    }


def compare_closing_prices(records: Iterable[object]) -> dict[str, float]:
    """Summarize taken-vs-closing decimal prices when closing fields exist."""
    ratios = []
    for record in records:
        for selection in SELECTIONS:
            taken = getattr(record, f"odds_{selection}", None)
            closing = getattr(record, f"closing_odds_{selection}", None)
            if taken is not None and closing is not None and closing > 1:
                ratios.append(float(taken) / float(closing) - 1.0)
    return {"n": float(len(ratios)), "mean_price_clv": sum(ratios) / len(ratios) if ratios else 0.0}
