"""Calibration and strictly out-of-sample market strategy evaluation."""

from dataclasses import asdict
from typing import Iterable


from parlay.evaluation.metrics import brier_score, log_loss

SELECTIONS = ("home", "draw", "away")


def _multiclass_scores(probabilities: dict[str, float], actual: str) -> tuple[float, float]:
    if not probabilities or actual not in probabilities:
        raise ValueError("actual must be one of the supplied market outcomes")
    if any(value < 0 for value in probabilities.values()) or abs(sum(probabilities.values()) - 1.0) > 1e-8:
        raise ValueError("market probabilities must be non-negative and sum to one")
    return -__import__("math").log(max(probabilities[actual], 1e-15)), sum(
        (value - float(key == actual)) ** 2 for key, value in probabilities.items()
    )


def evaluate_market_predictions(
    records: Iterable[object], *, probabilities_attr: str, actual_attr: str = "actual_market",
) -> dict[str, float]:
    """Score any mutually-exclusive market with log loss and multiclass Brier.

    ``probabilities_attr`` must point to a probability mapping and
    ``actual_attr`` to its realized selection. This deliberately does not
    assume football 1X2 field names.
    """
    rows = list(records)
    scored = []
    for record in rows:
        probabilities = getattr(record, probabilities_attr, None)
        actual = getattr(record, actual_attr, None)
        if probabilities is None or actual is None:
            continue
        scored.append(_multiclass_scores(probabilities, actual))
    if not scored:
        return {"n": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    return {
        "n": float(len(scored)),
        "log_loss": sum(item[0] for item in scored) / len(scored),
        "brier_score": sum(item[1] for item in scored) / len(scored),
    }


def evaluate_binary_market(
    records: Iterable[object], *, probability_attr: str, actual_attr: str,
) -> dict[str, float]:
    """Score a binary market, excluding explicit pushes from binary metrics."""
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
        push_probability = getattr(record, "push_probability", 0.0) or 0.0
        if push_probability:
            decisive_mass = 1.0 - float(push_probability)
            if decisive_mass <= 0:
                pushes += 1
                continue
            p = (p - 0.5 * float(push_probability)) / decisive_mass
        positive = actual in {"over", "yes"}
        p_actual = p if positive else 1.0 - p
        rows.append((-__import__("math").log(max(p_actual, 1e-15)), (p - float(positive)) ** 2))
    if not rows:
        return {"n": 0.0, "pushes": float(pushes), "log_loss": 0.0, "brier_score": 0.0}
    return {
        "n": float(len(rows)), "pushes": float(pushes),
        "log_loss": sum(row[0] for row in rows) / len(rows),
        "brier_score": sum(row[1] for row in rows) / len(rows),
    }


def market_calibration_bins(
    records: Iterable[object], *, probability_attr: str, actual_attr: str, bins: int = 10,
) -> list[dict[str, float]]:
    """Reliability bins for binary over/yes probabilities."""
    if bins < 1:
        raise ValueError("bins must be positive")
    values = []
    for record in records:
        probability = getattr(record, probability_attr, None)
        actual = getattr(record, actual_attr, None)
        if probability is not None and actual in {"over", "under", "yes", "no"}:
            values.append((float(probability), float(actual in {"over", "yes"})))
    output = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [row for row in values if lower <= row[0] < upper or (index == bins - 1 and row[0] <= upper)]
        if selected:
            output.append({
                "bin_lower": lower, "bin_upper": upper, "count": float(len(selected)),
                "mean_probability": sum(row[0] for row in selected) / len(selected),
                "empirical_frequency": sum(row[1] for row in selected) / len(selected),
            })
    return output


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
