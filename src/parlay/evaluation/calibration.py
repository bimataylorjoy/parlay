"""Probability calibration techniques for correcting model over/underconfidence."""

import math
from typing import Iterable
import numpy as np
from scipy.optimize import minimize_scalar

from parlay.evaluation.metrics import log_loss, brier_score

def temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Apply temperature scaling to a probability distribution.
    
    T > 1 softens the probabilities (moves them closer to uniform).
    T < 1 sharpens the probabilities (moves them closer to 0 or 1).
    T = 1 leaves them unchanged.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and strictly positive")
        
    items = list(probs.items())
    raw = np.array([v for _, v in items], dtype=float)
    
    if np.any(raw <= 0):
        # Fallback to avoid log(0) domain errors.
        # Add a tiny epsilon, scale, and re-normalize.
        raw = np.maximum(raw, 1e-15)
        raw /= np.sum(raw)
        
    # Exponentiating by 1/T
    scaled = np.power(raw, 1.0 / temperature)
    scaled /= np.sum(scaled)
    
    return {key: float(scaled[i]) for i, (key, _) in enumerate(items)}


def find_optimal_temperature(records: Iterable[object], *, bounds: tuple[float, float] = (0.5, 3.0)) -> float:
    """Find the optimal temperature T that minimizes log loss over the records."""
    rows = list(records)
    if not rows:
        return 1.0
        
    def objective(t: float) -> float:
        total_nll = 0.0
        for r in rows:
            probs = {
                "home_win": float(getattr(r, "home_win")),
                "draw": float(getattr(r, "draw")),
                "away_win": float(getattr(r, "away_win"))
            }
            scaled = temperature_scale(probs, t)
            total_nll += log_loss(scaled, getattr(r, "actual"))
        return total_nll
        
    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    if res.success:
        return float(res.x)
    return 1.0


def apply_calibration(records: Iterable[object], temperature: float) -> list[dict[str, object]]:
    """Return a list of records with their probabilities replaced by scaled versions,
    along with recomputed log_loss and brier_score.
    """
    import copy
    from dataclasses import asdict
    
    scaled_records = []
    for r in records:
        # Convert dataclass/namespace to dict
        d = asdict(r) if hasattr(r, "__dataclass_fields__") else vars(r).copy()
        
        probs = {
            "home_win": float(d["home_win"]),
            "draw": float(d["draw"]),
            "away_win": float(d["away_win"])
        }
        actual = d["actual"]
        
        scaled = temperature_scale(probs, temperature)
        
        d["home_win"] = scaled["home_win"]
        d["draw"] = scaled["draw"]
        d["away_win"] = scaled["away_win"]
        d["log_loss"] = log_loss(scaled, actual)
        d["brier_score"] = brier_score(scaled, actual)
        
        # Edge/EV are technically invalidated without recalculating against odds,
        # but for pure scoring evaluation we don't strictly need them recalculated here.
        # Ideally we'd re-run model_edge but we leave that to the caller if needed.
        
        scaled_records.append(d)
        
    return scaled_records
