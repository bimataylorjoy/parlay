"""Temperature scaling for 1X2 probabilities, fitted only on past forecasts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from parlay.evaluation.metrics import (
    OUTCOMES,
    aggregate_scores,
    brier_score,
    log_loss,
    ranked_probability_score,
)


def temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """``p_i^(1/T) / sum_j p_j^(1/T)``: ``T > 1`` softens, ``T < 1`` sharpens, ``T = 1`` is identity."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and strictly positive")
    keys = list(probs)
    raw = np.maximum(np.array([probs[key] for key in keys], dtype=float), 1e-15)
    scaled = raw ** (1.0 / temperature)
    scaled /= scaled.sum()
    return dict(zip(keys, scaled.tolist(), strict=True))


def _probability_matrix(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    probs = np.array([[float(getattr(r, key)) for key in OUTCOMES] for r in rows])
    actual = np.array([OUTCOMES.index(getattr(r, "actual")) for r in rows])  # noqa: B009
    return np.maximum(probs, 1e-15), actual


def find_optimal_temperature(records: Iterable[Any], *, bounds: tuple[float, float] = (0.5, 3.0)) -> float:
    """Temperature minimizing mean log loss over ``records`` (1.0 when empty)."""
    rows = list(records)
    if not rows:
        return 1.0
    log_p, actual = _probability_matrix(rows)
    log_p = np.log(log_p)
    picked = np.arange(len(rows))

    def objective(t: float) -> float:
        scaled = log_p / t
        scaled -= scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(scaled).sum(axis=1))
        return float(np.mean(log_norm - scaled[picked, actual]))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x) if result.success else 1.0


def apply_calibration(records: Iterable[Any], temperature: float) -> list[dict[str, Any]]:
    """Rescale each record's 1X2 probabilities and recompute its scores.

    Market-derived fields (edges, EV) are left as computed from the
    uncalibrated model.
    """
    output = []
    for record in records:
        row = asdict(record) if is_dataclass(record) and not isinstance(record, type) else dict(vars(record))
        scaled = temperature_scale({key: float(row[key]) for key in OUTCOMES}, temperature)
        row.update(scaled)
        row["log_loss"] = log_loss(scaled, row["actual"])
        row["brier_score"] = brier_score(scaled, row["actual"])
        if "rps" in row:
            row["rps"] = ranked_probability_score(scaled, row["actual"])
        output.append(row)
    return output


def temporally_safe_calibration(
    records: Iterable[Any], *, train_end: str | None = None, calibrate_end: str | None = None
) -> tuple[float, list[dict[str, Any]], dict[str, float]]:
    """Fit ``T`` on a calibration window and score it on a strictly later test window.

    With explicit boundaries the calibration window is
    ``train_end < forecast <= calibrate_end`` and the test window is
    ``forecast > calibrate_end``; otherwise the last ~40% of records is split
    ~20/20 chronologically at forecast-time boundaries.
    Returns ``(T, calibrated_test_records, test_metrics)``.
    """
    rows = sorted(records, key=lambda r: r.forecast_timestamp)
    if len(rows) < 10:
        raise ValueError("not enough records for temporally safe calibration (need >= 10)")
    if train_end and calibrate_end:
        calibrate = [r for r in rows if train_end < r.forecast_timestamp <= calibrate_end]
        test = [r for r in rows if r.forecast_timestamp > calibrate_end]
        if not calibrate or not test:
            raise ValueError("calibration/test windows are empty; check the timestamps")
    else:
        # Split on timestamps, not positions: many matches share a kickoff time.
        n_part = max(1, len(rows) // 5)
        test_start = rows[-n_part].forecast_timestamp
        calibrate_start = rows[-2 * n_part].forecast_timestamp
        test = [r for r in rows if r.forecast_timestamp >= test_start]
        calibrate = [r for r in rows if calibrate_start <= r.forecast_timestamp < test_start]
        if not calibrate:
            raise ValueError("not enough distinct forecast times for a calibration window")
    temperature = find_optimal_temperature(calibrate)
    calibrated = apply_calibration(test, temperature)
    return temperature, calibrated, aggregate_scores(calibrated)
