"""Shared numerical policy for rate bounds and finite score grids."""

from __future__ import annotations

import math


LOG_RATE_MIN = -3.0
LOG_RATE_MAX = 3.0
DEFAULT_TRUNCATION_EPSILON = 1e-6
DEFAULT_SCORE_CAP = 100


def bounded_log_rate(value: float) -> float:
    """Apply the model's shared finite log-rate domain."""
    return max(min(float(value), LOG_RATE_MAX), LOG_RATE_MIN)


def poisson_tail(rate: float, cutoff: int) -> float:
    """Return P(X > cutoff) using a stable recurrence."""
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and positive")
    if cutoff < 0:
        return 1.0
    probability = math.exp(-rate)
    cumulative = probability
    for k in range(1, cutoff + 1):
        probability *= rate / k
        cumulative += probability
    return max(0.0, min(1.0, 1.0 - cumulative))


def adaptive_support(
    rate: float,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
    cap: int = DEFAULT_SCORE_CAP,
) -> tuple[int, bool]:
    """Find the smallest cutoff with tail below epsilon.

    The boolean indicates whether the configured cap was reached first.
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be finite and positive")
    if cap < 0:
        raise ValueError("cap must be non-negative")
    for cutoff in range(cap + 1):
        if poisson_tail(rate, cutoff) < epsilon:
            return cutoff, False
    return cap, True
