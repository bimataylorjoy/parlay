"""Dixon-Coles low-score correction on an explicit finite score grid.

Dixon & Coles (1997) multiply the independent-Poisson probabilities of the
four low-score cells by ``tau``::

    tau(0,0) = 1 - lambda*mu*rho     tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho

and leave every other cell untouched.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .numerics import DEFAULT_TRUNCATION_EPSILON, grid_size, normalized, poisson_pmf

RHO_DOMAIN = (-0.3, 0.3)


def valid_rho_bounds(
    home_rate: float | np.ndarray, away_rate: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Per-match ``[lower, upper]`` interval for which every ``tau >= 0``."""
    hr = np.maximum(home_rate, 1e-6)
    ar = np.maximum(away_rate, 1e-6)
    upper = np.minimum(1.0, 1.0 / (hr * ar))
    lower = np.maximum(-1.0 / hr, -1.0 / ar)
    return lower * 0.99, upper * 0.99


def global_rho_bounds(
    home_rates: np.ndarray, away_rates: np.ndarray, pad: float = 0.99
) -> tuple[float, float]:
    """Intersection of all per-match intervals with the research domain ``RHO_DOMAIN``."""
    lowers, uppers = valid_rho_bounds(home_rates, away_rates)
    lower = max(float(np.max(lowers)), RHO_DOMAIN[0]) if np.size(lowers) else RHO_DOMAIN[0]
    upper = min(float(np.min(uppers)), RHO_DOMAIN[1]) if np.size(uppers) else RHO_DOMAIN[1]
    if lower >= upper:
        raise ValueError("no globally valid Dixon-Coles rho interval")
    return lower * pad, upper * pad


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: float | np.ndarray,
    away_rate: float | np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles correction factor for each score cell (clamped at zero)."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    correction = np.ones(np.broadcast(home_goals, away_goals).shape, dtype=float)
    correction = np.where(
        (home_goals == 0) & (away_goals == 0), 1.0 - home_rate * away_rate * rho, correction
    )
    correction = np.where((home_goals == 0) & (away_goals == 1), 1.0 + home_rate * rho, correction)
    correction = np.where((home_goals == 1) & (away_goals == 0), 1.0 + away_rate * rho, correction)
    correction = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, correction)
    return np.maximum(correction, 0.0)


def estimate_rho(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    grid_min: float | None = None,
    grid_max: float | None = None,
    grid_points: int = 61,
) -> float:
    """Profile ``rho`` by grid search on the weighted log-likelihood, rates held fixed.

    Used when rates come from an estimator that does not fit ``rho`` jointly
    (e.g. the Bayesian posterior mean).
    """
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)
    hr = np.asarray(home_rates, dtype=float)
    ar = np.asarray(away_rates, dtype=float)
    w = np.ones(len(hg)) if weights is None else np.asarray(weights, dtype=float)
    global_min, global_max = global_rho_bounds(hr, ar)
    lo = global_min if grid_min is None else max(grid_min, global_min)
    hi = global_max if grid_max is None else min(grid_max, global_max)
    if lo > hi:
        raise ValueError("rho grid does not overlap the globally valid interval")
    base_ll = (
        hg * np.log(np.maximum(hr, 1e-12)) - hr - gammaln(hg + 1)
        + ag * np.log(np.maximum(ar, 1e-12)) - ar - gammaln(ag + 1)
    )  # fmt: skip
    low = (hg <= 1) & (ag <= 1)
    best_rho, best_ll = 0.0, -np.inf
    for rho in np.linspace(lo, hi, grid_points):
        log_tau = np.zeros(len(hg))
        if np.any(low):
            values = tau(hg[low], ag[low], hr[low], ar[low], rho)
            if np.any(values <= 0):
                continue
            log_tau[low] = np.log(values)
        total = float(np.sum(w * (base_ll + log_tau)))
        if total > best_ll:
            best_ll, best_rho = total, float(rho)
    return best_rho


def score_matrix(
    home_rate: float,
    away_rate: float,
    rho: float = 0.0,
    max_goals: int | None = None,
    *,
    epsilon: float = DEFAULT_TRUNCATION_EPSILON,
) -> np.ndarray:
    """Return a normalized Dixon-Coles score matrix."""
    size = grid_size(home_rate, away_rate, max_goals, epsilon=epsilon)
    goals = np.arange(size + 1)
    matrix = np.outer(poisson_pmf(goals, home_rate), poisson_pmf(goals, away_rate))
    matrix *= tau(goals[:, None], goals[None, :], home_rate, away_rate, rho)
    return normalized(matrix, "Dixon-Coles score matrix")
