"""Dixon-Coles low-score correction on an explicit finite score grid."""

import math

import numpy as np

from .poisson import poisson_pmf
from .numerics import DEFAULT_TRUNCATION_EPSILON, adaptive_support


def valid_rho_bounds(home_rate: float | np.ndarray, away_rate: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Compute the theoretical [lower, upper] bounds of rho for given rates.

    Dixon & Coles (1997) require tau >= 0 for all four low-score cells:
      1. tau(0,0) = 1 - lambda * mu * rho >= 0  =>  rho <= 1 / (lambda * mu)
      2. tau(0,1) = 1 + lambda * rho >= 0       =>  rho >= -1 / lambda
      3. tau(1,0) = 1 + mu * rho >= 0           =>  rho >= -1 / mu
      4. tau(1,1) = 1 - rho >= 0                =>  rho <= 1
    """
    hr = np.maximum(home_rate, 1e-6)
    ar = np.maximum(away_rate, 1e-6)
    upper = np.minimum(1.0, 1.0 / (hr * ar))
    lower = np.maximum(-1.0 / hr, -1.0 / ar)
    # Give a small safety margin
    return lower * 0.99, upper * 0.99


def _global_rho_bounds(
    home_rates: np.ndarray, away_rates: np.ndarray, pad: float = 0.99
) -> tuple[float, float]:
    """Globally valid rho interval: intersection of per-match intervals."""
    lowers, uppers = valid_rho_bounds(home_rates, away_rates)
    # lowers/uppers may be arrays
    global_lower = float(np.max(lowers)) if np.asarray(lowers).size else -0.3
    global_upper = float(np.min(uppers)) if np.asarray(uppers).size else 0.3
    # Intersect with the documented research domain without expanding the
    # mathematically valid interval. The safety margin is applied inward.
    global_lower = max(global_lower, -0.3)
    global_upper = min(global_upper, 0.3)
    if global_lower >= global_upper:
        raise ValueError("no globally valid Dixon-Coles rho interval")
    return float(global_lower * pad), float(global_upper * pad)


def tau(home_goals: np.ndarray, away_goals: np.ndarray,
        home_rate: float | np.ndarray, away_rate: float | np.ndarray, rho: float) -> np.ndarray:
    """Return the Dixon-Coles correction factor for score cells.

    Note: per-match clipping of rho has been removed (§3). Caller must ensure
    rho lies within the globally valid domain (see _global_rho_bounds). Passing
    an out-of-bounds rho will produce negative corrections clamped to 0, but the
    optimizer should never propose such rho when bounds are set correctly.
    """
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    # No per-match mutation of rho — use rho as-is (globally consistent)
    rho_clamped = rho
    correction = np.ones(np.broadcast(home_goals, away_goals).shape, dtype=float)
    correction = np.where(
        (home_goals == 0) & (away_goals == 0),
        1.0 - home_rate * away_rate * rho_clamped,
        correction,
    )
    correction = np.where(
        (home_goals == 0) & (away_goals == 1),
        1.0 + home_rate * rho_clamped,
        correction,
    )
    correction = np.where(
        (home_goals == 1) & (away_goals == 0),
        1.0 + away_rate * rho_clamped,
        correction,
    )
    correction = np.where(
        (home_goals == 1) & (away_goals == 1),
        1.0 - rho_clamped,
        correction,
    )
    # Guarantee non-negativity
    correction = np.maximum(correction, 0.0)
    return correction


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
    """Estimate rho by maximizing weighted pseudo-log-likelihood over a grid.

    For each match we compute log(tau * Poisson_home * Poisson_away).  The rho
    that maximizes the sum of these (weighted) log-likelihoods is returned.
    Only (0,0), (0,1), (1,0), (1,1) cells are affected; higher scores
    contribute a constant so the grid search is efficient.
    """
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    hr = np.asarray(home_rates, dtype=float)
    ar = np.asarray(away_rates, dtype=float)
    w = np.ones(len(hg), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    global_min, global_max = _global_rho_bounds(hr, ar)
    grid_min = global_min if grid_min is None else max(grid_min, global_min)
    grid_max = global_max if grid_max is None else min(grid_max, global_max)
    if grid_min > grid_max:
        raise ValueError("rho grid does not overlap the globally valid interval")

    # Pre-compute base Poisson log-likelihoods (constant w.r.t. rho)
    base_ll = (
        hg * np.log(np.maximum(hr, 1e-12)) - hr - np.vectorize(math.lgamma)(hg + 1)
        + ag * np.log(np.maximum(ar, 1e-12)) - ar - np.vectorize(math.lgamma)(ag + 1)
    )

    # Mask for low-score matches (only these are affected by rho)
    low = (hg <= 1) & (ag <= 1)

    best_rho = 0.0
    best_ll_total = -np.inf
    for rho in np.linspace(grid_min, grid_max, grid_points):
        log_tau = np.zeros(len(hg))
        if np.any(low):
            tau_values = tau(hg[low], ag[low], hr[low], ar[low], rho)
            if np.any(tau_values <= 0):
                continue
            log_tau[low] = np.log(tau_values)
        total = np.sum(w * (base_ll + log_tau))
        if total > best_ll_total:
            best_ll_total = total
            best_rho = float(rho)
    return best_rho


def score_matrix(home_rate: float, away_rate: float, rho: float = 0.0,
                 max_goals: int | None = 10, *, epsilon: float = DEFAULT_TRUNCATION_EPSILON) -> np.ndarray:
    """Return a normalized Dixon-Coles score matrix.

    The finite grid is normalized after applying the correction. This makes
    truncation explicit and avoids silently dropping rejected samples.
    """
    if max_goals is not None and max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    if max_goals is None:
        home_k, _ = adaptive_support(home_rate, epsilon)
        away_k, _ = adaptive_support(away_rate, epsilon)
        max_goals = max(home_k, away_k)
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson_pmf(goals, home_rate), poisson_pmf(goals, away_rate))
    matrix *= tau(goals[:, None], goals[None, :], home_rate, away_rate, rho)
    total = matrix.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Dixon-Coles score matrix has invalid probability")
    return matrix / total


def sample_scores(matrix: np.ndarray, size: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample score pairs from a normalized score matrix."""
    if size < 1:
        raise ValueError("size must be positive")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score matrix must be square")
    if np.any(matrix < 0) or not np.isclose(matrix.sum(), 1.0, atol=1e-10):
        raise ValueError("score matrix must be non-negative and sum to one")
    generator = rng or np.random.default_rng()
    flat = generator.choice(matrix.size, size=size, p=matrix.ravel())
    return np.unravel_index(flat, matrix.shape)
