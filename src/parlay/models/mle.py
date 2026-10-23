"""Weighted maximum-likelihood engine for attack/defence count models.

One engine serves goals (Poisson, Dixon-Coles, NB2) and corners (Poisson,
NB2). The log-rate of each side is::

    log λ_home = μ + γ·venue + attack[home] - defense[away]
    log λ_away = μ           + attack[away] - defense[home]

with sum-to-zero constraints on ``attack`` and ``defense`` (the last team is
minus the sum of the others), a Gaussian (L2) prior of precision ``l2_reg`` on
team parameters, and per-match weights (e.g. exponential time decay). The
objective and its analytic gradient are fully vectorized, so L-BFGS-B needs
one function evaluation per step instead of one per parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from .dixon_coles import RHO_DOMAIN
from .numerics import LOG_RATE_MAX, LOG_RATE_MIN

Family = Literal["poisson", "negative_binomial"]

TEAM_BOUND = (-3.0, 3.0)
INTERCEPT_BOUND = (-2.0, 3.0)
HOME_ADVANTAGE_BOUND = (-1.0, 1.0)
LOG_DISPERSION_BOUND = (math.log(0.5), math.log(1000.0))
TAU_FLOOR = 1e-10


@dataclass(frozen=True, slots=True)
class RateFit:
    """Fitted parameters of an attack/defence rate model."""

    attack: np.ndarray
    defense: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    dispersion: float | None
    converged: bool
    negative_log_likelihood: float
    iterations: int
    message: str


def _expand(free: np.ndarray) -> np.ndarray:
    return np.append(free, -free.sum())


def _contract(grad_full: np.ndarray) -> np.ndarray:
    # d/d free_k of f(free_0..free_{n-2}, -sum(free)) = g_k - g_last
    return grad_full[:-1] - grad_full[-1]


def fit_rate_model(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_counts: np.ndarray,
    away_counts: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    *,
    family: Family = "poisson",
    dixon_coles: bool = False,
    l2_reg: float = 0.0,
    venue: np.ndarray | None = None,
    fixed_rho: float | None = None,
    fixed_dispersion: float | None = None,
    maxiter: int = 2000,
) -> RateFit:
    """Fit the rate model by weighted MLE (or MAP when ``l2_reg > 0``).

    ``venue`` is 1 where home advantage applies and 0 for neutral grounds.
    ``fixed_rho`` / ``fixed_dispersion`` hold those parameters constant
    instead of estimating them.
    """
    if n_teams < 2:
        raise ValueError("at least two teams are required")
    if family not in ("poisson", "negative_binomial"):
        raise ValueError("family must be 'poisson' or 'negative_binomial'")
    if dixon_coles and family != "poisson":
        raise ValueError("the Dixon-Coles correction is defined for the Poisson family")
    if l2_reg < 0:
        raise ValueError("l2_reg must be non-negative")

    h = np.asarray(home_idx, dtype=int)
    a = np.asarray(away_idx, dtype=int)
    yh = np.asarray(home_counts, dtype=float)
    ya = np.asarray(away_counts, dtype=float)
    w = np.asarray(weights, dtype=float)
    v = np.ones(len(h)) if venue is None else np.asarray(venue, dtype=float)

    n_free = n_teams - 1
    estimate_rho = dixon_coles and fixed_rho is None
    estimate_phi = family == "negative_binomial" and fixed_dispersion is None
    i_mu, i_gamma = 2 * n_free, 2 * n_free + 1
    i_rho = i_gamma + 1 if estimate_rho else None
    i_phi = (i_gamma + 1 + int(estimate_rho)) if estimate_phi else None
    n_params = i_gamma + 1 + int(estimate_rho) + int(estimate_phi)

    low = (yh <= 1) & (ya <= 1) if dixon_coles else np.zeros(len(h), dtype=bool)
    c00 = low & (yh == 0) & (ya == 0)
    c01 = low & (yh == 0) & (ya == 1)
    c10 = low & (yh == 1) & (ya == 0)
    c11 = low & (yh == 1) & (ya == 1)
    lgamma_yh = gammaln(yh + 1.0)
    lgamma_ya = gammaln(ya + 1.0)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float | None]:
        att = _expand(theta[:n_free])
        dfn = _expand(theta[n_free : 2 * n_free])
        rho = float(theta[i_rho]) if i_rho is not None else float(fixed_rho or 0.0)
        if family == "negative_binomial":
            phi = math.exp(theta[i_phi]) if i_phi is not None else float(fixed_dispersion)  # type: ignore[arg-type]
        else:
            phi = None
        return att, dfn, float(theta[i_mu]), float(theta[i_gamma]), rho, phi

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        att, dfn, mu, gamma, rho, phi = unpack(theta)
        eta_h = mu + gamma * v + att[h] - dfn[a]
        eta_a = mu + att[a] - dfn[h]
        in_h = (eta_h > LOG_RATE_MIN) & (eta_h < LOG_RATE_MAX)
        in_a = (eta_a > LOG_RATE_MIN) & (eta_a < LOG_RATE_MAX)
        eta_h = np.clip(eta_h, LOG_RATE_MIN, LOG_RATE_MAX)
        eta_a = np.clip(eta_a, LOG_RATE_MIN, LOG_RATE_MAX)
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)

        grad = np.zeros(n_params)
        if phi is None:
            ll = yh * eta_h - lam_h - lgamma_yh + ya * eta_a - lam_a - lgamma_ya
            g_h = yh - lam_h
            g_a = ya - lam_a
        else:
            log_ph = np.log(phi + lam_h)
            log_pa = np.log(phi + lam_a)
            log_phi = math.log(phi)
            lg_phi = math.lgamma(phi)
            ll = (
                gammaln(yh + phi) - lg_phi - lgamma_yh + phi * (log_phi - log_ph) + yh * (eta_h - log_ph)
                + gammaln(ya + phi) - lg_phi - lgamma_ya + phi * (log_phi - log_pa) + ya * (eta_a - log_pa)
            )  # fmt: skip
            g_h = phi * (yh - lam_h) / (phi + lam_h)
            g_a = phi * (ya - lam_a) / (phi + lam_a)
            if i_phi is not None:
                dg_phi = digamma(phi)
                d_phi = (
                    digamma(yh + phi) - dg_phi + log_phi - log_ph + 1.0 - (phi + yh) / (phi + lam_h)
                    + digamma(ya + phi) - dg_phi + log_phi - log_pa + 1.0 - (phi + ya) / (phi + lam_a)
                )  # fmt: skip
                grad[i_phi] = -float(np.sum(w * d_phi)) * phi

        if dixon_coles and np.any(low):
            t = np.ones(len(h))
            t[c00] = 1.0 - lam_h[c00] * lam_a[c00] * rho
            t[c01] = 1.0 + lam_h[c01] * rho
            t[c10] = 1.0 + lam_a[c10] * rho
            t[c11] = 1.0 - rho
            t_safe = np.maximum(t, TAU_FLOOR)
            ll = ll + np.where(low, np.log(t_safe), 0.0)
            # d log(tau) / d eta and d rho for each low-score cell
            dh = np.zeros(len(h))
            da = np.zeros(len(h))
            dr = np.zeros(len(h))
            dh[c00] = da[c00] = -lam_h[c00] * lam_a[c00] * rho / t_safe[c00]
            dr[c00] = -lam_h[c00] * lam_a[c00] / t_safe[c00]
            dh[c01] = lam_h[c01] * rho / t_safe[c01]
            dr[c01] = lam_h[c01] / t_safe[c01]
            da[c10] = lam_a[c10] * rho / t_safe[c10]
            dr[c10] = lam_a[c10] / t_safe[c10]
            dr[c11] = -1.0 / t_safe[c11]
            g_h = g_h + dh
            g_a = g_a + da
            if i_rho is not None:
                grad[i_rho] = -float(np.sum(w * dr))

        g_h = w * g_h * in_h
        g_a = w * g_a * in_a
        grad_att = np.bincount(h, g_h, n_teams) + np.bincount(a, g_a, n_teams)
        grad_def = -np.bincount(a, g_h, n_teams) - np.bincount(h, g_a, n_teams)

        nll = -float(np.sum(w * ll)) + 0.5 * l2_reg * float(att @ att + dfn @ dfn)
        grad[:n_free] = _contract(-grad_att + l2_reg * att)
        grad[n_free : 2 * n_free] = _contract(-grad_def + l2_reg * dfn)
        grad[i_mu] = -float(g_h.sum() + g_a.sum())
        grad[i_gamma] = -float(np.sum(g_h * v))
        return nll, grad

    theta0 = np.zeros(n_params)
    mean_count = max(0.05, float(np.average(np.r_[yh, ya], weights=np.r_[w, w])))
    theta0[i_mu] = math.log(mean_count)
    theta0[i_gamma] = 0.2
    bounds = [TEAM_BOUND] * (2 * n_free) + [INTERCEPT_BOUND, HOME_ADVANTAGE_BOUND]
    if i_rho is not None:
        bounds.append(RHO_DOMAIN)
    if i_phi is not None:
        theta0[i_phi] = math.log(20.0)
        bounds.append(LOG_DISPERSION_BOUND)

    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    att, dfn, mu, gamma, rho, phi = unpack(result.x)
    return RateFit(
        attack=att,
        defense=dfn,
        intercept=mu,
        home_advantage=gamma,
        rho=rho,
        dispersion=phi,
        converged=bool(result.success),
        negative_log_likelihood=float(result.fun),
        iterations=int(result.nit),
        message=str(result.message),
    )
