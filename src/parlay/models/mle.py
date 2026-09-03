"""Direct Maximum Likelihood Estimation (MLE) for football score models.

Solves the exact log-likelihood optimization problem for Poisson and Dixon-Coles
models with time-decay weights, sum-to-zero identifiability constraints, and
optional L2 shrinkage (Ridge regularization).
"""

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .dixon_coles import tau


@dataclass(frozen=True, slots=True)
class MLEResult:
    teams: tuple[str, ...]
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_advantage: float
    rho: float
    converged: bool
    negative_log_likelihood: float


def fit_poisson_mle(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    *,
    l2_reg: float = 0.01,
    include_rho: bool = False,
    maxiter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, float, float, bool, float]:
    """Optimize attack, defense, intercept, home_advantage, and optional rho.

    Parameters
    ----------
    home_idx : 1D array of home team integer indices in [0, n_teams-1]
    away_idx : 1D array of away team integer indices in [0, n_teams-1]
    home_goals : 1D array of observed home goals
    away_goals : 1D array of observed away goals
    weights : 1D array of match sample weights (e.g. time-decay)
    n_teams : Total number of unique teams
    l2_reg : L2 shrinkage parameter on team parameters (attack, defense)
    include_rho : If True, estimate Dixon-Coles rho jointly with team strengths
    maxiter : Maximum iterations for scipy.optimize.minimize (L-BFGS-B)

    Returns
    -------
    (attack, defense, intercept, home_advantage, rho, converged, nll)
    """
    # Parameter vector layout:
    # [0 : n_teams-1]               -> attack for teams 0 .. T-2 (team T-1 is -sum)
    # [n_teams-1 : 2*(n_teams-1)]   -> defense for teams 0 .. T-2 (team T-1 is -sum)
    # [2*(n_teams-1)]               -> intercept (mu)
    # [2*(n_teams-1) + 1]           -> home_advantage (gamma)
    # (optional) [2*(n_teams-1) + 2]-> rho

    n_free_team_params = n_teams - 1

    def unpack(params: np.ndarray):
        att_free = params[:n_free_team_params]
        def_free = params[n_free_team_params : 2 * n_free_team_params]
        att = np.append(att_free, -np.sum(att_free))
        defense = np.append(def_free, -np.sum(def_free))
        mu = params[2 * n_free_team_params]
        gamma = params[2 * n_free_team_params + 1]
        rho_val = params[2 * n_free_team_params + 2] if include_rho else 0.0
        return att, defense, mu, gamma, rho_val

    def objective(params: np.ndarray) -> float:
        att, defense, mu, gamma, rho_val = unpack(params)

        # Log-rates with clipping unified to [-3,3] (§4) — identical to TeamStrengthModel.expected_goals
        home_log = np.clip(mu + gamma + att[home_idx] - defense[away_idx], -3.0, 3.0)
        away_log = np.clip(mu + att[away_idx] - defense[home_idx], -3.0, 3.0)

        lambda_home = np.exp(home_log)
        lambda_away = np.exp(away_log)

        # Base Poisson log-likelihood
        ll = (
            home_goals * home_log - lambda_home
            + away_goals * away_log - lambda_away
        )

        if include_rho and abs(rho_val) > 1e-12:
            low = (home_goals <= 1) & (away_goals <= 1)
            if np.any(low):
                # Apply Dixon-Coles tau factor
                tau_vals = tau(
                    home_goals[low], away_goals[low],
                    lambda_home[low], lambda_away[low], rho_val,
                )
                tau_vals = np.maximum(tau_vals, 1e-12)
                ll[low] += np.log(tau_vals)

        # Weighted negative log-likelihood
        nll = -np.sum(weights * ll)

        # L2 Regularization on team strengths (prior toward average team)
        if l2_reg > 0:
            reg_penalty = 0.5 * l2_reg * (np.sum(att**2) + np.sum(defense**2))
            nll += reg_penalty

        return nll

    # Initial guess
    init_params = np.zeros(2 * n_free_team_params + 2 + (1 if include_rho else 0))
    # Intercept guess = log(average goals per match / 2)
    avg_goals = max(0.5, float(np.average(np.r_[home_goals, away_goals], weights=np.r_[weights, weights])))
    init_params[2 * n_free_team_params] = math.log(avg_goals)
    # Home advantage guess = 0.25
    init_params[2 * n_free_team_params + 1] = 0.25

    # Unified domain with prediction (§4): log_lambda clip [-3,3] so keep mu,gamma consistent
    bounds = [(-3.0, 3.0)] * (2 * n_free_team_params) + [(-2.0, 2.0), (-1.0, 1.0)]
    if include_rho:
        # Use globally valid rho domain (§3) instead of hard [-0.3,0.3]
        try:
            from .dixon_coles import _global_rho_bounds
            # Estimate lambda for bounds using initial rates (before optimization, conservative)
            # Use average rates as proxy; optimizer will respect global interval
            # For now keep [-0.3,0.3] and let objective penalize invalid via tau>=0,
            # but document that true global bounds are computed in fit_team_strength after first pass.
            bounds.append((-0.3, 0.3))
        except Exception:
            bounds.append((-0.3, 0.3))

    res = minimize(
        objective,
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-7},
    )

    att_opt, def_opt, mu_opt, gamma_opt, rho_opt = unpack(res.x)
    return att_opt, def_opt, mu_opt, gamma_opt, rho_opt, bool(res.success), float(res.fun)


def fit_negative_binomial_mle(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    *,
    l2_reg: float = 0.01,
    maxiter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, float, float, bool, float]:
    """Joint NB2 MLE: optimize attack, defense, mu, gamma, log_dispersion.

    NB2: Var = μ + μ²/φ,  φ = exp(logφ) >0. Jointly estimated (§7).
    """
    n_free = n_teams - 1

    def unpack_nb(params: np.ndarray):
        att_free = params[:n_free]
        def_free = params[n_free : 2 * n_free]
        att = np.append(att_free, -np.sum(att_free))
        defense = np.append(def_free, -np.sum(def_free))
        mu = params[2 * n_free]
        gamma = params[2 * n_free + 1]
        log_phi = params[2 * n_free + 2]
        phi = float(np.exp(log_phi))
        return att, defense, mu, gamma, phi

    def objective_nb(params: np.ndarray) -> float:
        att, defense, mu, gamma, phi = unpack_nb(params)
        home_log = np.clip(mu + gamma + att[home_idx] - defense[away_idx], -3.0, 3.0)
        away_log = np.clip(mu + att[away_idx] - defense[home_idx], -3.0, 3.0)
        mu_h = np.exp(home_log)
        mu_a = np.exp(away_log)
        # NB log pmf
        # lgamma(k+phi) - lgamma(phi) - lgamma(k+1) + phi*log(phi/(phi+mu)) + k*log(mu/(phi+mu))
        # Use vectorized lgamma
        import math as _math

        def nb_ll(k, mu_vec):
            # k, mu_vec are arrays
            phi_arr = np.full_like(mu_vec, phi, dtype=float)
            # Use np.vectorize for lgamma
            lg1 = np.vectorize(_math.lgamma)(k + phi_arr)
            lg2 = _math.lgamma(phi)
            lg3 = np.vectorize(_math.lgamma)(k + 1)
            return lg1 - lg2 - lg3 + phi_arr * np.log(phi_arr / (phi_arr + mu_vec)) + k * np.log(mu_vec / (phi_arr + mu_vec))

        ll_h = nb_ll(home_goals, mu_h)
        ll_a = nb_ll(away_goals, mu_a)
        ll = ll_h + ll_a
        nll = -np.sum(weights * ll)
        if l2_reg > 0:
            nll += 0.5 * l2_reg * (np.sum(att**2) + np.sum(defense**2))
        return float(nll)

    init = np.zeros(2 * n_free + 3)
    avg = max(0.5, float(np.average(np.r_[home_goals, away_goals], weights=np.r_[weights, weights])))
    init[2 * n_free] = math.log(avg)
    init[2 * n_free + 1] = 0.25
    init[2 * n_free + 2] = math.log(10.0)  # phi ~10

    bounds = [(-3.0, 3.0)] * (2 * n_free) + [(-2.0, 2.0), (-1.0, 1.0), (math.log(1.0), math.log(200.0))]
    res = minimize(objective_nb, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-7})
    att_opt, def_opt, mu_opt, gamma_opt, phi_opt = unpack_nb(res.x)
    return att_opt, def_opt, mu_opt, gamma_opt, phi_opt, bool(res.success), float(res.fun)
