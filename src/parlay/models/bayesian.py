"""Bayesian Hamiltonian Monte Carlo backend using PyMC.

This module provides an exact probabilistic posterior for the Poisson score
model. It mirrors the constraints and parametrization of the MLE optimizer,
but allows for full uncertainty quantification.
"""

from typing import Any
import numpy as np

from .numerics import LOG_RATE_MAX, LOG_RATE_MIN

# Lazy loading of PyMC to avoid heavy imports when not explicitly requested
_pymc = None


def _import_pymc():
    global _pymc
    if _pymc is None:
        import pymc as pm
        _pymc = pm
    return _pymc


def fit_poisson_bayesian(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    n_teams: int,
    *,
    weights: np.ndarray | None = None,
    tune: int = 1000,
    draws: int = 1000,
    chains: int = 4,
    random_seed: int | None = None,
) -> Any:
    """Fit a Bayesian Poisson model using PyMC (NUTS sampler).
    
    Currently, PyMC does not natively support sample-level continuous weights 
    (like time-decay) in its standard likelihood functions without custom
    Potential nodes. To keep it robust, if `weights` are provided, we will 
    multiply the log-likelihood via a pm.Potential.
    
    Returns the InferenceData object containing posterior traces.
    """
    pm = _import_pymc()
    import pytensor.tensor as pt
    
    if weights is None:
        weights = np.ones_like(home_goals, dtype=float)
        
    with pm.Model() as model:
        # Global parameters
        intercept = pm.Normal("intercept", mu=1.0, sigma=0.5)
        home_advantage = pm.Normal("home_advantage", mu=0.25, sigma=0.2)
        
        # Exchangeable priors for all teams (§6) — then center
        attack_raw = pm.Normal("attack_raw", mu=0.0, sigma=0.5, shape=n_teams)
        defense_raw = pm.Normal("defense_raw", mu=0.0, sigma=0.5, shape=n_teams)
        attack = pm.Deterministic("attack", attack_raw - pt.mean(attack_raw))
        defense = pm.Deterministic("defense", defense_raw - pt.mean(defense_raw))
        # Keep legacy free names for backward compat traces (optional)
        # attack_free/defense_free no longer used; attack/defense are now exchangeable
        
        # Log expected goals
        home_log = intercept + home_advantage + attack[home_idx] - defense[away_idx]
        away_log = intercept + attack[away_idx] - defense[home_idx]
        
        # Expected goals
        # Keep the Bayesian likelihood on the same finite rate domain used by
        # deterministic fitting and prediction.
        home_theta = pm.math.exp(pm.math.clip(home_log, LOG_RATE_MIN, LOG_RATE_MAX))
        away_theta = pm.math.exp(pm.math.clip(away_log, LOG_RATE_MIN, LOG_RATE_MAX))
        
        # Likelihood
        # If all weights are exactly 1.0, we can use the native ObservedRV
        if np.allclose(weights, 1.0):
            pm.Poisson("home_goals_obs", mu=home_theta, observed=home_goals)
            pm.Poisson("away_goals_obs", mu=away_theta, observed=away_goals)
        else:
            # For fractional/time-decay weights, we use a Potential
            home_ll = pm.logp(pm.Poisson.dist(mu=home_theta), home_goals)
            away_ll = pm.logp(pm.Poisson.dist(mu=away_theta), away_goals)
            pm.Potential("weighted_ll", pt.sum(weights * (home_ll + away_ll)))
            
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )
        
    return idata
