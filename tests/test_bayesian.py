import numpy as np
import pytest

from parlay.models.bayesian import fit_poisson_bayesian

def test_bayesian_model_compiles_and_samples():
    home_idx = np.array([0, 1, 0])
    away_idx = np.array([1, 0, 1])
    home_goals = np.array([2, 1, 1])
    away_goals = np.array([1, 2, 0])
    weights = np.ones(3)
    n_teams = 2
    
    # We do a tiny amount of tuning/draws just to prove the graph compiles and samples
    idata = fit_poisson_bayesian(
        home_idx, away_idx, home_goals, away_goals, n_teams,
        weights=weights, tune=5, draws=5, chains=1, random_seed=42
    )
    
    assert "posterior" in idata
    post = idata.posterior
    
    # Ensure our target deterministic parameters exist
    assert "attack" in post
    assert "defense" in post
    assert "intercept" in post
    assert "home_advantage" in post
    
    # Ensure constraint holds
    att = np.array(post["attack"].mean(dim=["chain", "draw"]))
    assert np.isclose(np.sum(att), 0.0, atol=1e-5)
    
    def_arr = np.array(post["defense"].mean(dim=["chain", "draw"]))
    assert np.isclose(np.sum(def_arr), 0.0, atol=1e-5)
