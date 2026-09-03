"""Posterior predictive inference for Bayesian team strength.

Implements P(Y_new | D) = ∫ P(Y_new | θ) P(θ|D) dθ via Monte Carlo over
posterior draws, avoiding collapse to posterior means (§5).
"""

from dataclasses import dataclass
import math
import numpy as np

from parlay.models.poisson import score_matrix as poisson_score_matrix
from parlay.models.dixon_coles import score_matrix as dc_score_matrix


@dataclass(frozen=True, slots=True)
class PredictionDistribution:
    mean_probability: dict[str, float]  # posterior predictive mean per outcome
    credible_interval: dict[str, tuple[float, float]]  # 5% / 95% per outcome
    lambda_home_mean: float
    lambda_home_ci: tuple[float, float]
    lambda_away_mean: float
    lambda_away_ci: tuple[float, float]
    posterior_samples: int
    # Optional diagnostics
    r_hat_max: float | None = None
    ess_min: int | None = None
    divergences: int | None = None


def posterior_predictive_1x2(
    idata,
    home_team: str,
    away_team: str,
    teams: tuple[str, ...],
    *,
    model: str = "poisson",
    rho: float = 0.0,
    max_goals: int = 10,
    subsample: int | None = 200,
    neutral: bool = False,
) -> PredictionDistribution:
    """Compute posterior predictive 1X2 distribution.

    Subsamples posterior draws to `subsample` (default 200) for speed; set None
    to use all draws.
    """
    # Extract posterior arrays: attack, defense, intercept, home_advantage
    post = idata.posterior
    # PyMC now stores attack/defense directly (exchangeable)
    attack = post["attack"].values  # shape (chain, draw, team)
    defense = post["defense"].values
    intercept = post["intercept"].values
    home_adv = post["home_advantage"].values

    # Flatten chains
    n_chains, n_draws = attack.shape[0], attack.shape[1]
    total = n_chains * n_draws
    idx = np.random.choice(total, size=subsample if subsample and subsample < total else total, replace=False) if subsample else np.arange(total)

    # Map team names to indices
    team_idx = {t: i for i, t in enumerate(teams)}
    hi = team_idx[home_team]
    ai = team_idx[away_team]

    # Gather per-draw parameters
    attack_flat = attack.reshape(total, -1)
    defense_flat = defense.reshape(total, -1)
    intercept_flat = intercept.reshape(total)
    home_adv_flat = home_adv.reshape(total)

    sel_attack = attack_flat[idx]
    sel_defense = defense_flat[idx]
    sel_intercept = intercept_flat[idx]
    sel_home_adv = home_adv_flat[idx]

    # Compute lambdas per draw
    lambdas_home = []
    lambdas_away = []
    home_wins = []
    draws = []
    away_wins = []
    for s in range(len(idx)):
        a_h = float(sel_attack[s, hi])
        d_a = float(sel_defense[s, ai])
        a_a = float(sel_attack[s, ai])
        d_h = float(sel_defense[s, hi])
        gamma = float(sel_home_adv[s]) if not neutral else 0.0
        mu = float(sel_intercept[s])
        lam_h = math.exp(max(min(mu + gamma + a_h - d_a, 3.0), -3.0))
        lam_a = math.exp(max(min(mu + a_a - d_h, 3.0), -3.0))
        lambdas_home.append(lam_h)
        lambdas_away.append(lam_a)
        # Score matrix per draw
        if model == "dixon_coles":
            M = dc_score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)
        else:
            M = poisson_score_matrix(lam_h, lam_a, max_goals=max_goals)
        # 1X2 from matrix
        import numpy as np2
        # reuse outcome logic: home = tril, draw = trace
        home_p = float(np2.tril(M, -1).sum())
        away_p = float(np2.triu(M, 1).sum())
        draw_p = float(np2.trace(M))
        home_wins.append(home_p)
        draws.append(draw_p)
        away_wins.append(away_p)

    def ci(arr):
        return (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))

    mean_prob = {
        "home_win": float(np.mean(home_wins)),
        "draw": float(np.mean(draws)),
        "away_win": float(np.mean(away_wins)),
    }
    # Renormalize mean (should already sum to 1)
    s = sum(mean_prob.values())
    mean_prob = {k: v / s for k, v in mean_prob.items()}

    # Per-outcome CIs
    ci_map = {
        "home_win": ci(home_wins),
        "draw": ci(draws),
        "away_win": ci(away_wins),
    }

    # Diagnostics if available
    r_hat = None
    ess = None
    divs = None
    try:
        import arviz as az
        r_hat = float(max(az.rhat(idata).to_array().values.max(), 0))
        ess_arr = az.ess(idata).to_array().values
        ess = int(ess_arr.min())
        # divergences
        if "sample_stats" in idata:
            divs = int((idata.sample_stats["diverging"].values).sum())
    except Exception:
        pass

    return PredictionDistribution(
        mean_probability=mean_prob,
        credible_interval=ci_map,
        lambda_home_mean=float(np.mean(lambdas_home)),
        lambda_home_ci=ci(lambdas_home),
        lambda_away_mean=float(np.mean(lambdas_away)),
        lambda_away_ci=ci(lambdas_away),
        posterior_samples=len(idx),
        r_hat_max=r_hat,
        ess_min=ess,
        divergences=divs,
    )
