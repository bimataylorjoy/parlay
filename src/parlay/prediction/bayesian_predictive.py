"""Posterior predictive 1X2 inference for the Bayesian team-strength model.

``P(Y_new | D) = ∫ P(Y_new | θ) P(θ | D) dθ`` is approximated by averaging
score-matrix probabilities over posterior draws instead of plugging in the
posterior mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parlay.models.dixon_coles import score_matrix as dc_score_matrix
from parlay.models.numerics import LOG_RATE_MAX, LOG_RATE_MIN
from parlay.models.poisson import score_matrix as poisson_score_matrix


@dataclass(frozen=True, slots=True)
class PredictionDistribution:
    mean_probability: dict[str, float]
    credible_interval: dict[str, tuple[float, float]]
    lambda_home_mean: float
    lambda_home_ci: tuple[float, float]
    lambda_away_mean: float
    lambda_away_ci: tuple[float, float]
    posterior_samples: int
    r_hat_max: float | None = None
    ess_min: int | None = None
    divergences: int | None = None


def _interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def posterior_predictive_1x2(
    idata: Any,
    home_team: str,
    away_team: str,
    teams: tuple[str, ...],
    *,
    model: str = "poisson",
    rho: float = 0.0,
    max_goals: int | None = None,
    subsample: int | None = 200,
    neutral: bool = False,
    random_seed: int | None = None,
) -> PredictionDistribution:
    """Posterior predictive 1X2 distribution, averaging over (a subsample of) posterior draws."""
    post = idata.posterior
    attack = np.asarray(post["attack"].values)
    defense = np.asarray(post["defense"].values)
    total = attack.shape[0] * attack.shape[1]
    attack = attack.reshape(total, -1)
    defense = defense.reshape(total, -1)
    intercept = np.asarray(post["intercept"].values).reshape(total)
    home_adv = np.asarray(post["home_advantage"].values).reshape(total)

    rng = np.random.default_rng(random_seed)
    if subsample and subsample < total:
        draws = rng.choice(total, size=subsample, replace=False)
    else:
        draws = np.arange(total)
    index = {team: i for i, team in enumerate(teams)}
    hi, ai = index[home_team], index[away_team]
    venue = 0.0 if neutral else 1.0
    lam_h = np.exp(
        np.clip(intercept + venue * home_adv + attack[:, hi] - defense[:, ai], LOG_RATE_MIN, LOG_RATE_MAX)
    )[draws]
    lam_a = np.exp(np.clip(intercept + attack[:, ai] - defense[:, hi], LOG_RATE_MIN, LOG_RATE_MAX))[draws]

    outcomes = np.empty((len(draws), 3))
    for s, (h, a) in enumerate(zip(lam_h, lam_a, strict=True)):
        matrix = (
            dc_score_matrix(h, a, rho=rho, max_goals=max_goals)
            if model == "dixon_coles"
            else poisson_score_matrix(h, a, max_goals=max_goals)
        )
        outcomes[s] = (np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum())
    mean = outcomes.mean(axis=0)
    mean /= mean.sum()

    r_hat = ess = divergences = None
    try:
        import arviz as az

        r_hat = float(az.rhat(idata).to_array().values.max())
        ess = int(az.ess(idata).to_array().values.min())
        if "sample_stats" in idata:
            divergences = int(idata.sample_stats["diverging"].values.sum())
    except (ImportError, KeyError, ValueError):
        pass

    keys = ("home_win", "draw", "away_win")
    return PredictionDistribution(
        mean_probability=dict(zip(keys, mean.tolist(), strict=True)),
        credible_interval={key: _interval(outcomes[:, i]) for i, key in enumerate(keys)},
        lambda_home_mean=float(lam_h.mean()),
        lambda_home_ci=_interval(lam_h),
        lambda_away_mean=float(lam_a.mean()),
        lambda_away_ci=_interval(lam_a),
        posterior_samples=len(draws),
        r_hat_max=r_hat,
        ess_min=ess,
        divergences=divergences,
    )
