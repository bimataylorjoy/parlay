"""Time-weighted attack/defence team-strength models for match goals.

``fit_team_strength`` estimates one shared parameterization for every goal
model (see :mod:`parlay.models.mle`) and returns a :class:`TeamStrengthModel`
that turns two team names into a full score matrix. Every market is derived
from that matrix.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches

from .numerics import bounded_log_rate

GoalModel = Literal["poisson", "dixon_coles", "negative_binomial"]
Estimator = Literal["mle", "bayesian_hmc"]

GOAL_MODELS: tuple[str, ...] = ("poisson", "dixon_coles", "negative_binomial")
ESTIMATORS: tuple[str, ...] = ("mle", "bayesian_hmc")

# Defaults selected by walk-forward tuning on EPL + Championship 2018-2026
# (see README "Benchmarks"). ``l2_reg`` is the precision of a N(0, 1/l2) prior
# on each attack/defence parameter.
DEFAULT_HALF_LIFE_DAYS = 365.0
DEFAULT_L2_REG = 2.5
DEFAULT_SOT_BLEND = 0.5

_CACHE_SIZE = 128
_FIT_CACHE: OrderedDict[str, TeamStrengthModel] = OrderedDict()


@dataclass(frozen=True, slots=True)
class TeamStrengthModel:
    teams: tuple[str, ...]
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_advantage: float
    model: str = "poisson"
    rho: float = 0.0
    home_dispersion: float = 1000.0
    away_dispersion: float = 1000.0
    max_goals: int | None = None
    fit_converged: bool = True
    fit_negative_log_likelihood: float | None = None
    fit_warning: str | None = None
    posterior: Any = field(default=None, compare=False, repr=False)
    historical_matches: dict[str, int] = field(default_factory=dict)
    effective_matches: dict[str, float] = field(default_factory=dict)

    def __contains__(self, team: object) -> bool:
        return team in self.attack

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        missing = [team for team in (home_team, away_team) if team not in self.attack]
        if missing:
            raise KeyError(f"teams not present in the fitted dataset: {missing}")
        venue = 0.0 if neutral else self.home_advantage
        home_log = self.intercept + venue + self.attack[home_team] - self.defense[away_team]
        away_log = self.intercept + self.attack[away_team] - self.defense[home_team]
        return math.exp(bounded_log_rate(home_log)), math.exp(bounded_log_rate(away_log))

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = False) -> np.ndarray:
        home_mean, away_mean = self.expected_goals(home_team, away_team, neutral)
        if self.model == "poisson":
            from .poisson import score_matrix

            return score_matrix(home_mean, away_mean, self.max_goals)
        if self.model == "dixon_coles":
            from .dixon_coles import score_matrix as dc_matrix

            return dc_matrix(home_mean, away_mean, self.rho, self.max_goals)
        if self.model == "negative_binomial":
            from .negative_binomial import score_matrix as nb_matrix

            return nb_matrix(home_mean, away_mean, self.home_dispersion, self.away_dispersion, self.max_goals)
        raise ValueError(f"unknown model: {self.model}")


def time_decay_weights(dates: list[date], reference: date, half_life_days: float | None) -> np.ndarray:
    """Exponential weights ``0.5 ** (age / half_life)``; uniform when ``half_life_days`` is None."""
    age = np.array([(reference - d).days for d in dates], dtype=float)
    if np.any(age < 0):
        raise ValueError("as_of cannot be before a match date")
    if half_life_days is None:
        return np.ones(len(dates))
    return np.exp(-math.log(2.0) * age / half_life_days)


def _cache_key(rows: list[Match], **config: object) -> str:
    digest = hashlib.sha256(repr(sorted(config.items())).encode())
    for row in rows:
        digest.update(
            f"{row.match_id}|{row.date}|{row.home_team}|{row.away_team}|{row.home_goals}|{row.away_goals}|"
            f"{int(row.neutral)}|{row.home_sot}|{row.away_sot}\n".encode()
        )
    return digest.hexdigest()


def _blend_shots_on_target(
    rows: list[Match],
    index: dict[str, int],
    attack: np.ndarray,
    defense: np.ndarray,
    weights: np.ndarray,
    venue: np.ndarray,
    l2_reg: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend goal-based strengths with strengths estimated from shots on target."""
    from .mle import fit_rate_model

    keep = [i for i, row in enumerate(rows) if row.home_sot is not None and row.away_sot is not None]
    if len(keep) < 2:
        return attack, defense
    covered = np.zeros(len(index), dtype=bool)
    home = np.array([index[rows[i].home_team] for i in keep])
    away = np.array([index[rows[i].away_team] for i in keep])
    covered[home] = covered[away] = True
    if covered.sum() < 2:
        return attack, defense
    fit = fit_rate_model(
        home,
        away,
        np.array([rows[i].home_sot for i in keep], dtype=float),
        np.array([rows[i].away_sot for i in keep], dtype=float),
        weights[keep],
        len(index),
        l2_reg=l2_reg,
        venue=venue[keep],
    )
    # Teams absent from the SOT data are pinned to 0 by the prior; keep their goal estimate.
    weight = np.where(covered, blend, 0.0)
    return (1 - weight) * attack + weight * fit.attack, (1 - weight) * defense + weight * fit.defense


def clear_fit_cache() -> None:
    _FIT_CACHE.clear()


def fit_team_strength(
    matches: list[Match],
    *,
    model: str = "dixon_coles",
    estimator: str = "mle",
    as_of: date | None = None,
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
    l2_reg: float = DEFAULT_L2_REG,
    rho: float | None = None,
    dispersion: float | None = None,
    max_goals: int | None = None,
    sot_blend: float = DEFAULT_SOT_BLEND,
    use_cache: bool = True,
) -> TeamStrengthModel:
    """Fit relative attack/defence strengths from completed matches.

    Parameters
    ----------
    matches : completed matches; the caller is responsible for passing only
        information available at forecast time (see ``InformationSet``).
    model : ``poisson``, ``dixon_coles`` or ``negative_binomial``.
    estimator : ``mle`` (penalized maximum likelihood) or ``bayesian_hmc``
        (PyMC NUTS, requires the ``bayes`` extra; Poisson/Dixon-Coles only).
    as_of : reference date for time decay (default: latest match date).
    half_life_days : exponential time-decay half-life; ``None`` disables decay.
    l2_reg : precision of the Gaussian prior on attack/defence parameters.
    rho : fix the Dixon-Coles ``rho`` instead of estimating it jointly.
    dispersion : fix the NB2 dispersion instead of estimating it jointly.
    max_goals : score-grid size; ``None`` sizes the grid adaptively.
    sot_blend : weight in ``[0, 1]`` of shots-on-target strengths. Attack and
        defence are also estimated from shots on target (same likelihood
        engine, Poisson), and each team parameter becomes
        ``(1 - w) * goals_estimate + w * sot_estimate``. Shots on target are
        roughly four times as frequent as goals, so they carry a less noisy
        signal of team quality. Teams without SOT data keep the goals
        estimate. ``0`` uses goals only.
    """
    rows = validate_matches(matches)
    if model not in GOAL_MODELS:
        raise ValueError(f"model must be one of {GOAL_MODELS}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("half_life_days must be positive or None")
    if dispersion is not None and dispersion <= 0:
        raise ValueError("dispersion must be positive")
    if l2_reg < 0:
        raise ValueError("l2_reg must be non-negative")
    if not 0.0 <= sot_blend <= 1.0:
        raise ValueError("sot_blend must be between 0 and 1")
    if not rows:
        raise ValueError("at least one match is required")
    if estimator == "bayesian_hmc" and model == "negative_binomial":
        raise ValueError("bayesian_hmc supports poisson and dixon_coles only")

    reference = as_of or max(row.date for row in rows)
    key = ""
    if use_cache:
        key = _cache_key(
            rows,
            reference=reference,
            model=model,
            estimator=estimator,
            half_life=half_life_days,
            l2=l2_reg,
            rho=rho,
            dispersion=dispersion,
            max_goals=max_goals,
            sot_blend=sot_blend,
        )
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            _FIT_CACHE.move_to_end(key)
            return cached

    teams = tuple(sorted({team for row in rows for team in (row.home_team, row.away_team)}))
    index = {team: i for i, team in enumerate(teams)}
    home = np.array([index[row.home_team] for row in rows], dtype=int)
    away = np.array([index[row.away_team] for row in rows], dtype=int)
    home_goals = np.array([row.home_goals for row in rows], dtype=float)
    away_goals = np.array([row.away_goals for row in rows], dtype=float)
    venue = np.array([0.0 if row.neutral else 1.0 for row in rows])
    weights = time_decay_weights([row.date for row in rows], reference, half_life_days)

    posterior = None
    fit_warning = None
    home_disp = away_disp = 1000.0
    if estimator == "mle":
        from .mle import fit_rate_model

        fit = fit_rate_model(
            home,
            away,
            home_goals,
            away_goals,
            weights,
            len(teams),
            family="negative_binomial" if model == "negative_binomial" else "poisson",
            dixon_coles=model == "dixon_coles",
            l2_reg=l2_reg,
            venue=venue,
            fixed_rho=rho,
            fixed_dispersion=dispersion,
        )
        attack, defense = fit.attack, fit.defense
        intercept, home_advantage = fit.intercept, fit.home_advantage
        fitted_rho = fit.rho if model == "dixon_coles" else 0.0
        if fit.dispersion is not None:
            home_disp = away_disp = fit.dispersion
        converged, nll = fit.converged, fit.negative_log_likelihood
        if not converged:
            fit_warning = f"optimizer did not converge: {fit.message}"
    else:
        from .bayesian import fit_poisson_bayesian
        from .dixon_coles import estimate_rho

        posterior = fit_poisson_bayesian(
            home, away, home_goals, away_goals, len(teams), weights=weights, chains=2, draws=500, tune=500
        )
        post = posterior.posterior
        attack = np.asarray(post["attack"].mean(dim=["chain", "draw"]))
        defense = np.asarray(post["defense"].mean(dim=["chain", "draw"]))
        intercept = float(post["intercept"].mean())
        home_advantage = float(post["home_advantage"].mean())
        converged, nll = True, None
        fitted_rho = 0.0
        if model == "dixon_coles":
            if rho is not None:
                fitted_rho = rho
            else:
                eta_h = np.clip(intercept + home_advantage * venue + attack[home] - defense[away], -3, 3)
                eta_a = np.clip(intercept + attack[away] - defense[home], -3, 3)
                fitted_rho = estimate_rho(home_goals, away_goals, np.exp(eta_h), np.exp(eta_a), weights)

    if sot_blend > 0:
        attack, defense = _blend_shots_on_target(
            rows, index, np.asarray(attack), np.asarray(defense), weights, venue, l2_reg, sot_blend
        )

    counts = np.bincount(home, minlength=len(teams)) + np.bincount(away, minlength=len(teams))
    effective = np.bincount(home, weights, len(teams)) + np.bincount(away, weights, len(teams))
    result = TeamStrengthModel(
        teams=teams,
        attack={team: float(attack[i]) for i, team in enumerate(teams)},
        defense={team: float(defense[i]) for i, team in enumerate(teams)},
        intercept=float(intercept),
        home_advantage=float(home_advantage),
        model=model,
        rho=float(fitted_rho),
        home_dispersion=float(home_disp),
        away_dispersion=float(away_disp),
        max_goals=max_goals,
        fit_converged=converged,
        fit_negative_log_likelihood=nll,
        fit_warning=fit_warning,
        posterior=posterior,
        historical_matches={team: int(counts[i]) for i, team in enumerate(teams)},
        effective_matches={team: float(effective[i]) for i, team in enumerate(teams)},
    )
    if use_cache:
        _FIT_CACHE[key] = result
        while len(_FIT_CACHE) > _CACHE_SIZE:
            _FIT_CACHE.popitem(last=False)
    return result
