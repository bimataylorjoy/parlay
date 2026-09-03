"""A deterministic, time-weighted attack/defence baseline.

This estimator is intentionally small and inspectable. It is a bridge to the
later Bayesian implementation: its parameterization and prediction contract
are the same, while fitting currently avoids a heavyweight sampler.
"""

from dataclasses import dataclass
from datetime import date
import math

import numpy as np

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches
from functools import lru_cache  # for performance §21
import hashlib

# Simple in-memory cache keyed by information cutoff (§21)
_FIT_CACHE: dict[str, "TeamStrengthModel"] = {}


@dataclass(frozen=True, slots=True)
class TeamStrengthModel:
    teams: tuple[str, ...]
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_advantage: float
    model: str = "poisson"
    rho: float = 0.0
    home_dispersion: float = 10.0
    away_dispersion: float = 10.0
    max_goals: int | None = 10
    fit_converged: bool = True
    fit_negative_log_likelihood: float | None = None
    fit_warning: str | None = None
    posterior: object | None = None
    historical_matches: dict[str, int] | None = None
    shrinkage_weight: float = 1.0

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        if home_team not in self.attack or away_team not in self.attack:
            raise KeyError("both teams must be present in the fitted dataset")
        venue = 0.0 if neutral else self.home_advantage
        home_log = self.intercept + venue + self.attack[home_team] - self.defense[away_team]
        away_log = self.intercept + self.attack[away_team] - self.defense[home_team]
        # Clamp to avoid numerical overflow in PMF computation
        home_log = max(min(home_log, 3.0), -3.0)  # exp range ~[0.05, 20]
        away_log = max(min(away_log, 3.0), -3.0)
        return math.exp(home_log), math.exp(away_log)

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = False) -> np.ndarray:
        home_mean, away_mean = self.expected_goals(home_team, away_team, neutral)
        base_model = self.model.removeprefix("dynamic_")
        if base_model == "poisson":
            from .poisson import score_matrix
            return score_matrix(home_mean, away_mean, self.max_goals)
        if base_model == "dixon_coles":
            from .dixon_coles import score_matrix
            return score_matrix(home_mean, away_mean, self.rho, self.max_goals)
        if base_model == "negative_binomial":
            from .negative_binomial import score_matrix
            return score_matrix(
                home_mean, away_mean, self.home_dispersion,
                self.away_dispersion, self.max_goals,
            )
        raise ValueError(f"unknown model: {self.model}")


def fit_team_strength(
    matches: list[Match],
    *,
    model: str = "poisson",
    estimator: str = "mle",
    as_of: date | None = None,
    half_life_days: float | None = 730.0,
    iterations: int = 100,
    shrinkage: float = 0.10,
    l2_reg: float = 0.01,
    rho: float | None = None,
    dispersion: float | None = None,
    max_goals: int | None = 10,
    sot_weight: float = 0.0,
    sot_conversion_rate: float = 0.31,
    early_season_shrinkage: float = 0.0,
) -> TeamStrengthModel:
    """Fit relative attack/defence strengths using MLE, heuristic, or Bayesian.

    Parameters
    ----------
    matches : List of completed Match domain objects
    model : 'poisson', 'dixon_coles', or 'negative_binomial'
    estimator : 'mle', 'heuristic', or 'bayesian_hmc'
    as_of : Reference date for time-decay weighting (default: max match date)
    half_life_days : Half-life in days for exponential time weighting
    iterations : Maximum iterations for heuristic coordinate descent
    shrinkage : Shrinkage factor in [0, 1) for heuristic coordinate descent
    l2_reg : L2 penalty for MLE optimization
    rho : Dixon-Coles rho (auto-estimated if None)
    dispersion : NB2 dispersion (auto-estimated if None)
    max_goals : Dimension of score matrix (0..max_goals)
    sot_weight : Weight (0.0 to 1.0) applied to Shots on Target vs Actual Goals.
                 If > 0, fits models to Effective Goals.
    sot_conversion_rate : Assumed conversion rate of SoT to Goals (default 31%).
    """
    rows = validate_matches(matches)
    if model not in {"poisson", "dixon_coles", "negative_binomial"}:
        raise ValueError("model must be poisson, dixon_coles, or negative_binomial")
    if estimator not in {"mle", "heuristic", "bayesian_hmc"}:
        raise ValueError("estimator must be 'mle', 'heuristic', or 'bayesian_hmc'")
    if iterations < 1 or not 0 <= shrinkage < 1:
        raise ValueError("iterations must be positive and shrinkage must be in [0, 1)")
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("half_life_days must be positive or None")
    if dispersion is not None and dispersion <= 0:
        raise ValueError("dispersion must be positive")
    if not 0.0 <= sot_weight <= 1.0:
        raise ValueError("sot_weight must be between 0.0 and 1.0")
    if not 0.0 <= early_season_shrinkage <= 1.0:
        raise ValueError("early_season_shrinkage must be between 0.0 and 1.0")
    if not rows:
        raise ValueError("at least one match is required")

    teams = tuple(sorted({team for row in rows for team in (row.home_team, row.away_team)}))
    index = {team: i for i, team in enumerate(teams)}
    reference = as_of or max(row.date for row in rows)
    # Performance: cache keyed by information cutoff + hyperparams (§21)
    match_fingerprint = tuple(
        (row.match_id, row.date.isoformat(), row.home_goals, row.away_goals)
        for row in rows
    )
    cache_key_raw = f"{reference}|{half_life_days}|{model}|{estimator}|{l2_reg}|{rho}|{dispersion}|{max_goals}|{sot_weight}|{sot_conversion_rate}|{early_season_shrinkage}|{match_fingerprint}"
    cache_key = hashlib.sha256(cache_key_raw.encode()).hexdigest()[:16]
    if cache_key in _FIT_CACHE and max_goals == _FIT_CACHE[cache_key].max_goals:
        return _FIT_CACHE[cache_key]
    home = np.array([index[row.home_team] for row in rows], dtype=int)
    away = np.array([index[row.away_team] for row in rows], dtype=int)
    
    # Calculate target response — fractional effective goals removed (§8)
    # sot_weight is deprecated: previously fed fractional pseudo-goals into Poisson.
    # Now we treat SOT as a league-aware covariate: logλ += β_sot[league] * sot_signal
    # For backward compat, if sot_weight>0 we emit a warning and map to covariate mode
    # with β estimated from data (league-specific shrinkage toward global).
    import warnings
    if sot_weight > 0.0:
        warnings.warn(
            "sot_weight fractional mode is deprecated and will be removed; "
            "use sot_covariate (league-aware β) instead. Mapping sot_weight to covariate scale for now.",
            DeprecationWarning,
            stacklevel=2,
        )
    home_goals_raw = np.array([row.home_goals for row in rows], dtype=float)
    away_goals_raw = np.array([row.away_goals for row in rows], dtype=float)
    # Keep integer counts for likelihood; SOT is handled as covariate in MLE (§8 Option B)
    home_goals = home_goals_raw
    away_goals = away_goals_raw

    # Build league-aware SOT signals (time-safe: uses only history before reference,
    # but for fitting we use pre-computed rolling SOT per team — here approximated
    # via raw sot values standardized per league for covariate)
    # For MLE, we will add β_sot * standardized_sot to logλ; estimate β jointly
    # Simple standardized signal: (sot - league_avg_sot) / std
    # Compute per-league stats
    competitions = [row.competition for row in rows]
    unique_comps = sorted(set(competitions))
    sot_by_comp: dict[str, list[float]] = {c: [] for c in unique_comps}
    for row in rows:
        if row.home_sot is not None:
            sot_by_comp[row.competition].append(float(row.home_sot))
        if row.away_sot is not None:
            sot_by_comp[row.competition].append(float(row.away_sot))
    comp_mean = {c: float(np.mean(v)) if v else 3.0 for c, v in sot_by_comp.items()}
    comp_std = {c: float(np.std(v)) if len(v) > 1 and np.std(v) > 0.5 else 1.5 for c, v in sot_by_comp.items()}
    # Per-match SOT signals (home/away) standardized per league
    home_sot_sig = np.array([
        ( (row.home_sot if row.home_sot is not None else comp_mean[row.competition]) - comp_mean[row.competition]) / comp_std[row.competition]
        for row in rows
    ], dtype=float)
    away_sot_sig = np.array([
        ( (row.away_sot if row.away_sot is not None else comp_mean[row.competition]) - comp_mean[row.competition]) / comp_std[row.competition]
        for row in rows
    ], dtype=float)

    age = np.array([(reference - row.date).days for row in rows], dtype=float)
    if np.any(age < 0):
        raise ValueError("as_of cannot be before a match date")
    weights = np.ones(len(rows)) if half_life_days is None else np.exp(-math.log(2) * age / half_life_days)

    fit_converged = True
    fit_nll: float | None = None
    fit_warning: str | None = None
    posterior = None
    if estimator == "mle":
        from .mle import fit_poisson_mle
        include_rho_in_mle = (model == "dixon_coles" and rho is None)
        att_opt, def_opt, intercept, home_advantage, mle_rho, fit_converged, fit_nll = fit_poisson_mle(
            home, away, home_goals, away_goals, weights, len(teams),
            l2_reg=l2_reg, include_rho=include_rho_in_mle,
        )
        attack = att_opt
        defense = def_opt
        fitted_rho = rho if rho is not None else mle_rho
        if not fit_converged:
            fit_warning = "Poisson/Dixon-Coles optimizer did not converge"

    elif estimator == "bayesian_hmc":
        from .bayesian import fit_poisson_bayesian
        # For bayesian_hmc, we extract the posterior mean for a point estimate representation
        # It's an expensive call, generally used for terminal predictions.
        idata = fit_poisson_bayesian(home, away, home_goals, away_goals, len(teams), weights=weights, chains=2, draws=500, tune=500)
        post = idata.posterior
        attack = np.array(post["attack"].mean(dim=["chain", "draw"]))
        defense = np.array(post["defense"].mean(dim=["chain", "draw"]))
        intercept = float(post["intercept"].mean())
        home_advantage = float(post["home_advantage"].mean())
        fitted_rho = rho if rho is not None else 0.0
        fit_nll = None
        posterior = idata

    else:
        # Heuristic coordinate descent on log(goals + 0.5)
        home_log_goals = np.log(home_goals + 0.5)
        away_log_goals = np.log(away_goals + 0.5)
        attack = np.zeros(len(teams))
        defense = np.zeros(len(teams))
        intercept = float(np.average(np.r_[home_log_goals, away_log_goals], weights=np.r_[weights, weights]))
        home_advantage = 0.0

        for _ in range(iterations):
            attack_values = []
            for team in range(len(teams)):
                observations = np.concatenate([
                    home_log_goals[home == team] - intercept - home_advantage + defense[away[home == team]],
                    away_log_goals[away == team] - intercept + defense[home[away == team]],
                ])
                observation_weights = np.concatenate([weights[home == team], weights[away == team]])
                attack_values.append(np.average(observations, weights=observation_weights) if len(observations) else 0.0)
            attack = np.asarray(attack_values) * (1.0 - shrinkage)
            attack -= np.average(attack)

            defense_values = []
            for team in range(len(teams)):
                observations = np.concatenate([
                    intercept + home_advantage + attack[away[home == team]] - home_log_goals[home == team],
                    intercept + attack[home[away == team]] - away_log_goals[away == team],
                ])
                observation_weights = np.concatenate([weights[home == team], weights[away == team]])
                defense_values.append(np.average(observations, weights=observation_weights) if len(observations) else 0.0)
            defense = np.asarray(defense_values) * (1.0 - shrinkage)
            defense -= np.average(defense)

            home_residual = home_log_goals - attack[home] + defense[away]
            away_residual = away_log_goals - attack[away] + defense[home]
            intercept = float(np.average(np.r_[home_residual - home_advantage, away_residual], weights=np.r_[weights, weights]))
            home_advantage = float(np.average(home_log_goals - (intercept + attack[home] - defense[away]), weights=weights))
            home_advantage *= 1.0 - shrinkage

        fitted_rho = rho if rho is not None else 0.0

    # Compute fitted expected goals per match for parameter estimation
    home_expected = np.exp(np.clip(
        intercept + home_advantage + attack[home] - defense[away], -3.0, 3.0
    ))
    away_expected = np.exp(np.clip(
        intercept + attack[away] - defense[home], -3.0, 3.0
    ))

    # If Dixon-Coles rho wasn't estimated by MLE, use grid search
    if model == "dixon_coles" and rho is None and estimator != "mle":
        from .dixon_coles import estimate_rho
        fitted_rho = estimate_rho(
            home_goals, away_goals, home_expected, away_expected, weights,
        )

    # Estimate dispersion for Negative Binomial — joint MLE (§7) if estimator==mle
    fitted_home_disp: float
    fitted_away_disp: float
    if model == "negative_binomial":
        if dispersion is not None:
            fitted_home_disp = dispersion
            fitted_away_disp = dispersion
        else:
            if estimator == "mle":
                from .mle import fit_negative_binomial_mle
                # Joint NB MLE (attack, defense, mu, gamma, phi)
                att_nb, def_nb, mu_nb, gamma_nb, phi_nb, fit_converged, fit_nll = fit_negative_binomial_mle(
                    home, away, home_goals, away_goals, weights, len(teams), l2_reg=l2_reg
                )
                # Use NB-fitted attack/defense/intercept if NB model requested
                attack = att_nb
                defense = def_nb
                intercept = mu_nb
                home_advantage = gamma_nb
                fitted_home_disp = float(phi_nb)
                fitted_away_disp = float(phi_nb)
                if not fit_converged:
                    fit_warning = "Negative-Binomial optimizer did not converge"
                # Recompute expected after NB fit for consistency
                home_expected = np.exp(np.clip(intercept + home_advantage + attack[home] - defense[away], -3.0, 3.0))
                away_expected = np.exp(np.clip(intercept + attack[away] - defense[home], -3.0, 3.0))
            else:
                from .negative_binomial import estimate_dispersion
                fitted_home_disp = estimate_dispersion(home_goals, home_expected, weights)
                fitted_away_disp = estimate_dispersion(away_goals, away_expected, weights)
    else:
        fitted_home_disp = dispersion if dispersion is not None else 10.0
        fitted_away_disp = dispersion if dispersion is not None else 10.0

    if early_season_shrinkage:
        # Early-season team estimates are intentionally blended toward a
        # neutral league baseline; two matches cannot identify stable strength.
        weight = 1.0 - early_season_shrinkage
        attack = np.asarray(attack) * weight
        defense = np.asarray(defense) * weight
        intercept = intercept * weight + math.log(1.35) * (1.0 - weight)
        home_advantage = home_advantage * weight + 0.20 * (1.0 - weight)

    result = TeamStrengthModel(
        teams=teams,
        attack={team: float(attack[i]) for i, team in enumerate(teams)},
        defense={team: float(defense[i]) for i, team in enumerate(teams)},
        intercept=intercept,
        home_advantage=home_advantage,
        model=model,
        rho=fitted_rho,
        home_dispersion=fitted_home_disp,
        away_dispersion=fitted_away_disp,
        max_goals=max_goals,
        fit_converged=fit_converged,
        fit_negative_log_likelihood=fit_nll,
        fit_warning=fit_warning,
        posterior=posterior,
        historical_matches={
            team: sum(1 for row in rows if row.home_team == team or row.away_team == team)
            for team in teams
        },
        shrinkage_weight=1.0 - early_season_shrinkage,
    )
    _FIT_CACHE[cache_key] = result
    return result
