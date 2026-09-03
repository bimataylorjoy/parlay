"""Dynamic team strength research model (§10) — random walk benchmark.

Keep stable baseline (TeamStrengthModel) untouched; this is research alternative:
α[i,t] ~ Normal(α[i,t-1], σ_attack), β[i,t] ~ Normal(β[i,t-1], σ_defense)
Evaluated via walk-forward, benchmarked vs baseline.
"""

from dataclasses import dataclass
from datetime import date
import math
import numpy as np

from parlay.data.schemas import Match
from parlay.data.information import InformationSet
from parlay.models.team_strength import TeamStrengthModel


@dataclass(frozen=True, slots=True)
class DynamicStrengthConfig:
    sigma_attack: float = 0.05
    sigma_defense: float = 0.05
    half_life_days: float = 365.0


def fit_dynamic_strength(
    matches: list[Match],
    *,
    config: DynamicStrengthConfig = DynamicStrengthConfig(),
    as_of: date | None = None,
) -> TeamStrengthModel:
    """Simple dynamic research model: time-decayed baseline + random-walk drift.

    For now implemented as exponential decay baseline plus small drift noise
    proportional to sigma. Full state-space (Kalman) can replace this when
    diagnostics justify complexity. Keeps same prediction contract as baseline.
    """
    from parlay.models.team_strength import fit_team_strength

    # Fit baseline
    base = fit_team_strength(matches, half_life_days=config.half_life_days, as_of=as_of)

    # Apply a deterministic, data-derived recent drift. This is a lightweight
    # research baseline, not a claim of full state-space inference.
    ordered = sorted(matches, key=lambda row: row.date)
    recent = ordered[-min(10, len(ordered)):]
    attack = dict(base.attack)
    defense = dict(base.defense)
    for row in recent:
        attack[row.home_team] += config.sigma_attack * (row.home_goals - row.away_goals)
        attack[row.away_team] += config.sigma_attack * (row.away_goals - row.home_goals)
        defense[row.home_team] += config.sigma_defense * (row.away_goals - row.home_goals)
        defense[row.away_team] += config.sigma_defense * (row.home_goals - row.away_goals)
    mean_attack = sum(attack.values()) / len(attack)
    mean_defense = sum(defense.values()) / len(defense)
    attack = {team: value - mean_attack for team, value in attack.items()}
    defense = {team: value - mean_defense for team, value in defense.items()}
    return TeamStrengthModel(
        teams=base.teams,
        attack=attack,
        defense=defense,
        intercept=base.intercept,
        home_advantage=base.home_advantage,
        model="dynamic_" + base.model,
        rho=base.rho,
        home_dispersion=base.home_dispersion,
        away_dispersion=base.away_dispersion,
        max_goals=base.max_goals,
        fit_converged=base.fit_converged,
        fit_negative_log_likelihood=base.fit_negative_log_likelihood,
        fit_warning="dynamic research drift applied",
    )


def does_dynamic_improve(
    baseline_metrics: dict, dynamic_metrics: dict
) -> bool:
    """Does additional dynamic complexity improve out-of-sample?"""
    return bool(dynamic_metrics.get("log_loss", 9e9) < baseline_metrics.get("log_loss", 9e9))
