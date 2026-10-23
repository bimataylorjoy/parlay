"""Corner-count model, fitted independently of the goal model.

Corners share the attack/defence parameterization of goals (``defense`` here
means suppressing the opponent's corners) and are fitted with the same MLE
engine using an NB2 likelihood whose dispersion is estimated jointly, because
corner counts are typically over-dispersed relative to Poisson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from parlay.data.schemas import Match

from .mle import fit_rate_model
from .numerics import bounded_log_rate, negative_binomial_pmf, normalized, poisson_pmf
from .team_strength import time_decay_weights

POISSON_DISPERSION_THRESHOLD = 500.0
DEFAULT_MAX_CORNERS = 25
NEUTRAL_LOG_RATE = math.log(5.0)


@dataclass(frozen=True, slots=True)
class CornerStrengthModel:
    """Team-level corner attack (winning corners) and defence (preventing them)."""

    intercept: float
    home_advantage: float
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    dispersion: float = 20.0
    max_corners: int = DEFAULT_MAX_CORNERS
    fit_converged: bool = True
    fit_warning: str | None = None

    def expected_corners(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected corners ``(home, away)``; unseen teams get league-average strength."""
        home_log = (
            self.intercept
            + self.home_advantage
            + self.attack.get(home_team, 0.0)
            - self.defense.get(away_team, 0.0)
        )
        away_log = self.intercept + self.attack.get(away_team, 0.0) - self.defense.get(home_team, 0.0)
        return math.exp(bounded_log_rate(home_log)), math.exp(bounded_log_rate(away_log))

    def corner_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Joint distribution ``P(home_corners=i, away_corners=j)``."""
        exp_h, exp_a = self.expected_corners(home_team, away_team)
        grid = np.arange(self.max_corners + 1)
        if self.dispersion >= POISSON_DISPERSION_THRESHOLD:
            h_pmf, a_pmf = poisson_pmf(grid, exp_h), poisson_pmf(grid, exp_a)
        else:
            h_pmf = negative_binomial_pmf(grid, exp_h, self.dispersion)
            a_pmf = negative_binomial_pmf(grid, exp_a, self.dispersion)
        return normalized(np.outer(h_pmf, a_pmf), "corner matrix")


def fit_corner_strength(
    matches: list[Match],
    *,
    as_of: date | None = None,
    half_life_days: float | None = 365.0,
    l2_reg: float = 5.0,
    dispersion: float | None = None,
    max_corners: int = DEFAULT_MAX_CORNERS,
) -> CornerStrengthModel:
    """Fit corner strengths by weighted NB2 MLE (dispersion estimated unless fixed)."""
    rows = [m for m in matches if m.home_corners is not None and m.away_corners is not None]
    teams = sorted({m.home_team for m in rows} | {m.away_team for m in rows})
    if len(teams) < 2:
        return CornerStrengthModel(
            intercept=NEUTRAL_LOG_RATE,
            home_advantage=0.1,
            dispersion=dispersion or 20.0,
            max_corners=max_corners,
            fit_converged=False,
            fit_warning="no completed corner observations; neutral fallback used",
        )
    index = {team: i for i, team in enumerate(teams)}
    reference = as_of or max(m.date for m in rows)
    fit = fit_rate_model(
        np.array([index[m.home_team] for m in rows]),
        np.array([index[m.away_team] for m in rows]),
        np.array([m.home_corners for m in rows], dtype=float),
        np.array([m.away_corners for m in rows], dtype=float),
        time_decay_weights([m.date for m in rows], reference, half_life_days),
        len(teams),
        family="negative_binomial",
        l2_reg=l2_reg,
        venue=np.array([0.0 if m.neutral else 1.0 for m in rows]),
        fixed_dispersion=dispersion,
    )
    return CornerStrengthModel(
        intercept=fit.intercept,
        home_advantage=fit.home_advantage,
        attack={team: float(fit.attack[i]) for team, i in index.items()},
        defense={team: float(fit.defense[i]) for team, i in index.items()},
        dispersion=float(fit.dispersion or 1000.0),
        max_corners=max_corners,
        fit_converged=fit.converged,
        fit_warning=None if fit.converged else f"corner optimizer did not converge: {fit.message}",
    )
