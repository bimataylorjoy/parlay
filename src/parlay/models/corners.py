"""Dedicated modeling and market derivation for football match corners."""

import math
from dataclasses import dataclass
from datetime import date
import numpy as np
from scipy.optimize import minimize

from parlay.data.schemas import Match
from parlay.models.poisson import poisson_pmf
from parlay.models.negative_binomial import negative_binomial_pmf


@dataclass(frozen=True, slots=True)
class CornerStrengthModel:
    """Team-level corner attacking and conceding strengths."""
    intercept: float
    home_advantage: float
    corner_attack: dict[str, float]
    corner_conceding: dict[str, float]
    dispersion: float = 20.0
    max_corners: int = 20

    def expected_corners(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return expected corner rates (lambda_home, lambda_away)."""
        att_h = self.corner_attack.get(home_team, 0.0)
        conc_a = self.corner_conceding.get(away_team, 0.0)
        att_a = self.corner_attack.get(away_team, 0.0)
        conc_h = self.corner_conceding.get(home_team, 0.0)

        # Log rates with safety bounds
        log_h = max(min(self.intercept + self.home_advantage + att_h + conc_a, 3.5), 0.5)
        log_a = max(min(self.intercept + att_a + conc_h, 3.5), 0.5)
        return float(math.exp(log_h)), float(math.exp(log_a))

    def corner_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return joint probability matrix P(home_corners=i, away_corners=j)."""
        exp_h, exp_a = self.expected_corners(home_team, away_team)
        grid = np.arange(self.max_corners + 1)
        if self.dispersion < 100.0:
            h_pmf = negative_binomial_pmf(grid, exp_h, self.dispersion)
            a_pmf = negative_binomial_pmf(grid, exp_a, self.dispersion)
        else:
            h_pmf = poisson_pmf(grid, exp_h)
            a_pmf = poisson_pmf(grid, exp_a)
        matrix = np.outer(h_pmf, a_pmf)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix


def fit_corner_strength(
    matches: list[Match],
    *,
    half_life_days: float = 365.0,
    l2_reg: float = 0.02,
    dispersion: float = 15.0,
    max_corners: int = 20,
) -> CornerStrengthModel:
    """Fit corner attack and conceding strengths using weighted Poisson/NB MLE."""
    valid_matches = [
        m for m in matches
        if m.home_corners is not None and m.away_corners is not None
    ]
    if not valid_matches:
        # Fallback if no corner data exists: neutral average 5.5 corners per match
        return CornerStrengthModel(
            intercept=math.log(5.0),
            home_advantage=0.15,
            corner_attack={},
            corner_conceding={},
            dispersion=dispersion,
            max_corners=max_corners,
        )

    teams = sorted({m.home_team for m in valid_matches} | {m.away_team for m in valid_matches})
    n_teams = len(teams)
    team_idx = {team: i for i, team in enumerate(teams)}

    # Exponential time weights
    latest_date = max(m.date for m in valid_matches)
    weights = np.array([
        0.5 ** ((latest_date - m.date).days / half_life_days)
        for m in valid_matches
    ], dtype=float)

    h_idx = np.array([team_idx[m.home_team] for m in valid_matches], dtype=int)
    a_idx = np.array([team_idx[m.away_team] for m in valid_matches], dtype=int)
    h_corn = np.array([m.home_corners for m in valid_matches], dtype=float)
    a_corn = np.array([m.away_corners for m in valid_matches], dtype=float)

    # Param vector: [att_0..att_T-2, conc_0..conc_T-2, intercept, home_advantage]
    # Sum-to-zero constraint: last team = -sum(others)
    n_free = n_teams - 1
    init_params = np.zeros(2 * n_free + 2)
    avg_total_corners = float((h_corn.sum() + a_corn.sum()) / (2 * len(valid_matches)))
    init_params[2 * n_free] = math.log(max(avg_total_corners, 1.0))
    init_params[2 * n_free + 1] = 0.15

    def objective(params: np.ndarray) -> float:
        att_free = params[:n_free]
        conc_free = params[n_free : 2 * n_free]
        mu = params[2 * n_free]
        gamma = params[2 * n_free + 1]

        att = np.append(att_free, -np.sum(att_free))
        conc = np.append(conc_free, -np.sum(conc_free))

        log_lambda_h = np.clip(mu + gamma + att[h_idx] + conc[a_idx], 0.1, 3.5)
        log_lambda_a = np.clip(mu + att[a_idx] + conc[h_idx], 0.1, 3.5)

        lam_h = np.exp(log_lambda_h)
        lam_a = np.exp(log_lambda_a)

        # Likelihood: Poisson if dispersion>=100, else NB2 joint (§12 audit)
        if dispersion >= 100.0:
            ll_h = weights * (h_corn * log_lambda_h - lam_h)
            ll_a = weights * (a_corn * log_lambda_a - lam_a)
        else:
            # NB2 log pmf joint (§7)
            import math as _math
            phi = float(dispersion)
            # NB ll per observation
            def nb_ll(k, mu_vec):
                phi_arr = np.full_like(mu_vec, phi, dtype=float)
                lg1 = np.vectorize(_math.lgamma)(k + phi_arr)
                lg2 = _math.lgamma(phi)
                lg3 = np.vectorize(_math.lgamma)(k + 1)
                return lg1 - lg2 - lg3 + phi_arr * np.log(phi_arr / (phi_arr + mu_vec)) + k * np.log(mu_vec / (phi_arr + mu_vec))
            ll_h = weights * nb_ll(h_corn, lam_h)
            ll_a = weights * nb_ll(a_corn, lam_a)

        reg = l2_reg * (np.sum(att**2) + np.sum(conc**2))
        return -float(np.sum(ll_h) + np.sum(ll_a)) + reg

    res = minimize(objective, init_params, method="L-BFGS-B")
    params = res.x

    att_free = params[:n_free]
    conc_free = params[n_free : 2 * n_free]
    att = np.append(att_free, -np.sum(att_free))
    conc = np.append(conc_free, -np.sum(conc_free))

    return CornerStrengthModel(
        intercept=float(params[2 * n_free]),
        home_advantage=float(params[2 * n_free + 1]),
        corner_attack={team: float(att[i]) for team, i in team_idx.items()},
        corner_conceding={team: float(conc[i]) for team, i in team_idx.items()},
        dispersion=dispersion,
        max_corners=max_corners,
    )


def corner_totals_probabilities(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Return over/push/under probabilities for total match corners at a given line."""
    matrix = np.asarray(matrix, dtype=float)
    if not math.isfinite(line) or line * 4 != int(line * 4):
        raise ValueError("line must be a multiple of 0.25 (integer, half, or quarter)")

    # Quarter lines: 50% split across adjacent lines
    if (line * 2) % 1 != 0:
        lower = math.floor(line * 2) / 2
        upper = math.ceil(line * 2) / 2
        p_lower = corner_totals_probabilities(matrix, lower)
        p_upper = corner_totals_probabilities(matrix, upper)
        return {
            "over": 0.5 * (p_lower["over"] + p_upper["over"]),
            "push": 0.5 * (p_lower["push"] + p_upper["push"]),
            "under": 0.5 * (p_lower["under"] + p_upper["under"]),
        }

    home, away = np.indices(matrix.shape)
    total = home + away
    return {
        "over": float(matrix[total > line].sum()),
        "push": float(matrix[total == line].sum()),
        "under": float(matrix[total < line].sum()),
    }


def corner_match_betting(matrix: np.ndarray) -> dict[str, float]:
    """Return Most Corners 1X2 market: home_most, tie, away_most."""
    matrix = np.asarray(matrix, dtype=float)
    home, away = np.indices(matrix.shape)
    return {
        "home_most": float(matrix[home > away].sum()),
        "tie": float(matrix[home == away].sum()),
        "away_most": float(matrix[home < away].sum()),
    }
