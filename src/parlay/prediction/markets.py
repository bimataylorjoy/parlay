"""Market probabilities derived from one normalized score (or corner) matrix.

Every market is a sum over cells of the joint distribution. Lines may be
integer, half or quarter values; a quarter line is a stake split equally
across the two adjacent half/integer lines, so its settlement probabilities
are the average of those two lines and expected values stay exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime

import numpy as np
from scipy.optimize import brentq

from parlay.data.schemas import OddsSnapshot
from parlay.models.numerics import validated_matrix

# --------------------------------------------------------------------------- #
# Score-matrix markets
# --------------------------------------------------------------------------- #


def _check_quarter_line(line: float, what: str = "line") -> None:
    if not math.isfinite(line) or line * 4 != int(line * 4):
        raise ValueError(f"{what} must be an integer, half or quarter value (multiple of 0.25)")


def _split_quarter(line: float) -> tuple[float, float] | None:
    if (line * 2) % 1 == 0:
        return None
    return math.floor(line * 2) / 2, math.ceil(line * 2) / 2


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """1X2 probabilities: ``home_win`` (lower triangle), ``draw`` (diagonal), ``away_win``."""
    matrix = validated_matrix(matrix)
    return {
        "home_win": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, 1).sum()),
    }


def totals_probabilities(matrix: np.ndarray, line: float) -> dict[str, float]:
    """``over`` / ``push`` / ``under`` settlement probabilities for a total line."""
    matrix = np.asarray(matrix, dtype=float)
    _check_quarter_line(line)
    halves = _split_quarter(line)
    if halves is not None:
        lower, upper = (totals_probabilities(matrix, part) for part in halves)
        return {key: 0.5 * (lower[key] + upper[key]) for key in ("over", "push", "under")}
    home, away = np.indices(matrix.shape)
    total = home + away
    return {
        "over": float(matrix[total > line].sum()),
        "push": float(matrix[total == line].sum()),
        "under": float(matrix[total < line].sum()),
    }


def team_totals_probabilities(matrix: np.ndarray, line: float, side: str) -> dict[str, float]:
    """Over/push/under for one team's goals (``side`` = ``home`` or ``away``)."""
    if side not in ("home", "away"):
        raise ValueError("side must be 'home' or 'away'")
    matrix = np.asarray(matrix, dtype=float)
    marginal = matrix.sum(axis=1) if side == "home" else matrix.sum(axis=0)
    # Embed the marginal as a 1-row matrix so total == team goals.
    return totals_probabilities(marginal[None, :], line)


def asian_handicap(matrix: np.ndarray, handicap: float) -> dict[str, float]:
    """Home-side ``win`` / ``push`` / ``loss`` for an Asian handicap line (e.g. -0.75)."""
    matrix = np.asarray(matrix, dtype=float)
    _check_quarter_line(handicap, "handicap")
    halves = _split_quarter(handicap)
    if halves is not None:
        lower, upper = (asian_handicap(matrix, part) for part in halves)
        return {key: 0.5 * (lower[key] + upper[key]) for key in ("win", "push", "loss")}
    home, away = np.indices(matrix.shape)
    margin = home - away + handicap
    return {
        "win": float(matrix[margin > 0].sum()),
        "push": float(matrix[np.isclose(margin, 0)].sum()),
        "loss": float(matrix[margin < 0].sum()),
    }


def btts_probabilities(matrix: np.ndarray) -> dict[str, float]:
    matrix = validated_matrix(matrix)
    yes = float(matrix[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def grouped_score_markets(matrix: np.ndarray) -> dict[str, float]:
    """BTTS, win-to-nil, (score)less draw and double-chance probabilities."""
    matrix = np.asarray(matrix, dtype=float)
    home, away = np.indices(matrix.shape)
    p_home = float(matrix[home > away].sum())
    p_draw = float(matrix[home == away].sum())
    p_away = float(matrix[home < away].sum())
    btts_yes = float(matrix[(home >= 1) & (away >= 1)].sum())
    return {
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "home_win_to_nil": float(matrix[(home > 0) & (away == 0)].sum()),
        "away_win_to_nil": float(matrix[(home == 0) & (away > 0)].sum()),
        "scoreless_draw": float(matrix[0, 0]),
        "score_draw": float(matrix[(home == away) & (home > 0)].sum()),
        "double_chance_1x": p_home + p_draw,
        "double_chance_x2": p_away + p_draw,
        "double_chance_12": p_home + p_away,
    }


def correct_score_probabilities(
    matrix: np.ndarray, max_home: int | None = None, max_away: int | None = None
) -> dict[str, float]:
    """``"i-j"`` scoreline probabilities; cells beyond the limits go to ``"other"``."""
    matrix = np.asarray(matrix, dtype=float)
    h_len, a_len = matrix.shape
    h_limit = min(max_home + 1 if max_home is not None else h_len, h_len)
    a_limit = min(max_away + 1 if max_away is not None else a_len, a_len)
    result = {f"{i}-{j}": float(matrix[i, j]) for i in range(h_limit) for j in range(a_limit)}
    covered = sum(result.values())
    if covered < 1.0 - 1e-9:
        result["other"] = max(0.0, 1.0 - covered)
    return result


def most_likely_scores(matrix: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
    matrix = np.asarray(matrix, dtype=float)
    flat = np.argsort(matrix, axis=None)[::-1][:top]
    return [
        (f"{i}-{j}", float(matrix[i, j])) for i, j in zip(*np.unravel_index(flat, matrix.shape), strict=True)
    ]


def match_winner_counts(matrix: np.ndarray) -> dict[str, float]:
    """``home_most`` / ``tie`` / ``away_most`` — e.g. the "most corners" market."""
    matrix = np.asarray(matrix, dtype=float)
    home, away = np.indices(matrix.shape)
    return {
        "home_most": float(matrix[home > away].sum()),
        "tie": float(matrix[home == away].sum()),
        "away_most": float(matrix[home < away].sum()),
    }


# --------------------------------------------------------------------------- #
# Prices, margins and staking
# --------------------------------------------------------------------------- #


def fair_odds(probability: float) -> float:
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    return 1.0 / probability


def expected_value(
    win_probability: float, decimal_odds: float, loss_probability: float | None = None
) -> float:
    """Expected profit per unit stake. Push mass (``1 - win - loss``) returns the stake."""
    if decimal_odds <= 1:
        raise ValueError("decimal odds must be greater than 1")
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be in [0, 1]")
    loss = 1.0 - win_probability if loss_probability is None else loss_probability
    return win_probability * (decimal_odds - 1.0) - loss


def kelly_criterion(
    win_probability: float,
    decimal_odds: float,
    fraction: float = 1.0,
    *,
    loss_probability: float | None = None,
    max_stake: float = 1.0,
) -> float:
    """Fractional Kelly stake as a share of bankroll (0 when there is no edge).

    With a push outcome the growth-optimal stake is
    ``(b·p_win - p_loss) / (b·(p_win + p_loss))`` where ``b = odds - 1``.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if not 0 < max_stake <= 1:
        raise ValueError("max_stake must be in (0, 1]")
    loss = 1.0 - win_probability if loss_probability is None else loss_probability
    if expected_value(win_probability, decimal_odds, loss) <= 0:
        return 0.0
    b = decimal_odds - 1.0
    full_kelly = (b * win_probability - loss) / (b * (win_probability + loss))
    return min(float(full_kelly * fraction), max_stake)


def de_vig_multiplicative(probs: dict[str, float]) -> dict[str, float]:
    """Proportional normalization by the bookmaker overround."""
    total = sum(probs.values())
    if total <= 0 or any(value < 0 for value in probs.values()):
        raise ValueError("implied probabilities must be non-negative and non-zero")
    return {key: value / total for key, value in probs.items()}


def de_vig_power(probs: dict[str, float]) -> dict[str, float]:
    """Power method: find ``k`` with ``sum(p_i ** k) == 1``."""
    keys = list(probs)
    raw = np.array([probs[key] for key in keys], dtype=float)
    if np.any(raw <= 0) or np.any(raw >= 1):
        raise ValueError("implied probabilities must be in (0, 1)")
    total = raw.sum()
    if math.isclose(total, 1.0, abs_tol=1e-12):
        return dict(zip(keys, raw.tolist(), strict=True))
    k = brentq(lambda exponent: float(np.sum(raw**exponent) - 1.0), 0.05, 20.0)
    normalized_probs = raw**k
    normalized_probs /= normalized_probs.sum()
    return dict(zip(keys, normalized_probs.tolist(), strict=True))


def de_vig_shin(probs: dict[str, float]) -> dict[str, float]:
    """Shin (1993): solve for the insider share ``z`` so that fair probabilities sum to one.

    Corrects the favourite-longshot bias better than proportional scaling.
    """
    keys = list(probs)
    beta = np.array([probs[key] for key in keys], dtype=float)
    if np.any(beta <= 0):
        raise ValueError("implied probabilities must be positive")
    total = beta.sum()
    if total <= 1.0 + 1e-12:
        return de_vig_multiplicative(probs)

    def fair(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4.0 * (1.0 - z) * beta**2 / total) - z) / (2.0 * (1.0 - z))

    z = brentq(lambda value: float(fair(value).sum() - 1.0), 0.0, 0.999)
    p = fair(z)
    p /= p.sum()
    return dict(zip(keys, p.tolist(), strict=True))


DE_VIG_METHODS = ("shin", "power", "multiplicative")


def de_vig(probs: dict[str, float], method: str = "shin") -> dict[str, float]:
    """Remove the bookmaker margin with ``shin`` (default), ``power`` or ``multiplicative``."""
    if method == "shin":
        return de_vig_shin(probs)
    if method == "power":
        return de_vig_power(probs)
    if method in ("multiplicative", "proportional"):
        return de_vig_multiplicative(probs)
    raise ValueError(f"unknown de-vig method: {method}")


def overround(decimal_odds: Iterable[float]) -> float:
    """Bookmaker margin: ``sum(1/odds) - 1``."""
    return sum(1.0 / value for value in decimal_odds) - 1.0


def implied_probabilities(
    odds: Iterable[OddsSnapshot], *, bookmaker: str | None = None, method: str = "shin"
) -> dict[str, float]:
    """De-vigged 1X2 probabilities (``home_win``/``draw``/``away_win``) from snapshots."""
    rows = [row for row in odds if row.market == "1x2" and (bookmaker is None or row.bookmaker == bookmaker)]
    values = {row.selection: 1.0 / row.odds for row in rows}
    if set(values) != {"home", "draw", "away"}:
        raise ValueError("1x2 odds must contain exactly home, draw, and away")
    return de_vig(
        {"home_win": values["home"], "draw": values["draw"], "away_win": values["away"]}, method=method
    )


def latest_odds_as_of(
    odds: Iterable[OddsSnapshot],
    as_of: datetime,
    *,
    bookmaker: str | None = None,
    market: str = "1x2",
) -> list[OddsSnapshot]:
    """Latest snapshot of one market captured no later than ``as_of`` (never closing lines)."""
    rows = [
        row
        for row in odds
        if row.market == market
        and row.captured_at <= as_of
        and not row.is_closing
        and (bookmaker is None or row.bookmaker == bookmaker)
    ]
    if not rows:
        return []
    latest = max(row.captured_at for row in rows)
    return [row for row in rows if row.captured_at == latest]
