"""Market transformations from one normalized score distribution."""

import math
import numpy as np
from collections.abc import Iterable
from parlay.data.schemas import OddsSnapshot

from parlay.models.poisson import outcome_probabilities


def asian_handicap(matrix: np.ndarray, handicap: float) -> dict[str, float]:
    """Return win/push/loss for a home Asian handicap line.

    Supports integer, half, and quarter handicap lines (e.g. 0.0, -0.5, -0.75, +1.25).
    Quarter lines represent a 50/50 split stake across the two nearest half/integer lines.
    """
    if not math.isfinite(handicap) or handicap * 4 != int(handicap * 4):
        raise ValueError("handicap must be an integer, half, or quarter value (multiple of 0.25)")

    # Quarter lines (e.g. -0.25, +0.75): split into lower and upper sub-lines
    if (handicap * 2) % 1 != 0:
        lower = math.floor(handicap * 2) / 2
        upper = math.ceil(handicap * 2) / 2
        p_lower = asian_handicap(matrix, lower)
        p_upper = asian_handicap(matrix, upper)
        return {
            "win": 0.5 * (p_lower["win"] + p_upper["win"]),
            "push": 0.5 * (p_lower["push"] + p_upper["push"]),
            "loss": 0.5 * (p_lower["loss"] + p_upper["loss"]),
        }

    home, away = np.indices(matrix.shape)
    margin = home - away + handicap
    return {
        "win": float(matrix[margin > 0].sum()),
        "push": float(matrix[np.isclose(margin, 0)].sum()),
        "loss": float(matrix[margin < 0].sum()),
    }


def fair_odds(probability: float) -> float:
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    return 1.0 / probability


def de_vig_multiplicative(probs: dict[str, float]) -> dict[str, float]:
    """Proportional normalization (dividing by the bookmaker overround)."""
    total = sum(probs.values())
    if total <= 0 or any(value < 0 for value in probs.values()):
        raise ValueError("implied probabilities must be non-negative and non-zero")
    return {key: value / total for key, value in probs.items()}


def de_vig_power(probs: dict[str, float]) -> dict[str, float]:
    """Power / odds-ratio method: solve sum(p_i ** k) == 1."""
    from scipy.optimize import root_scalar

    items = list(probs.items())
    raw = np.array([v for _, v in items], dtype=float)
    if np.any(raw <= 0):
        raise ValueError("implied probabilities must be positive")

    def obj(k: float) -> float:
        return float(np.sum(raw ** k) - 1.0)

    res = root_scalar(obj, bracket=[0.5, 3.0], method="brentq")
    k_opt = res.root
    normalized = raw ** k_opt
    normalized /= np.sum(normalized)
    return {key: float(normalized[i]) for i, (key, _) in enumerate(items)}


def de_vig_shin(probs: dict[str, float], *, max_iter: int = 100) -> dict[str, float]:
    """Shin's (1992, 1993) model for insider trading in betting markets.

    Solves for the proportion of insider traders z in [0, 1) such that
    sum(p_i) == 1, accounting for the favorite-longshot bias.
    """
    items = list(probs.items())
    beta = np.array([v for _, v in items], dtype=float)
    if np.any(beta <= 0):
        raise ValueError("implied probabilities must be positive")

    # If already sums to 1 (fair book), return directly
    total_beta = np.sum(beta)
    if math.isclose(total_beta, 1.0, abs_tol=1e-8):
        return {key: float(beta[i]) for i, (key, _) in enumerate(items)}

    # Bisection search for z in [0, 1 - 1e-6]
    low, high = 0.0, 1.0 - 1e-6

    def compute_p(z: float) -> np.ndarray:
        sqrt_term = np.sqrt(z**2 + 4 * (1.0 - z) * (beta**2) / total_beta)
        p = (sqrt_term - z) / (2.0 * (1.0 - z))
        return p

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        p_mid = compute_p(mid)
        diff = np.sum(p_mid) - 1.0
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            low = mid
        else:
            high = mid

    p_final = compute_p((low + high) / 2.0)
    p_final /= np.sum(p_final)
    return {key: float(p_final[i]) for i, (key, _) in enumerate(items)}


def de_vig(probs: dict[str, float], method: str = "shin") -> dict[str, float]:
    """De-vig implied probabilities using the requested method.

    Supported methods: 'shin' (default), 'power', 'multiplicative' / 'proportional'.
    """
    if method == "shin":
        return de_vig_shin(probs)
    if method == "power":
        return de_vig_power(probs)
    if method in {"multiplicative", "proportional"}:
        return de_vig_multiplicative(probs)
    raise ValueError(f"unknown de-vig method: {method}")


def model_edge(model_probability: float, decimal_odds: float) -> float:
    if not 0 <= model_probability <= 1 or decimal_odds <= 1:
        raise ValueError("invalid probability or decimal odds")
    return model_probability * decimal_odds - 1.0


def kelly_criterion(
    model_probability: float,
    decimal_odds: float,
    fraction: float = 1.0,
    *,
    max_stake: float = 1.0,
) -> float:
    """Calculate the Kelly criterion fraction for bankroll sizing.
    
    Returns the recommended fraction of the bankroll to wager.
    If the edge is negative, returns 0.0.
    `fraction` allows for Fractional Kelly (e.g., 0.25 for Quarter Kelly).
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if not 0 < max_stake <= 1:
        raise ValueError("max_stake must be in (0, 1]")
    ev = model_edge(model_probability, decimal_odds)
    if ev <= 0:
        return 0.0
    
    b = decimal_odds - 1.0
    full_kelly = ev / b
    return min(float(full_kelly * fraction), max_stake)


def totals_settlement_probabilities(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Return win/push/loss probabilities for an integer, half, or quarter goal line.

    Supports lines like 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5.
    Quarter lines represent a 50/50 split stake across the two adjacent lines.
    """
    matrix = np.asarray(matrix, dtype=float)
    if not math.isfinite(line) or line * 4 != int(line * 4):
        raise ValueError("line must be an integer, half-goal, or quarter-goal value (multiple of 0.25)")

    # Quarter lines (e.g. 2.25, 2.75): 50% split across adjacent integer/half lines
    if (line * 2) % 1 != 0:
        lower = math.floor(line * 2) / 2
        upper = math.ceil(line * 2) / 2
        p_lower = totals_settlement_probabilities(matrix, lower)
        p_upper = totals_settlement_probabilities(matrix, upper)
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


def totals_expected_value(probabilities: dict[str, float], odds: float, side: str = "over") -> float:
    """Expected return minus stake for Over or Under, properly accounting for push settlement."""
    if odds <= 1:
        raise ValueError("decimal odds must be greater than 1")
    side = side.lower()
    if side not in {"over", "under"}:
        raise ValueError("side must be 'over' or 'under'")
    win_p = probabilities[side]
    lose_p = probabilities["under" if side == "over" else "over"]
    # On push, stake is returned (net profit 0). EV = P(win)*(odds - 1) - P(loss)*1.0
    return win_p * (odds - 1.0) - lose_p


def correct_score_probabilities(
    matrix: np.ndarray,
    max_home: int | None = None,
    max_away: int | None = None,
) -> dict[str, float]:
    """Return P(home=i, away=j) for each scoreline in the matrix.

    Keys are formatted as "i-j" (e.g. "1-0", "2-1").
    If max_home / max_away are provided, unlisted score combinations are aggregated
    into an "other" bucket.
    """
    matrix = np.asarray(matrix, dtype=float)
    h_len, a_len = matrix.shape
    h_limit = min(max_home + 1 if max_home is not None else h_len, h_len)
    a_limit = min(max_away + 1 if max_away is not None else a_len, a_len)

    result: dict[str, float] = {}
    for i in range(h_limit):
        for j in range(a_limit):
            result[f"{i}-{j}"] = float(matrix[i, j])

    covered = sum(result.values())
    if covered < 1.0 - 1e-9:
        result["other"] = float(max(0.0, 1.0 - covered))
    return result


def grouped_score_markets(matrix: np.ndarray) -> dict[str, float]:
    """Extract standard grouped / derivative betting markets from a score matrix.

    Includes:
    - btts_yes / btts_no (Both Teams To Score)
    - home_win_to_nil / away_win_to_nil (Win to Nil)
    - score_draw / scoreless_draw (Draw with or without goals)
    - double_chance_1x / double_chance_x2 / double_chance_12
    """
    matrix = np.asarray(matrix, dtype=float)
    home, away = np.indices(matrix.shape)

    btts_yes = float(matrix[(home >= 1) & (away >= 1)].sum())
    btts_no = float(1.0 - btts_yes)

    home_win_to_nil = float(matrix[(home > 0) & (away == 0)].sum())
    away_win_to_nil = float(matrix[(home == 0) & (away > 0)].sum())

    scoreless_draw = float(matrix[0, 0])
    score_draw = float(matrix[np.isclose(home, away) & (home > 0)].sum())

    p_home = float(matrix[home > away].sum())
    p_draw = float(matrix[home == away].sum())
    p_away = float(matrix[home < away].sum())

    return {
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "home_win_to_nil": home_win_to_nil,
        "away_win_to_nil": away_win_to_nil,
        "scoreless_draw": scoreless_draw,
        "score_draw": score_draw,
        "double_chance_1x": p_home + p_draw,
        "double_chance_x2": p_away + p_draw,
        "double_chance_12": p_home + p_away,
    }


def implied_probabilities(
    odds: Iterable[OddsSnapshot],
    *,
    bookmaker: str | None = None,
    method: str = "shin",
) -> dict[str, float]:
    rows = [row for row in odds if bookmaker is None or row.bookmaker == bookmaker]
    rows = [row for row in rows if row.market == "1x2"]
    values = {row.selection: 1.0 / row.odds for row in rows}
    required = {"home", "draw", "away"}
    if set(values) != required:
        raise ValueError("1x2 odds must contain exactly home, draw, and away")
    return de_vig(
        {
            "home_win": values["home"],
            "draw": values["draw"],
            "away_win": values["away"],
        },
        method=method,
    )


def latest_odds_as_of(odds: Iterable[OddsSnapshot], as_of, *, bookmaker: str | None = None) -> list[OddsSnapshot]:
    rows = [row for row in odds if row.captured_at <= as_of and (bookmaker is None or row.bookmaker == bookmaker)]
    if not rows:
        return []
    latest = max(row.captured_at for row in rows)
    return [row for row in rows if row.captured_at == latest and row.market == "1x2"]
