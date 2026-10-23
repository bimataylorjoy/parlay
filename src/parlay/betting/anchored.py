"""Market-anchored adjustment: does anything improve on the sharp price?

Instead of forecasting from scratch, the model corrects the de-vigged
Pinnacle pre-match probabilities with features the market may price slowly
(Egidi, Pauli & Torelli, 2018; Hubáček et al., 2019 model the residual
against the market rather than the outcome)::

    score_k = log p_market_k + Σ_j w_j · x_jk          p = softmax(score)

Features are outcome-specific: the goal model's log-ratio to the market,
the soft bookmaker's (Bet365) log-ratio to the market, rest-day and
promoted-team differences (+ for home, − for away), and ``log p_market``
itself (a market recalibration term). With ``w = 0`` the forecast *is* the
market, so any out-of-sample gain is information the market lacked.

Weights are fitted walk-forward by season (train on earlier seasons, test on
the next) with an L2 penalty, and the result is accepted only if it beats
the market on held-out log loss *and* raises closing-line value.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy.optimize import minimize

from parlay.data.odds_columns import PSEUDO_BOOKMAKERS
from parlay.data.schemas import Match, OddsSnapshot
from parlay.features.historical import build_pre_match_features
from parlay.prediction.markets import de_vig

from .reference import fair_price
from .value import commission_for, latest_quotes

OUTCOMES = ("home", "draw", "away")
FEATURES = ("model_edge", "soft_edge", "rest_diff", "new_team_diff", "market_log_prob")
NEW_TEAM_MATCHES = 10
LATEST = datetime.max.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class AnchoredRow:
    match_id: str
    season: str
    match_date: date
    actual: int
    market: np.ndarray
    features: np.ndarray  # shape (3 outcomes, len(FEATURES))
    best_odds: np.ndarray  # best bettable effective price per outcome (nan when none)
    closing: np.ndarray | None


@dataclass(frozen=True, slots=True)
class AnchoredReport:
    seasons: list[dict[str, float | str]]
    pooled: dict[str, float]
    weights: dict[str, dict[str, float]] = field(default_factory=dict)


def _devig_vector(quotes: Mapping[str, Mapping[str, float]], policy: str) -> np.ndarray | None:
    fair = fair_price(quotes, OUTCOMES, policy)
    return np.array([fair.probabilities[o] for o in OUTCOMES]) if fair else None


def _new_team_flags(matches: list[Match]) -> dict[str, tuple[bool, bool]]:
    """Whether each side had fewer than 10 matches in this competition in the previous year."""
    history: dict[tuple[str, str], list[date]] = defaultdict(list)
    flags: dict[str, tuple[bool, bool]] = {}
    for m in sorted(matches, key=lambda m: (m.date, m.match_id)):
        cutoff = m.date - timedelta(days=365)

        def is_new(team: str, m: Match = m, cutoff: date = cutoff) -> bool:
            return sum(1 for d in history[(team, m.competition)] if cutoff <= d < m.date) < NEW_TEAM_MATCHES

        flags[m.match_id] = (is_new(m.home_team), is_new(m.away_team))
        history[(m.home_team, m.competition)].append(m.date)
        history[(m.away_team, m.competition)].append(m.date)
    return flags


def build_rows(
    matches: list[Match],
    odds: Iterable[OddsSnapshot],
    model_probabilities: Mapping[str, tuple[float, float, float]],
    *,
    soft_book: str = "Bet365",
    bettable: tuple[str, ...] | None = None,
) -> list[AnchoredRow]:
    """One row per completed match with a Pinnacle pre-match price and a model forecast."""
    completed = [m for m in matches if m.home_goals is not None and m.away_goals is not None]
    rest = build_pre_match_features(completed)
    new_team = _new_team_flags(completed)
    by_match: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for row in odds:
        if row.market == "1x2":
            by_match[row.match_id].append(row)
    rows = []
    for m in completed:
        model = model_probabilities.get(m.match_id)
        snapshots = by_match.get(m.match_id)
        if model is None or not snapshots:
            continue
        pre = latest_quotes([s for s in snapshots if not s.is_closing], LATEST).get("1x2", {})
        market = _devig_vector(pre, "pinnacle")
        if market is None:
            continue
        soft_prices = pre.get(soft_book, {})
        soft = (
            np.array(list(de_vig({o: 1.0 / soft_prices[o] for o in OUTCOMES}).values()))
            if all(o in soft_prices for o in OUTCOMES)
            else None
        )
        closing_quotes: dict[str, dict[str, float]] = defaultdict(dict)
        for s in snapshots:
            if s.is_closing:
                closing_quotes[s.bookmaker][s.selection] = s.odds
        closing = _devig_vector(closing_quotes, "auto") if closing_quotes else None
        best = np.full(3, np.nan)
        for book, prices in pre.items():
            if book == "Pinnacle" or book in PSEUDO_BOOKMAKERS or (bettable and book not in bettable):
                continue
            for k, outcome in enumerate(OUTCOMES):
                if outcome in prices:
                    price = 1.0 + (prices[outcome] - 1.0) * (1.0 - commission_for(book))
                    best[k] = price if np.isnan(best[k]) else max(best[k], price)
        log_market = np.log(market)
        model_vec = np.clip(np.asarray(model, dtype=float), 1e-6, 1.0)
        soft_edge = np.log(soft) - log_market if soft is not None else np.zeros(3)
        features = rest.get(m.match_id, {})
        rest_gap = features.get("home_rest_days", 7.0) - features.get("away_rest_days", 7.0)
        rest_diff = max(-7.0, min(7.0, rest_gap)) / 7
        home_new, away_new = new_team.get(m.match_id, (False, False))
        side = np.array([1.0, 0.0, -1.0])
        x = np.column_stack(
            [
                np.log(model_vec / model_vec.sum()) - log_market,
                soft_edge,
                side * rest_diff,
                side * (float(home_new) - float(away_new)),
                log_market,
            ]
        )
        actual = 0 if m.home_goals > m.away_goals else 2 if m.home_goals < m.away_goals else 1  # type: ignore[operator]
        rows.append(AnchoredRow(m.match_id, m.season, m.date, actual, market, x, best, closing))
    return rows


def _probabilities(rows: list[AnchoredRow], weights: np.ndarray) -> np.ndarray:
    scores = np.stack([np.log(r.market) + r.features @ weights for r in rows])
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(axis=1, keepdims=True)


def fit_weights(rows: list[AnchoredRow], *, l2: float = 1.0) -> np.ndarray:
    """L2-penalized multinomial log-likelihood with the market as offset."""
    log_market = np.stack([np.log(r.market) for r in rows])
    x = np.stack([r.features for r in rows])  # (n, 3, f)
    y = np.array([r.actual for r in rows])
    n = len(rows)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        scores = log_market + x @ w
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(p[np.arange(n), y])) + 0.5 * l2 * float(w @ w) / n
        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1.0
        grad = np.einsum("nk,nkf->f", p - onehot, x) / n + l2 * w / n
        return float(nll), grad

    return minimize(objective, np.zeros(x.shape[2]), jac=True, method="L-BFGS-B").x


def _log_losses(p: np.ndarray, rows: list[AnchoredRow]) -> np.ndarray:
    y = np.array([r.actual for r in rows])
    return -np.log(np.maximum(p[np.arange(len(rows)), y], 1e-15))


def _betting(p: np.ndarray, rows: list[AnchoredRow], min_ev: float) -> dict[str, float]:
    profits, clvs = [], []
    for probs, row in zip(p, rows, strict=True):
        ev = row.best_odds * probs - 1.0
        if np.all(np.isnan(ev)) or np.nanmax(ev) < min_ev:
            continue
        k = int(np.nanargmax(ev))
        odds = row.best_odds[k]
        profits.append(odds - 1.0 if row.actual == k else -1.0)
        if row.closing is not None:
            clvs.append(odds * row.closing[k] - 1.0)

    def mean_se(values: list[float]) -> tuple[float, float]:
        if len(values) < 2:
            return (values[0] if values else 0.0), 0.0
        arr = np.asarray(values)
        return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))

    y_mean, y_se = mean_se(profits)
    c_mean, c_se = mean_se(clvs)
    return {
        "bets": float(len(profits)),
        "yield": y_mean,
        "yield_se": y_se,
        "mean_clv": c_mean,
        "clv_se": c_se,
    }


def evaluate_anchored(
    rows: list[AnchoredRow], *, l2: float = 1.0, min_ev: float = 0.02, min_train_seasons: int = 2
) -> AnchoredReport:
    """Walk-forward by season: fit on all earlier seasons, score the next one."""
    seasons = sorted({r.season for r in rows})
    season_rows: list[dict[str, float | str]] = []
    weights: dict[str, dict[str, float]] = {}
    market_ll, anchored_ll = [], []
    market_bets, anchored_bets = [], []
    for i, season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = [r for r in rows if r.season < season]
        test = [r for r in rows if r.season == season]
        if not train or not test:
            continue
        w = fit_weights(train, l2=l2)
        weights[season] = dict(zip(FEATURES, w.round(4).tolist(), strict=True))
        p_market = np.stack([r.market for r in test])
        p_anchor = _probabilities(test, w)
        ll_m, ll_a = _log_losses(p_market, test), _log_losses(p_anchor, test)
        market_ll.append(ll_m)
        anchored_ll.append(ll_a)
        market_bets.append((p_market, test))
        anchored_bets.append((p_anchor, test))
        diff = ll_a - ll_m
        season_rows.append(
            {
                "season": season,
                "n": float(len(test)),
                "market_log_loss": float(ll_m.mean()),
                "anchored_log_loss": float(ll_a.mean()),
                "improvement": float(-diff.mean()),
                "improvement_se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
            }
        )
    pooled: dict[str, float] = {}
    if market_ll:
        diff = np.concatenate(anchored_ll) - np.concatenate(market_ll)
        pooled = {
            "n": float(len(diff)),
            "market_log_loss": float(np.concatenate(market_ll).mean()),
            "anchored_log_loss": float(np.concatenate(anchored_ll).mean()),
            "improvement": float(-diff.mean()),
            "improvement_se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
        }
        p_m = np.concatenate([p for p, _ in market_bets])
        p_a = np.concatenate([p for p, _ in anchored_bets])
        test_rows = [r for _, rs in market_bets for r in rs]
        for prefix, p in (("market", p_m), ("anchored", p_a)):
            for key, value in _betting(p, test_rows, min_ev).items():
                pooled[f"{prefix}_{key}"] = value
    return AnchoredReport(seasons=season_rows, pooled=pooled, weights=weights)
