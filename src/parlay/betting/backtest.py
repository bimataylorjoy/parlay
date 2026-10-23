"""Historical test of the value-betting engine on football-data prices.

The same :func:`find_value_bets` used live is run on every completed match
with the pre-match snapshot only; bets are settled on the result and scored
against the closing line. Two staking views are reported: flat one-unit bets
(yield) and compounding fractional Kelly (bankroll growth).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from parlay.data.odds_columns import parse_market
from parlay.data.schemas import Match, OddsSnapshot

from .ledger import _mean_se
from .reference import fair_price
from .settlement import selections_for, settle
from .value import ValueBet, ValueConfig, find_value_bets


@dataclass(frozen=True, slots=True)
class SettledValueBet:
    bet: ValueBet
    result: str
    flat_profit: float
    closing_probability: float | None
    closing_source: str | None

    @property
    def clv(self) -> float | None:
        if self.closing_probability is None:
            return None
        return self.bet.effective_odds * self.closing_probability - 1.0


@dataclass(frozen=True, slots=True)
class ValueBacktestResult:
    bets: list[SettledValueBet]
    metrics: dict[str, float]
    breakdown: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _closing(rows: list[OddsSnapshot], market: str, method: str) -> tuple[dict[str, float], str] | None:
    quotes: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.is_closing and row.market == market:
            quotes[row.bookmaker][row.selection] = row.odds
    fair = fair_price(quotes, selections_for(market), "auto", method=method) if quotes else None
    return (fair.probabilities, fair.source) if fair else None


def _summary(settled: list[SettledValueBet]) -> dict[str, float]:
    if not settled:
        return {"bets": 0.0}
    profits = [s.flat_profit for s in settled]
    yield_mean, yield_se = _mean_se(profits)
    clvs = [s.clv for s in settled if s.clv is not None]
    clv_mean, clv_se = _mean_se(clvs)
    return {
        "bets": float(len(settled)),
        "mean_odds": sum(s.bet.odds for s in settled) / len(settled),
        "mean_expected_value": sum(s.bet.expected_value for s in settled) / len(settled),
        "flat_profit": sum(profits),
        "yield": yield_mean,
        "yield_se": yield_se,
        "clv_n": float(len(clvs)),
        "mean_clv": clv_mean,
        "clv_se": clv_se,
    }


def run_value_backtest(
    matches: list[Match],
    odds: list[OddsSnapshot],
    config: ValueConfig | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    one_bet_per_market: bool = True,
) -> ValueBacktestResult:
    """Replay the value engine over completed matches between ``start`` and ``end``."""
    config = config or ValueConfig()
    completed = sorted(
        (
            m
            for m in matches
            if m.home_goals is not None
            and m.away_goals is not None
            and (start is None or m.date >= start)
            and (end is None or m.date <= end)
        ),
        key=lambda m: (m.date, m.match_id),
    )
    by_match: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for row in odds:
        by_match[row.match_id].append(row)
    lookup = {m.match_id: m for m in completed}

    settled: list[SettledValueBet] = []
    for match in completed:
        rows = by_match.get(match.match_id, [])
        bets = find_value_bets([match], rows, config)
        if one_bet_per_market:
            best: dict[str, ValueBet] = {}
            for bet in bets:
                if bet.market not in best or bet.expected_value > best[bet.market].expected_value:
                    best[bet.market] = bet
            bets = list(best.values())
        for bet in bets:
            outcome = settle(bet.market, bet.selection, match.home_goals, match.away_goals)  # type: ignore[arg-type]
            closing = _closing(rows, bet.market, config.de_vig_method)
            settled.append(
                SettledValueBet(
                    bet=bet,
                    result=outcome.label,
                    flat_profit=outcome.profit(1.0, bet.effective_odds),
                    closing_probability=closing[0][bet.selection] if closing else None,
                    closing_source=closing[1] if closing else None,
                )
            )

    metrics = _summary(settled)
    # Compounding fractional Kelly in date order; same-day bets are sized on the same bankroll.
    log_growth = 0.0
    for day in sorted({s.bet.match_date for s in settled}):
        today = [s for s in settled if s.bet.match_date == day]
        change = sum(s.bet.stake_fraction * s.flat_profit for s in today)
        log_growth += math.log(max(1e-9, 1.0 + change))
    if settled:
        metrics["kelly_bankroll_multiple"] = math.exp(log_growth)
        metrics["matches"] = float(len(completed))

    groupings: dict[str, Callable[[SettledValueBet], str]] = {
        "market": lambda s: parse_market(s.bet.market)[0],
        "reference": lambda s: s.bet.reference,
        "bookmaker": lambda s: s.bet.bookmaker,
        "competition": lambda s: lookup[s.bet.match_id].competition,
        "season": lambda s: lookup[s.bet.match_id].season,
    }
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for name, key in groupings.items():
        groups: dict[str, list[SettledValueBet]] = defaultdict(list)
        for s in settled:
            groups[key(s)].append(s)
        breakdown[name] = {label: _summary(rows) for label, rows in sorted(groups.items())}
    return ValueBacktestResult(bets=settled, metrics=metrics, breakdown=breakdown)
