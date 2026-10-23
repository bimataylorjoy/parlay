"""Find value bets: prices above the fair price implied by a sharp reference.

For every upcoming match and market, the latest pre-match snapshot of each
bookmaker is collected, a fair price is derived from the reference
(:mod:`parlay.betting.reference`), and the best price among the bookmakers
you can actually use is compared with it::

    EV = effective_odds × p_fair − 1

For markets with a void outcome (integer and quarter lines) both ``p_fair``
and EV are conditional on the bet not being void. Exchange prices are
reduced by the commission. Stakes are fractional Kelly with a shrinkage for
the uncertainty of ``p_fair`` (Baker & McHale, 2013).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from parlay.data.odds_columns import EXCHANGE, MARKET_AVG, MARKET_MAX, PSEUDO_BOOKMAKERS, parse_market
from parlay.data.schemas import Match, OddsSnapshot
from parlay.prediction.markets import DE_VIG_METHODS

from .reference import REFERENCE_POLICIES, FairPrice, fair_price
from .settlement import SELECTIONS, selections_for

MARKET_KINDS = tuple(SELECTIONS)


@dataclass(frozen=True, slots=True)
class ValueConfig:
    reference: str = "auto"
    bookmakers: tuple[str, ...] | None = None  # where you can bet; None = every real bookmaker
    markets: tuple[str, ...] = MARKET_KINDS
    min_ev: float = 0.02
    max_ev: float = 0.30  # larger "edges" are almost always stale or erroneous prices
    min_odds: float = 1.15
    max_odds: float = 10.0
    exchange_commission: float = 0.05
    kelly_fraction: float = 0.25
    max_stake: float = 0.02
    probability_sd: float = 0.02
    de_vig_method: str = "shin"
    include_market_max: bool = False  # football-data's best price is not a single bettable book

    def __post_init__(self) -> None:
        if self.reference not in REFERENCE_POLICIES:
            raise ValueError(f"reference must be one of {REFERENCE_POLICIES}")
        if self.de_vig_method not in DE_VIG_METHODS:
            raise ValueError(f"de_vig_method must be one of {DE_VIG_METHODS}")
        unknown = set(self.markets) - set(MARKET_KINDS)
        if unknown:
            raise ValueError(f"unknown market kinds {sorted(unknown)}; expected {MARKET_KINDS}")
        if not 0 <= self.exchange_commission < 1 or not 0 <= self.kelly_fraction <= 1:
            raise ValueError("exchange_commission must be in [0, 1) and kelly_fraction in [0, 1]")
        if not 0 < self.max_stake <= 1 or self.probability_sd < 0:
            raise ValueError("max_stake must be in (0, 1] and probability_sd non-negative")


@dataclass(frozen=True, slots=True)
class ValueBet:
    match_id: str
    competition: str
    match_date: date
    kickoff_at: datetime | None
    home_team: str
    away_team: str
    market: str
    selection: str
    bookmaker: str
    odds: float
    effective_odds: float
    fair_probability: float
    expected_value: float
    stake_fraction: float
    reference: str
    reference_books: int
    captured_at: datetime

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.fair_probability

    @property
    def key(self) -> str:
        raw = (
            f"{self.match_id}|{self.market}|{self.selection}|{self.bookmaker}|{self.odds}|{self.captured_at}"
        )
        return "vb:" + hashlib.sha1(raw.encode()).hexdigest()[:8]

    @property
    def description(self) -> str:
        kind, line = parse_market(self.market)
        if kind == "1x2":
            pick = {"home": self.home_team, "away": self.away_team, "draw": "Draw"}[self.selection]
        elif kind == "totals":
            pick = f"{self.selection.capitalize()} {line:g}"
        else:
            assert line is not None
            team_line = line if self.selection == "home" else -line
            team = self.home_team if self.selection == "home" else self.away_team
            pick = f"{team} {team_line + 0.0:+g}"
        return f"{self.home_team} vs {self.away_team}: {pick}"


def effective_odds(odds: float, bookmaker: str, commission: float) -> float:
    """Exchange winnings are reduced by the commission."""
    return 1.0 + (odds - 1.0) * (1.0 - commission) if bookmaker == EXCHANGE else odds


def shrunk_kelly(
    probability: float, odds: float, *, fraction: float, max_stake: float, probability_sd: float
) -> float:
    """Fractional Kelly with shrinkage for an uncertain probability.

    With ``f = (p·o − 1)/(o − 1)`` and ``Var(f) ≈ (sd · o/(o − 1))²`` the
    growth-optimal multiplier is ``f² / (f² + Var(f))`` (Baker & McHale, 2013).
    """
    edge = probability * odds - 1.0
    if edge <= 0:
        return 0.0
    full = edge / (odds - 1.0)
    variance = (probability_sd * odds / (odds - 1.0)) ** 2
    shrink = full**2 / (full**2 + variance) if variance > 0 else 1.0
    return min(full * shrink * fraction, max_stake)


def latest_quotes(
    snapshots: Iterable[OddsSnapshot], as_of: datetime
) -> dict[str, dict[str, dict[str, float]]]:
    """``market -> bookmaker -> selection -> odds`` from each bookmaker's latest pre-match snapshot."""
    latest: dict[tuple[str, str], datetime] = {}
    rows: list[OddsSnapshot] = []
    for row in snapshots:
        if row.is_closing or row.captured_at > as_of:
            continue
        rows.append(row)
        key = (row.market, row.bookmaker)
        if key not in latest or row.captured_at > latest[key]:
            latest[key] = row.captured_at
    quotes: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.captured_at == latest[(row.market, row.bookmaker)]:
            quotes[row.market][row.bookmaker][row.selection] = row.odds
    return quotes


def default_forecast_time(match: Match, lead_minutes: int = 60) -> datetime:
    if match.kickoff_at is not None:
        kickoff = (
            match.kickoff_at if match.kickoff_at.tzinfo else match.kickoff_at.replace(tzinfo=timezone.utc)
        )
        return kickoff - timedelta(minutes=lead_minutes)
    return datetime.combine(match.date, time(0, 0), tzinfo=timezone.utc)


def _bettable(book: str, config: ValueConfig) -> bool:
    if book == MARKET_AVG:
        return False
    if book == MARKET_MAX:
        return config.include_market_max
    return config.bookmakers is None or book in config.bookmakers


def market_value_bets(
    match: Match,
    market: str,
    quotes: Mapping[str, Mapping[str, float]],
    captured_at: datetime,
    config: ValueConfig,
) -> list[ValueBet]:
    """Value bets (best qualifying price per selection) in one market of one match."""
    try:
        selections = selections_for(market)
    except ValueError:
        return []
    fair: FairPrice | None = fair_price(quotes, selections, config.reference, method=config.de_vig_method)
    if fair is None:
        return []
    bets = []
    for selection in selections:
        offers = [
            (effective_odds(prices[selection], book, config.exchange_commission), prices[selection], book)
            for book, prices in quotes.items()
            if selection in prices and _bettable(book, config) and book != fair.source
        ]
        if not offers:
            continue
        effective, odds, book = max(offers)
        p = fair.probabilities[selection]
        ev = effective * p - 1.0
        if not (config.min_ev <= ev <= config.max_ev and config.min_odds <= odds <= config.max_odds):
            continue
        bets.append(
            ValueBet(
                match_id=match.match_id,
                competition=match.competition,
                match_date=match.date,
                kickoff_at=match.kickoff_at,
                home_team=match.home_team,
                away_team=match.away_team,
                market=market,
                selection=selection,
                bookmaker=book,
                odds=odds,
                effective_odds=effective,
                fair_probability=p,
                expected_value=ev,
                stake_fraction=shrunk_kelly(
                    p,
                    effective,
                    fraction=config.kelly_fraction,
                    max_stake=config.max_stake,
                    probability_sd=config.probability_sd,
                ),
                reference=fair.source,
                reference_books=fair.books,
                captured_at=captured_at,
            )
        )
    return bets


def find_value_bets(
    matches: Iterable[Match],
    odds: Iterable[OddsSnapshot],
    config: ValueConfig | None = None,
    *,
    as_of: datetime | None = None,
) -> list[ValueBet]:
    """Value bets for ``matches`` using prices captured no later than ``as_of``.

    ``as_of=None`` uses each match's forecast time (kickoff − 60 min), which
    is what historical backtests need; live use passes the current time.
    """
    config = config or ValueConfig()
    by_match: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for row in odds:
        by_match[row.match_id].append(row)
    bets: list[ValueBet] = []
    for match in matches:
        rows = by_match.get(match.match_id)
        if not rows:
            continue
        cutoff = as_of or default_forecast_time(match)
        captured = max(
            (r.captured_at for r in rows if not r.is_closing and r.captured_at <= cutoff), default=None
        )
        if captured is None:
            continue
        for market, quotes in latest_quotes(rows, cutoff).items():
            if parse_market(market)[0] in config.markets:
                bets.extend(market_value_bets(match, market, quotes, captured, config))
    bets.sort(key=lambda bet: bet.expected_value, reverse=True)
    return bets


def real_bookmakers(odds: Iterable[OddsSnapshot]) -> list[str]:
    return sorted({row.bookmaker for row in odds} - PSEUDO_BOOKMAKERS)
