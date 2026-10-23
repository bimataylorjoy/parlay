"""Accumulator (mix parlay) tickets priced against sharp fair probabilities.

For legs on *different* matches the outcomes are close to independent, so::

    ticket odds        = Π leg odds × (1 + boost)
    win probability    = Π p_fair
    expected value     = ticket odds × Π p_fair − 1  =  (1 + boost) Π (1 + EV_leg) − 1

The bookmaker margin compounds with every leg: legs with negative EV make
the ticket much worse, and only legs that are value bets on their own
compound into a positive ticket. Legs on the same match are correlated and
cannot be priced from market prices alone, so they are rejected.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

from parlay.data.odds_columns import EXCHANGES
from parlay.data.schemas import Match, OddsSnapshot

from .reference import fair_price
from .settlement import selections_for
from .value import ValueBet, describe_pick, latest_quotes, shrunk_kelly

MAX_LEGS = 20


@dataclass(frozen=True, slots=True)
class Leg:
    match_id: str
    competition: str
    match_date: date
    kickoff_at: datetime | None
    home_team: str
    away_team: str
    market: str
    selection: str
    odds: float
    fair_probability: float | None
    reference: str | None

    @property
    def description(self) -> str:
        return describe_pick(self.home_team, self.away_team, self.market, self.selection)

    @property
    def expected_value(self) -> float | None:
        return None if self.fair_probability is None else self.odds * self.fair_probability - 1.0

    @classmethod
    def from_value_bet(cls, bet: ValueBet) -> Leg:
        return cls(
            bet.match_id,
            bet.competition,
            bet.match_date,
            bet.kickoff_at,
            bet.home_team,
            bet.away_team,
            bet.market,
            bet.selection,
            bet.odds,
            bet.fair_probability,
            bet.reference,
        )


@dataclass(frozen=True, slots=True)
class Ticket:
    bookmaker: str
    legs: tuple[Leg, ...]
    boost: float = 0.0
    ticket_odds: float | None = None  # the price the bookmaker shows, if it differs from the product
    warnings: tuple[str, ...] = field(default=())

    @property
    def combined_odds(self) -> float:
        if self.ticket_odds is not None:
            return self.ticket_odds
        return math.prod(leg.odds for leg in self.legs) * (1.0 + self.boost)

    @property
    def priced(self) -> bool:
        return all(leg.fair_probability is not None for leg in self.legs)

    @property
    def win_probability(self) -> float | None:
        if not self.priced:
            return None
        return math.prod(leg.fair_probability or 0.0 for leg in self.legs)

    @property
    def expected_value(self) -> float | None:
        p = self.win_probability
        return None if p is None else self.combined_odds * p - 1.0

    @property
    def fair_odds(self) -> float | None:
        p = self.win_probability
        return None if not p else 1.0 / p

    def stake_fraction(
        self, *, kelly_fraction: float = 0.25, max_stake: float = 0.02, leg_sd: float = 0.02
    ) -> float:
        """Shrunk fractional Kelly; the product's uncertainty grows with every leg.

        ``sd(Π p) ≈ Π p · sqrt(Σ (sd_leg / p_leg)²)`` (delta method).
        """
        p = self.win_probability
        if p is None:
            return 0.0
        probabilities = [leg.fair_probability for leg in self.legs if leg.fair_probability is not None]
        relative = math.sqrt(sum((leg_sd / q) ** 2 for q in probabilities))
        return shrunk_kelly(
            p, self.combined_odds, fraction=kelly_fraction, max_stake=max_stake, probability_sd=p * relative
        )

    @property
    def key(self) -> str:
        raw = "|".join(sorted(f"{leg.match_id}:{leg.market}:{leg.selection}" for leg in self.legs))
        return "tk:" + hashlib.sha1(f"{self.bookmaker}|{raw}".encode()).hexdigest()[:8]


def validate_legs(legs: Iterable[Leg]) -> tuple[tuple[Leg, ...], list[str]]:
    """Reject duplicate/same-match legs; return the legs and human-readable warnings."""
    legs = tuple(legs)
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two legs")
    if len(legs) > MAX_LEGS:
        raise ValueError(f"at most {MAX_LEGS} legs per ticket")
    seen: dict[str, Leg] = {}
    for leg in legs:
        if leg.match_id in seen:
            raise ValueError(
                f"two legs on {leg.home_team} vs {leg.away_team}: same-match legs are correlated and "
                "cannot be priced from market odds; pick one leg per match"
            )
        seen[leg.match_id] = leg
        selections_for(leg.market)
    warnings = []
    for leg in legs:
        if leg.fair_probability is None:
            warnings.append(f"{leg.description}: no reference price, the ticket EV is unknown")
        elif leg.expected_value is not None and leg.expected_value < 0:
            warnings.append(
                f"{leg.description}: leg EV {leg.expected_value:+.1%} — it drags the whole ticket down"
            )
    return legs, warnings


def build_ticket(
    legs: Iterable[Leg], bookmaker: str, *, boost: float = 0.0, ticket_odds: float | None = None
) -> Ticket:
    if bookmaker in EXCHANGES:
        raise ValueError(f"{bookmaker} is an exchange; exchanges do not offer accumulators")
    if boost < 0 or (ticket_odds is not None and ticket_odds <= 1):
        raise ValueError("boost must be non-negative and ticket odds above 1")
    checked, warnings = validate_legs(legs)
    return Ticket(bookmaker, checked, boost, ticket_odds, tuple(warnings))


def price_leg(
    match: Match,
    market: str,
    selection: str,
    odds: Iterable[OddsSnapshot],
    *,
    bookmaker: str,
    as_of: datetime,
    reference: str = "pinnacle",
    leg_odds: float | None = None,
) -> Leg:
    """A leg priced with the bookmaker's latest odds (or ``leg_odds``) and the reference fair probability."""
    selections = selections_for(market)
    if selection not in selections:
        raise ValueError(f"selection must be one of {selections} for {market}")
    quotes = latest_quotes(odds, as_of).get(market, {})
    fair = fair_price(quotes, selections, reference)
    price = leg_odds or quotes.get(bookmaker, {}).get(selection)
    if price is None:
        pick = describe_pick(match.home_team, match.away_team, market, selection)
        raise ValueError(f"no {bookmaker} price stored for {pick}")
    return Leg(
        match.match_id,
        match.competition,
        match.date,
        match.kickoff_at,
        match.home_team,
        match.away_team,
        market,
        selection,
        float(price),
        fair.probabilities[selection] if fair else None,
        fair.source if fair else None,
    )


def favourite_leg(
    match: Match,
    odds: Iterable[OddsSnapshot],
    *,
    bookmaker: str,
    as_of: datetime,
    reference: str = "pinnacle",
) -> Leg:
    """The 1X2 favourite (lowest bookmaker price) of one match."""
    quotes = latest_quotes(odds, as_of).get("1x2", {}).get(bookmaker, {})
    if not quotes:
        raise ValueError(f"no {bookmaker} 1X2 price for {match.home_team} vs {match.away_team}")
    selection = min(quotes, key=quotes.get)  # type: ignore[arg-type]
    return price_leg(match, "1x2", selection, odds, bookmaker=bookmaker, as_of=as_of, reference=reference)


def value_tickets(
    bets: Iterable[ValueBet], *, max_legs: int = 3, min_legs: int = 2, min_leg_ev: float = 0.0
) -> list[Ticket]:
    """One ticket per bookmaker from its best value bet on each distinct match (highest EV first)."""
    if not 2 <= min_legs <= max_legs <= MAX_LEGS:
        raise ValueError(f"need 2 <= min_legs <= max_legs <= {MAX_LEGS}")
    by_book: dict[str, dict[str, ValueBet]] = defaultdict(dict)
    for bet in bets:
        if bet.bookmaker in EXCHANGES or bet.expected_value < min_leg_ev:
            continue
        current = by_book[bet.bookmaker].get(bet.match_id)
        if current is None or bet.expected_value > current.expected_value:
            by_book[bet.bookmaker][bet.match_id] = bet
    tickets = []
    for book, best in by_book.items():
        chosen = sorted(best.values(), key=lambda b: b.expected_value, reverse=True)[:max_legs]
        if len(chosen) >= min_legs:
            tickets.append(build_ticket([Leg.from_value_bet(b) for b in chosen], book))
    tickets.sort(key=lambda t: t.expected_value or 0.0, reverse=True)
    return tickets
