"""Timestamp-safe external match information and market movement contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean


@dataclass(frozen=True, slots=True)
class TeamNews:
    team: str
    category: str
    status: str
    source: str
    published_at: datetime
    available_at: datetime
    reliability: float = 1.0
    note: str | None = None

    def is_available_at(self, as_of: datetime) -> bool:
        cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        available = self.available_at if self.available_at.tzinfo else self.available_at.replace(tzinfo=timezone.utc)
        return available <= cutoff


@dataclass(frozen=True, slots=True)
class LineupStatus:
    match_id: str
    team: str
    status: str
    source: str
    published_at: datetime
    available_at: datetime
    player_ids: tuple[str, ...] = ()

    def is_available_at(self, as_of: datetime) -> bool:
        cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        available = self.available_at if self.available_at.tzinfo else self.available_at.replace(tzinfo=timezone.utc)
        return available <= cutoff


@dataclass(frozen=True, slots=True)
class InjuryReport:
    team: str
    player_id: str
    status: str
    source: str
    published_at: datetime
    available_at: datetime
    expected_return: str | None = None

    def is_available_at(self, as_of: datetime) -> bool:
        cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        available = self.available_at if self.available_at.tzinfo else self.available_at.replace(tzinfo=timezone.utc)
        return available <= cutoff


def available_team_information(items, as_of: datetime) -> list[object]:
    """Filter news, lineups, and injuries by declared availability timestamp."""
    return [item for item in items if item.is_available_at(as_of)]


@dataclass(frozen=True, slots=True)
class OddsMovementSummary:
    bookmaker: str
    market: str
    selection: str
    first_odds: float
    latest_odds: float
    captured_at: datetime
    observations: int
    log_return: float
    implied_probability_change: float


def summarize_odds_movement(snapshots, *, as_of: datetime | None = None) -> list[OddsMovementSummary]:
    """Summarize observed price movement without treating it as fair probability.

    The output is descriptive only. It does not infer unobserved order flow or
    claim that a price move represents information rather than margin/liquidity.
    """
    cutoff = as_of
    rows = [row for row in snapshots if cutoff is None or row.captured_at <= cutoff]
    groups = {}
    for row in rows:
        groups.setdefault((row.bookmaker, row.market, row.selection), []).append(row)
    output = []
    for (bookmaker, market, selection), values in groups.items():
        values.sort(key=lambda row: row.captured_at)
        first, latest = values[0], values[-1]
        output.append(OddsMovementSummary(
            bookmaker=bookmaker,
            market=market,
            selection=selection,
            first_odds=float(first.odds),
            latest_odds=float(latest.odds),
            captured_at=latest.captured_at,
            observations=len(values),
            log_return=__import__("math").log(latest.odds / first.odds),
            implied_probability_change=(1.0 / latest.odds) - (1.0 / first.odds),
        ))
    return output
