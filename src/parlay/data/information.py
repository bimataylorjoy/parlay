"""Canonical information availability abstraction.

Centralizes the answer to: was this information knowable before `as_of`?

Distinguishes:
- forecast_timestamp  (when we make the prediction)
- kickoff_at          (when the fixture starts)
- known_at            (when the result becomes available: kickoff + duration)

Per-competition result lag preserves correctness for EPL vs Championship
(115m vs 118m by default, ~90m + added time).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date, time
from typing import Iterable

from .schemas import Match


# Per-competition conservative result availability lag
RESULT_LAG_MINUTES: dict[str, int] = {
    "EPL": 115,
    "Championship": 118,
    "default": 120,
}

FORECAST_LEAD_MINUTES_DEFAULT = 60  # predict 60m before kickoff


@dataclass(frozen=True, slots=True)
class InformationSet:
    """Defines what information was available at prediction time."""

    as_of: datetime  # forecast_timestamp, must be timezone-aware
    competition: str | None = None
    result_lag_minutes: int | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("InformationSet.as_of must be timezone-aware")

    @property
    def effective_lag(self) -> int:
        if self.result_lag_minutes is not None:
            return self.result_lag_minutes
        if self.competition is not None and self.competition in RESULT_LAG_MINUTES:
            return RESULT_LAG_MINUTES[self.competition]
        return RESULT_LAG_MINUTES["default"]

    def result_known_at(self, match: Match) -> datetime:
        """When the match result becomes knowable (conservative)."""
        if match.kickoff_at is not None:
            base = match.kickoff_at
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return base + timedelta(minutes=self.effective_lag)
        # Fallback when kickoff time unavailable: end of match day
        return datetime.combine(match.date, time(23, 59), tzinfo=timezone.utc) + timedelta(minutes=self.effective_lag)

    def is_result_knowable(self, match: Match) -> bool:
        return self.result_known_at(match) <= self.as_of

    def filter_knowable(self, matches: Iterable[Match]) -> list[Match]:
        return [m for m in matches if self.is_result_knowable(m)]

    def is_feature_knowable(self, computed_at: datetime, available_at: datetime) -> bool:
        # Feature must have been computed and its availability <= forecast
        ts = computed_at if computed_at.tzinfo else computed_at.replace(tzinfo=timezone.utc)
        avail = available_at if available_at.tzinfo else available_at.replace(tzinfo=timezone.utc)
        return ts <= self.as_of and avail <= self.as_of


def forecast_timestamp_for_match(match: Match, lead_minutes: int = FORECAST_LEAD_MINUTES_DEFAULT) -> datetime:
    if match.kickoff_at is not None:
        ts = match.kickoff_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts - timedelta(minutes=lead_minutes)
    return datetime.combine(match.date, time(0, 0), tzinfo=timezone.utc)
