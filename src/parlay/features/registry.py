"""Feature values with provenance: where they came from and when they became knowable."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class FeatureValue:
    value: float
    source: str  # e.g. "football-data E0", "sportmonks lineups"
    computed_at: datetime
    available_at: datetime  # when the value became knowable


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Feature values for one match, grouped by family (``historical``, ``external``, ...)."""

    match_id: str
    as_of: datetime
    groups: dict[str, dict[str, FeatureValue]]

    def is_knowable(self, forecast_timestamp: datetime) -> bool:
        """Whether every feature was computed and available by the forecast cutoff."""
        cutoff = (
            forecast_timestamp
            if forecast_timestamp.tzinfo
            else forecast_timestamp.replace(tzinfo=timezone.utc)
        )
        return all(
            value.computed_at <= cutoff and value.available_at <= cutoff
            for group in self.groups.values()
            for value in group.values()
        )
