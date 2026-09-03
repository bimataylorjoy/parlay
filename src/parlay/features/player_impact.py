"""Conservative player availability adjustments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerRating:
    player_id: str
    team: str
    attacking_value: float
    defensive_value: float
    expected_minutes: float = 90.0
    source: str = "historical_player_model"


@dataclass(frozen=True, slots=True)
class AvailabilityAdjustment:
    attack_delta: float
    defense_delta: float
    covered_players: int
    missing_players: int
    status: str


def availability_adjustment(
    ratings: list[PlayerRating], unavailable_player_ids: set[str], *, max_abs_delta: float = 0.35,
) -> AvailabilityAdjustment:
    """Estimate deltas only from supplied historical player ratings."""
    if max_abs_delta <= 0:
        raise ValueError("max_abs_delta must be positive")
    selected = [rating for rating in ratings if rating.player_id in unavailable_player_ids]
    attack = sum(rating.attacking_value * max(0.0, min(rating.expected_minutes, 90.0)) / 90.0 for rating in selected)
    defense = sum(rating.defensive_value * max(0.0, min(rating.expected_minutes, 90.0)) / 90.0 for rating in selected)
    return AvailabilityAdjustment(
        attack_delta=max(-max_abs_delta, min(max_abs_delta, -attack)),
        defense_delta=max(-max_abs_delta, min(max_abs_delta, -defense)),
        covered_players=len(selected),
        missing_players=max(0, len(unavailable_player_ids) - len(selected)),
        status="estimated" if selected else "unavailable",
    )
