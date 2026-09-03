"""Per-player rating estimation from timestamped match statistics."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from parlay.data.schemas import PlayerMatchStat


@dataclass(frozen=True, slots=True)
class PlayerRatingEstimate:
    player_id: str
    team_id: str
    position: str | None
    attack_value: float
    defense_value: float
    minutes: float
    observations: int
    reliability: float
    source: str = "player_match_stats_ridge_shrinkage"


def estimate_player_ratings(
    stats: list[PlayerMatchStat], *, as_of=None, prior_strength_minutes: float = 900.0,
) -> list[PlayerRatingEstimate]:
    """Estimate shrunken per-90 contributions using only available history."""
    if prior_strength_minutes <= 0:
        raise ValueError("prior_strength_minutes must be positive")
    rows = [row for row in stats if as_of is None or row.available_at <= as_of]
    groups = defaultdict(list)
    for row in rows:
        groups[(row.player_id, row.team_id)].append(row)
    position_values = defaultdict(lambda: [[], []])
    raw = []
    for (player_id, team_id), values in groups.items():
        minutes = sum(max(0.0, row.minutes) for row in values)
        if minutes <= 0:
            continue
        position = next((row.position for row in values if row.position), None)
        attack = sum((row.goals + row.assists + (row.xg or 0.0) + (row.xa or 0.0)) * 90.0 / max(row.minutes, 1.0) for row in values) / len(values)
        defense = sum((row.tackles + row.interceptions + row.clearances + row.blocks - row.errors) * 90.0 / max(row.minutes, 1.0) for row in values) / len(values)
        raw.append((player_id, team_id, position, attack, defense, minutes, len(values)))
        position_values[position][0].append(attack)
        position_values[position][1].append(defense)
    output = []
    for player_id, team_id, position, attack, defense, minutes, observations in raw:
        prior_attack = sum(position_values[position][0]) / len(position_values[position][0]) if position_values[position][0] else 0.0
        prior_defense = sum(position_values[position][1]) / len(position_values[position][1]) if position_values[position][1] else 0.0
        reliability = minutes / (minutes + prior_strength_minutes)
        output.append(PlayerRatingEstimate(
            player_id=player_id, team_id=team_id, position=position,
            attack_value=reliability * attack + (1.0 - reliability) * prior_attack,
            defense_value=reliability * defense + (1.0 - reliability) * prior_defense,
            minutes=minutes, observations=observations, reliability=reliability,
        ))
    return output
