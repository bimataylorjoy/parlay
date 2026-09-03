"""Small, dependency-light contracts for time-safe football data."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class Match:
    match_id: str
    date: date
    competition: str
    season: str
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    neutral: bool = False
    source: str = "unknown"
    source_timestamp: Optional[datetime] = None
    kickoff_at: Optional[datetime] = None
    home_sot: Optional[int] = None
    away_sot: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None


@dataclass(frozen=True, slots=True)
class OddsSnapshot:
    match_id: str
    bookmaker: str
    market: str
    selection: str
    odds: float
    captured_at: datetime
    is_closing: bool = False


@dataclass(frozen=True, slots=True)
class PlayerMatchStat:
    fixture_id: str
    player_id: str
    team_id: str
    opponent_id: str | None
    position: str | None
    started: bool
    minutes: float
    goals: float
    assists: float
    shots: float
    shots_on_target: float
    key_passes: float
    tackles: float
    interceptions: float
    clearances: float
    blocks: float
    errors: float
    xg: float | None
    xa: float | None
    rating: float | None
    observed_at: datetime
    available_at: datetime
    source: str
