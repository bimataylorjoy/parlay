"""Validation rules that prevent silent data and leakage errors."""

from collections import Counter
from datetime import date
from typing import Iterable

from .schemas import Match, OddsSnapshot


def validate_matches(matches: Iterable[Match], *, require_result: bool = True) -> list[Match]:
    rows = list(matches)
    errors: list[str] = []
    ids = [row.match_id for row in rows]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate match_id: {duplicates}")
    for row in rows:
        if not row.match_id.strip():
            errors.append("match_id must not be empty")
        if not row.home_team.strip() or not row.away_team.strip():
            errors.append(f"{row.match_id}: team name must not be empty")
        if row.home_team == row.away_team:
            errors.append(f"{row.match_id}: home and away teams are identical")
        if require_result and (row.home_goals is None or row.away_goals is None):
            errors.append(f"{row.match_id}: final score is required")
        if row.home_goals is not None and row.home_goals < 0:
            errors.append(f"{row.match_id}: home goals cannot be negative")
        if row.away_goals is not None and row.away_goals < 0:
            errors.append(f"{row.match_id}: away goals cannot be negative")
        if row.home_corners is not None and row.home_corners < 0:
            errors.append(f"{row.match_id}: home corners cannot be negative")
        if row.away_corners is not None and row.away_corners < 0:
            errors.append(f"{row.match_id}: away corners cannot be negative")
        if row.home_shots is not None and row.home_shots < 0:
            errors.append(f"{row.match_id}: home shots cannot be negative")
        if row.away_shots is not None and row.away_shots < 0:
            errors.append(f"{row.match_id}: away shots cannot be negative")
    if errors:
        raise ValueError("Invalid matches:\n- " + "\n- ".join(errors))
    return rows


def validate_odds(odds: Iterable[OddsSnapshot], *, as_of: date | None = None) -> list[OddsSnapshot]:
    rows = list(odds)
    errors: list[str] = []
    for row in rows:
        if row.odds <= 1.0:
            errors.append(f"{row.match_id}: decimal odds must be > 1")
        if as_of is not None and row.captured_at.date() > as_of:
            errors.append(f"{row.match_id}: odds snapshot is after forecast date")
    if errors:
        raise ValueError("Invalid odds:\n- " + "\n- ".join(errors))
    return rows
