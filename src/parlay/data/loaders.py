"""Loader for football-data.co.uk CSV files (results, match stats and odds)."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Iterable
from datetime import date, datetime, time, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .normalization import TeamRegistry
from .odds_columns import row_odds
from .schemas import Match, OddsSnapshot
from .teams import canonical_team_name
from .validation import validate_matches

logger = logging.getLogger(__name__)


def parse_date(value: str) -> date:
    value = value.strip()
    for format_ in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, format_).date()
        except ValueError:
            continue
        return parsed.replace(year=parsed.year + 100) if parsed.year < 1970 else parsed
    raise ValueError(f"Unsupported match date: {value!r} (expected DD/MM/YYYY, DD/MM/YY or YYYY-MM-DD)")


try:
    UK_TIME: tzinfo = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:  # pragma: no cover - systems without a tz database
    UK_TIME = timezone.utc


def _parse_kickoff(match_date: date, value: str) -> datetime | None:
    """football-data kickoff times are UK local time; return them in UTC."""
    value = value.strip()
    if not value:
        return None
    for format_ in ("%H:%M", "%H:%M:%S"):
        try:
            local = datetime.combine(match_date, datetime.strptime(value, format_).time(), tzinfo=UK_TIME)
        except ValueError:
            continue
        return local.astimezone(timezone.utc)
    raise ValueError(f"Unsupported kickoff time: {value!r}")


def _integer(value: str | None, field: str, *, required: bool = True) -> int | None:
    value = (value or "").strip()
    if not value:
        if required:
            raise ValueError(f"Missing required field: {field}")
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid integer in {field}: {value!r}") from exc


def season_for(match_date: date) -> str:
    """European season label (July-June), e.g. 2025-08-16 → ``2025-26``."""
    start = match_date.year if match_date.month >= 7 else match_date.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def match_id_for(match_date: date, home_team: str, away_team: str, competition: str) -> str:
    """Stable ID from date, canonical teams and competition (independent of file and season label)."""
    raw = "|".join(
        [match_date.isoformat(), home_team.casefold(), away_team.casefold(), competition.casefold()]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def load_football_data_csv(
    path: str | Path,
    registry: TeamRegistry | None = None,
    competition: str = "unknown",
    season: str | None = None,
) -> tuple[list[Match], list[OddsSnapshot]]:
    """Load completed matches, match statistics and every recognized bookmaker's odds.

    Pre-match prices are stamped at 00:00 UTC of the match day (football-data
    collects them before the round); closing prices at kickoff.

    Team names are canonicalized (through ``registry`` when given) before
    ``Match`` objects are built. ``season=None`` derives the label from each
    match date. Malformed prices are skipped (and counted in a log warning)
    rather than aborting the whole file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        matches: list[Match] = []
        odds: list[OddsSnapshot] = []
        skipped_prices = 0
        for row in reader:
            if not (row.get("Date") or "").strip() or not (row.get("HomeTeam") or "").strip():
                continue  # football-data files sometimes end with blank rows
            name = registry.canonical_name if registry is not None else canonical_team_name
            home, away = name(row["HomeTeam"]), name(row["AwayTeam"])
            match_date = parse_date(row["Date"])
            kickoff_at = _parse_kickoff(match_date, row.get("Time", "") or "")
            match_id = match_id_for(match_date, home, away, competition)
            matches.append(
                Match(
                    match_id=match_id,
                    date=match_date,
                    competition=competition,
                    season=season or season_for(match_date),
                    home_team=home,
                    away_team=away,
                    home_goals=_integer(row["FTHG"], "FTHG"),
                    away_goals=_integer(row["FTAG"], "FTAG"),
                    source=path.name,
                    kickoff_at=kickoff_at,
                    home_sot=_integer(row.get("HST"), "HST", required=False),
                    away_sot=_integer(row.get("AST"), "AST", required=False),
                    home_corners=_integer(row.get("HC"), "HC", required=False),
                    away_corners=_integer(row.get("AC"), "AC", required=False),
                    home_shots=_integer(row.get("HS"), "HS", required=False),
                    away_shots=_integer(row.get("AS"), "AS", required=False),
                )
            )
            pre_match_at = datetime.combine(match_date, time(0, 0), tzinfo=timezone.utc)
            snapshots, skipped = row_odds(
                row,
                match_id,
                pre_match_at=pre_match_at,
                closing_at=kickoff_at or pre_match_at.replace(hour=12),
            )
            odds.extend(snapshots)
            skipped_prices += skipped
    if skipped_prices:
        logger.warning("%s: skipped %d incomplete or malformed odds groups", path.name, skipped_prices)
    return validate_matches(matches), odds


def load_many(
    paths: Iterable[str | Path],
    registry: TeamRegistry | None = None,
    competition: str = "unknown",
    season: str | None = None,
) -> tuple[list[Match], list[OddsSnapshot]]:
    matches: list[Match] = []
    odds: list[OddsSnapshot] = []
    for path in paths:
        loaded_matches, loaded_odds = load_football_data_csv(path, registry, competition, season)
        matches.extend(loaded_matches)
        odds.extend(loaded_odds)
    return matches, odds


def load_football_data_fixtures(
    path: str | Path,
    *,
    captured_at: datetime,
    leagues: dict[str, str] | None = None,
) -> tuple[list[Match], list[OddsSnapshot]]:
    """Upcoming matches and their prices from football-data's ``fixtures.csv``.

    ``leagues`` maps division codes (``Div`` column) to competition labels;
    other divisions are skipped. Prices are stamped with ``captured_at`` (the
    download time), since the file carries no capture timestamp of its own.
    """
    from .sources import FOOTBALL_DATA_LEAGUES

    leagues = leagues or FOOTBALL_DATA_LEAGUES
    matches: list[Match] = []
    odds: list[OddsSnapshot] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            competition = leagues.get((row.get("Div") or "").strip())
            if competition is None or not (row.get("HomeTeam") or "").strip():
                continue
            home, away = canonical_team_name(row["HomeTeam"]), canonical_team_name(row["AwayTeam"])
            match_date = parse_date(row["Date"])
            kickoff_at = _parse_kickoff(match_date, row.get("Time", "") or "")
            match_id = match_id_for(match_date, home, away, competition)
            matches.append(
                Match(
                    match_id=match_id,
                    date=match_date,
                    competition=competition,
                    season=season_for(match_date),
                    home_team=home,
                    away_team=away,
                    home_goals=None,
                    away_goals=None,
                    source="football-data fixtures",
                    kickoff_at=kickoff_at,
                )
            )
            snapshots, _ = row_odds(row, match_id, pre_match_at=captured_at, closing_at=captured_at)
            odds.extend(s for s in snapshots if not s.is_closing)
    return validate_matches(matches, require_result=False), odds
