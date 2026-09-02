"""Loaders for the common football-data.co.uk CSV format."""

import csv
from datetime import date, datetime, timezone
import hashlib
import re
from pathlib import Path
from typing import Iterable

from .normalization import TeamRegistry
from .schemas import Match, OddsSnapshot
from .validation import validate_matches, validate_odds


def _parse_date(value: str) -> date:
    value = value.strip()
    for format_ in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, format_).date()
            if parsed.year < 1970:
                parsed = parsed.replace(year=parsed.year + 100)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"Unsupported match date: {value!r}")


def _parse_kickoff(match_date: date, value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    for format_ in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, format_).time()
            return datetime.combine(match_date, parsed, tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported kickoff time: {value!r}")


def _integer(value: str, field: str, *, required: bool = True) -> int | None:
    value = value.strip()
    if not value:
        if required:
            raise ValueError(f"Missing required field: {field}")
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid integer in {field}: {value!r}") from exc


def _decimal(value: str, field: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid decimal in {field}: {value!r}") from exc


def _match_id(row: dict[str, str], competition: str, season: str) -> str:
    raw = "|".join([
        row.get("Date", "").strip(), row.get("HomeTeam", "").strip().casefold(),
        row.get("AwayTeam", "").strip().casefold(), competition, season,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def load_football_data_csv(path: str | Path, registry: TeamRegistry | None = None,
                           competition: str = "unknown", season: str = "unknown") -> tuple[list[Match], list[OddsSnapshot]]:
    """Load final scores and recognized 1X2 bookmaker columns.

    Unknown odds columns are ignored. Team aliases are resolved before the
    Match objects are created, preventing provider names from leaking into the
    model layer.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        matches: list[Match] = []
        odds: list[OddsSnapshot] = []
        source = str(path)
        for row in reader:
            home = row["HomeTeam"].strip()
            away = row["AwayTeam"].strip()
            if registry is not None:
                registry.resolve(home)
                registry.resolve(away)
            match_id = _match_id(row, competition, season)
            match_date = _parse_date(row["Date"])
            kickoff_at = _parse_kickoff(match_date, row.get("Time", ""))
            
            home_sot = _integer(row.get("HST", ""), "HST", required=False)
            away_sot = _integer(row.get("AST", ""), "AST", required=False)
            home_corners = _integer(row.get("HC", ""), "HC", required=False)
            away_corners = _integer(row.get("AC", ""), "AC", required=False)
            home_shots = _integer(row.get("HS", ""), "HS", required=False)
            away_shots = _integer(row.get("AS", ""), "AS", required=False)
            
            matches.append(Match(
                match_id, match_date, competition, season, home, away,
                _integer(row["FTHG"], "FTHG"), _integer(row["FTAG"], "FTAG"),
                source=source,
                source_timestamp=datetime.now(timezone.utc),
                kickoff_at=kickoff_at,
                home_sot=home_sot,
                away_sot=away_sot,
                home_corners=home_corners,
                away_corners=away_corners,
                home_shots=home_shots,
                away_shots=away_shots,
            ))
            
            # Base timestamp for standard odds
            captured = datetime.combine(match_date, datetime.min.time(), tzinfo=timezone.utc)
            # Use kickoff_at for closing odds if available, otherwise add 12 hours to base
            closing_time = kickoff_at if kickoff_at else captured.replace(hour=12)
            
            # Bet365
            for selection, column in (("home", "B365H"), ("draw", "B365D"), ("away", "B365A")):
                value = _decimal(row.get(column, ""), column)
                if value is not None:
                    odds.append(OddsSnapshot(match_id, "Bet365", "1x2", selection, value, captured))
            
            # Pinnacle (Smart Money Baseline)
            for selection, column in (("home", "PSH"), ("draw", "PSD"), ("away", "PSA")):
                value = _decimal(row.get(column, ""), column)
                if value is not None:
                    odds.append(OddsSnapshot(match_id, "Pinnacle", "1x2", selection, value, captured))
                    
            # Pinnacle Closing Line
            for selection, column in (("home", "PSCH"), ("draw", "PSCD"), ("away", "PSCA")):
                value = _decimal(row.get(column, ""), column)
                if value is not None:
                    odds.append(OddsSnapshot(match_id, "Pinnacle", "1x2", selection, value, closing_time, is_closing=True))
    return validate_matches(matches), validate_odds(odds)


def load_many(paths: Iterable[str | Path], registry: TeamRegistry | None = None,
              competition: str = "unknown", season: str = "unknown") -> tuple[list[Match], list[OddsSnapshot]]:
    matches: list[Match] = []
    odds: list[OddsSnapshot] = []
    for path in paths:
        loaded_matches, loaded_odds = load_football_data_csv(path, registry, competition, season)
        matches.extend(loaded_matches)
        odds.extend(loaded_odds)
    return matches, odds
