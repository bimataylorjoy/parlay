"""Loader for football-data.co.uk CSV files (results, match stats and odds)."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

from .normalization import TeamRegistry
from .schemas import Match, OddsSnapshot
from .teams import canonical_team_name
from .validation import validate_matches

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _OddsColumns:
    bookmaker: str
    market: str
    columns: tuple[tuple[str, str], ...]  # (selection, column)
    closing: bool


# Pre-match ("opening" in football-data terms: collected before the weekend
# or midweek round) and closing prices. Closing prices are only ever used as
# an evaluation benchmark.
ODDS_COLUMNS: tuple[_OddsColumns, ...] = (
    _OddsColumns("Bet365", "1x2", (("home", "B365H"), ("draw", "B365D"), ("away", "B365A")), False),
    _OddsColumns("Pinnacle", "1x2", (("home", "PSH"), ("draw", "PSD"), ("away", "PSA")), False),
    _OddsColumns("Bet365", "1x2", (("home", "B365CH"), ("draw", "B365CD"), ("away", "B365CA")), True),
    _OddsColumns("Pinnacle", "1x2", (("home", "PSCH"), ("draw", "PSCD"), ("away", "PSCA")), True),
    _OddsColumns("Bet365", "totals_2.5", (("over", "B365>2.5"), ("under", "B365<2.5")), False),
    _OddsColumns("Pinnacle", "totals_2.5", (("over", "P>2.5"), ("under", "P<2.5")), False),
    _OddsColumns("Bet365", "totals_2.5", (("over", "B365C>2.5"), ("under", "B365C<2.5")), True),
    _OddsColumns("Pinnacle", "totals_2.5", (("over", "PC>2.5"), ("under", "PC<2.5")), True),
)


def parse_date(value: str) -> date:
    value = value.strip()
    for format_ in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, format_).date()
        except ValueError:
            continue
        return parsed.replace(year=parsed.year + 100) if parsed.year < 1970 else parsed
    raise ValueError(f"Unsupported match date: {value!r} (expected DD/MM/YYYY, DD/MM/YY or YYYY-MM-DD)")


def _parse_kickoff(match_date: date, value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    for format_ in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.combine(match_date, datetime.strptime(value, format_).time(), tzinfo=timezone.utc)
        except ValueError:
            continue
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


def _price(value: str | None) -> float | None:
    """Decimal odds, or ``None`` for blank / malformed / non-positive-edge prices."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        price = float(value)
    except ValueError:
        return None
    return price if price > 1.0 else None


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
    """Load completed matches, match statistics and recognized odds columns.

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
        available = set(reader.fieldnames)
        odds_groups = [group for group in ODDS_COLUMNS if all(col in available for _, col in group.columns)]
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
            closing_at = kickoff_at or pre_match_at.replace(hour=12)
            for group in odds_groups:
                parsed = {selection: _price(row.get(column)) for selection, column in group.columns}
                prices = {selection: value for selection, value in parsed.items() if value is not None}
                if len(prices) != len(parsed):
                    if any((row.get(column) or "").strip() for _, column in group.columns):
                        skipped_prices += 1
                    continue
                captured = closing_at if group.closing else pre_match_at
                odds.extend(
                    OddsSnapshot(
                        match_id, group.bookmaker, group.market, selection, price, captured, group.closing
                    )
                    for selection, price in prices.items()
                )
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
