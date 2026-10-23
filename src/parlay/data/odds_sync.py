"""Store upcoming matches and prices from the free odds feeds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .database import ResearchDatabase
from .loaders import load_football_data_fixtures
from .sources import FOOTBALL_DATA_LEAGUES, acquire_csv

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


@dataclass(frozen=True, slots=True)
class SyncResult:
    matches: int
    prices: int
    bookmakers: tuple[str, ...]
    credits_remaining: int | None = None


def sync_football_data_fixtures(
    database: ResearchDatabase,
    leagues: Iterable[str],
    *,
    now: datetime,
    snapshot_dir: str | Path = "data/raw/snapshots",
) -> SyncResult:
    """Download football-data ``fixtures.csv`` (free, no key) and store its prices."""
    codes = list(leagues)
    unknown = [code for code in codes if code not in FOOTBALL_DATA_LEAGUES]
    if unknown:
        raise ValueError(f"unknown division codes {unknown}; known: {', '.join(FOOTBALL_DATA_LEAGUES)}")
    snapshot, _ = acquire_csv(FIXTURES_URL, snapshot_dir)
    matches, odds = load_football_data_fixtures(
        snapshot, captured_at=now, leagues={code: FOOTBALL_DATA_LEAGUES[code] for code in codes}
    )
    database.insert_future_matches(matches)
    database.insert_odds(odds)
    return SyncResult(len(matches), len(odds), tuple(sorted({o.bookmaker for o in odds})))


def sync_the_odds_api(
    database: ResearchDatabase,
    api_key: str,
    competitions: Iterable[str],
    *,
    now: datetime,
    regions: str = "eu",
    markets: str = "h2h,totals",
) -> SyncResult:
    """Fetch The Odds API prices for upcoming (not yet started) matches and store them.

    Costs ``markets × regions`` credits per competition that has events.
    """
    from .the_odds_api import SPORT_KEYS, fetch_odds, parse_events

    matches, odds, quota = [], [], None
    for competition in competitions:
        if competition not in SPORT_KEYS:
            raise ValueError(f"unknown competition {competition!r}; known: {', '.join(SPORT_KEYS)}")
        events, quota = fetch_odds(api_key, SPORT_KEYS[competition], regions=regions, markets=markets)
        upcoming = [
            e for e in events if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now
        ]
        found_matches, found_odds = parse_events(upcoming, competition, retrieved_at=now)
        matches += found_matches
        odds += found_odds
    database.insert_future_matches(matches)
    database.insert_odds(odds)
    return SyncResult(
        len(matches),
        len(odds),
        tuple(sorted({o.bookmaker for o in odds})),
        quota.remaining if quota else None,
    )


def credit_cost(markets: str, regions: str, competitions: int = 1) -> int:
    """The Odds API cost of one sync: markets × regions per competition."""
    return (
        len([m for m in markets.split(",") if m]) * len([r for r in regions.split(",") if r]) * competitions
    )
