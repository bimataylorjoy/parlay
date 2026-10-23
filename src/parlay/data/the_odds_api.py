"""The Odds API (the-odds-api.com) v4 adapter.

The free plan gives 500 credits per month; one ``/odds`` call costs
``markets × regions`` credits (only when events are returned). Region ``eu``
includes Pinnacle (scraped from its public site, possibly delayed) and
Betfair Exchange, which makes it the free source of a sharp reference price.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .loaders import match_id_for, season_for
from .odds_columns import EXCHANGE, asian_market, totals_market
from .schemas import Match, OddsSnapshot
from .teams import canonical_team_name

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {
    "EPL": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "League One": "soccer_england_league1",
    "League Two": "soccer_england_league2",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "Eredivisie": "soccer_netherlands_eredivisie",
}
BOOKMAKER_NAMES = {
    "pinnacle": "Pinnacle",
    "betfair_ex_eu": EXCHANGE,
    "betfair_ex_uk": EXCHANGE,
    "williamhill": "William Hill",
    "onexbet": "1xBet",
    "sport888": "888sport",
    "unibet_eu": "Unibet",
    "unibet_uk": "Unibet",
}
MARKETS = {"h2h": "1x2", "totals": "totals", "spreads": "ah"}


@dataclass(frozen=True, slots=True)
class Quota:
    remaining: int | None
    used: int | None
    last_cost: int | None


def _header_int(headers: Any, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def fetch_odds(
    api_key: str,
    sport_key: str,
    *,
    regions: str = "eu",
    markets: str = "h2h,totals",
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], Quota]:
    """Upcoming events with bookmaker prices for one sport, plus the remaining quota."""
    if not api_key.strip():
        raise ValueError("The Odds API key must not be empty")
    query = urlencode(
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
    )
    request = Request(
        f"{BASE_URL}/sports/{sport_key}/odds?{query}", headers={"User-Agent": "parlay-research/0.2"}
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            headers = response.headers
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OSError(f"The Odds API returned HTTP {exc.code}: {detail}") from exc
    quota = Quota(
        _header_int(headers, "x-requests-remaining"),
        _header_int(headers, "x-requests-used"),
        _header_int(headers, "x-requests-last"),
    )
    return payload, quota


def _timestamp(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_events(
    events: Iterable[dict[str, Any]], competition: str, *, retrieved_at: datetime | None = None
) -> tuple[list[Match], list[OddsSnapshot]]:
    """Convert API events into scheduled matches and timestamped price snapshots.

    Teams are canonicalized so that match IDs line up with football-data
    results. Spreads become Asian-handicap markets keyed by the home line.
    """
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    matches: list[Match] = []
    snapshots: list[OddsSnapshot] = []
    for event in events:
        kickoff = _timestamp(event.get("commence_time"), retrieved_at)
        home_raw, away_raw = event["home_team"], event["away_team"]
        home, away = canonical_team_name(home_raw), canonical_team_name(away_raw)
        match_id = match_id_for(kickoff.date(), home, away, competition)
        matches.append(
            Match(
                match_id=match_id,
                date=kickoff.date(),
                competition=competition,
                season=season_for(kickoff.date()),
                home_team=home,
                away_team=away,
                home_goals=None,
                away_goals=None,
                source="the-odds-api",
                kickoff_at=kickoff,
            )
        )
        for book in event.get("bookmakers", []):
            bookmaker = BOOKMAKER_NAMES.get(book["key"], book.get("title") or book["key"])
            for market in book.get("markets", []):
                kind = MARKETS.get(market.get("key", ""))
                if kind is None:
                    continue
                captured = _timestamp(market.get("last_update") or book.get("last_update"), retrieved_at)
                captured = min(captured, retrieved_at)
                if kind == "1x2":
                    names = {home_raw: "home", away_raw: "away", "Draw": "draw"}
                    prices = {
                        names[o["name"]]: float(o["price"]) for o in market["outcomes"] if o["name"] in names
                    }
                    if set(prices) == {"home", "draw", "away"}:
                        snapshots.extend(
                            OddsSnapshot(match_id, bookmaker, "1x2", sel, price, captured)
                            for sel, price in prices.items()
                        )
                    continue
                grouped: dict[float, dict[str, float]] = {}
                for outcome in market["outcomes"]:
                    point = outcome.get("point")
                    if point is None:
                        continue
                    if kind == "totals":
                        side = outcome["name"].lower()
                        if side in ("over", "under"):
                            grouped.setdefault(float(point), {})[side] = float(outcome["price"])
                    elif outcome["name"] == home_raw:
                        grouped.setdefault(float(point), {})["home"] = float(outcome["price"])
                    elif outcome["name"] == away_raw:
                        grouped.setdefault(-float(point), {})["away"] = float(outcome["price"])
                for line, prices in grouped.items():
                    if len(prices) != 2:
                        continue
                    key = totals_market(line) if kind == "totals" else asian_market(line)
                    snapshots.extend(
                        OddsSnapshot(match_id, bookmaker, key, sel, p, captured) for sel, p in prices.items()
                    )
    return matches, snapshots
