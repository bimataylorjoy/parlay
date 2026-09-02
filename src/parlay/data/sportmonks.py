"""Sportmonks fixture and totals-odds adapter."""

from datetime import date, datetime, timezone
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .schemas import Match, OddsSnapshot
from .normalization import canonical_team_name


BASE_URL = "https://api.sportmonks.com/v3/football"


def fetch_fixtures_with_odds(token: str, dates: tuple[date, ...], *, league_id: int = 8) -> list[dict]:
    if not token.strip():
        raise ValueError("Sportmonks token must not be empty")
    fixtures: list[dict] = []
    for match_date in dates:
        query = urlencode({"api_token": token, "include": "participants;odds"})
        request = Request(f"{BASE_URL}/fixtures/date/{match_date.isoformat()}?{query}", headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        fixtures.extend(row for row in payload.get("data", []) if row.get("league_id") == league_id)
    return fixtures


def fixture_to_match(fixture: dict, *, competition: str = "EPL", season: str = "2026-27") -> Match:
    participants = fixture.get("participants", [])
    home = next(row for row in participants if row.get("meta", {}).get("location") == "home")
    away = next(row for row in participants if row.get("meta", {}).get("location") == "away")
    kickoff = datetime.fromisoformat(fixture["starting_at"].replace("Z", "+00:00"))
    home_name = canonical_team_name(home["name"])
    away_name = canonical_team_name(away["name"])
    return Match(
        match_id=f"sportmonks:{fixture['id']}", date=kickoff.date(),
        competition=competition, season=season,
        home_team=home_name, away_team=away_name,
        home_goals=None, away_goals=None, kickoff_at=kickoff,
        source="sportmonks", source_timestamp=datetime.now(timezone.utc),
    )


def extract_totals_25(fixture: dict, *, bookmaker_id: int | None = None) -> dict[str, object] | None:
    candidates = []
    for odd in fixture.get("odds", []):
        if odd.get("market_description") not in {"Goals Over/Under", "Goal Line"}:
            continue
        if str(odd.get("total")) != "2.5" or odd.get("label") not in {"Over", "Under"}:
            continue
        if odd.get("stopped") or bookmaker_id is not None and odd.get("bookmaker_id") != bookmaker_id:
            continue
        candidates.append(odd)
    by_label: dict[str, dict] = {}
    for odd in sorted(candidates, key=lambda row: row.get("latest_bookmaker_update") or "", reverse=True):
        by_label.setdefault(odd["label"], odd)
    if set(by_label) != {"Over", "Under"}:
        return None
    return {
        "fixture_id": fixture["id"],
        "start_time": fixture["starting_at"],
        "fixture": fixture["name"],
        "over_odds": float(by_label["Over"]["value"]),
        "under_odds": float(by_label["Under"]["value"]),
        "bookmaker_id": by_label["Over"].get("bookmaker_id"),
        "updated_at": max(by_label["Over"].get("latest_bookmaker_update", ""), by_label["Under"].get("latest_bookmaker_update", "")),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def totals_to_odds(fixture: dict, totals: dict[str, object], *, bookmaker: str = "sportmonks") -> list[OddsSnapshot]:
    captured = datetime.now(timezone.utc)
    match_id = f"sportmonks:{fixture['id']}"
    return [
        OddsSnapshot(match_id, bookmaker, "totals_2.5", "over", float(totals["over_odds"]), captured),
        OddsSnapshot(match_id, bookmaker, "totals_2.5", "under", float(totals["under_odds"]), captured),
    ]
