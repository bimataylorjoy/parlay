"""Sportmonks fixture and totals-odds adapter."""

from datetime import date, datetime, timezone
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .schemas import Match, OddsSnapshot, PlayerMatchStat
from .market_information import InjuryReport, LineupStatus, TeamNews
from .normalization import canonical_team_name


BASE_URL = "https://api.sportmonks.com/v3/football"


def _get_json(token: str, path: str, *, include: str | None = None, timeout: int = 30) -> dict:
    if not token.strip():
        raise ValueError("Sportmonks token must not be empty")
    params = {"api_token": token}
    if include:
        params["include"] = include
    request = Request(
        f"{BASE_URL}/{path.lstrip('/')}?{urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": "parlay-research/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_fixtures_with_odds(token: str, dates: tuple[date, ...], *, league_id: int = 8) -> list[dict]:
    if not token.strip():
        raise ValueError("Sportmonks token must not be empty")
    fixtures: list[dict] = []
    for match_date in dates:
        payload = _get_json(token, f"fixtures/date/{match_date.isoformat()}", include="participants;odds")
        fixtures.extend(row for row in payload.get("data", []) if row.get("league_id") == league_id)
    return fixtures


def fetch_fixture_enrichment(token: str, fixture_id: int | str) -> dict:
    """Fetch timestamped pre-match information for one fixture.

    The raw response is retained so provider-specific fields are not lost.
    """
    return _get_json(
        token,
        f"fixtures/{fixture_id}",
        include="participants;lineups;predictedLineups;sidelined.sideline;sidelined.player;prematchNews;odds",
    ).get("data", {})


def _parse_available(value: str | datetime | None, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return fallback


def parse_fixture_information(fixture: dict, *, source: str = "sportmonks") -> dict[str, list[object]]:
    """Convert known enrichment entities into timestamp-aware domain objects.

    Sportmonks payload shapes can vary by subscription. Unknown/missing fields
    are ignored rather than fabricated.
    """
    kickoff = datetime.fromisoformat(fixture["starting_at"].replace("Z", "+00:00"))
    published_default = kickoff
    output: dict[str, list[object]] = {"lineups": [], "injuries": [], "news": []}
    for row in fixture.get("lineups", []) or []:
        participant = row.get("team_id") or row.get("team", {}).get("id")
        team = str(participant) if participant is not None else str(row.get("team_name", "unknown"))
        published = _parse_available(row.get("updated_at") or row.get("created_at"), published_default)
        output["lineups"].append(LineupStatus(
            match_id=f"sportmonks:{fixture['id']}", team=team,
            status=str(row.get("type", row.get("confirmed", "unknown"))), source=source,
            published_at=published, available_at=published,
            player_ids=tuple(str(p.get("player_id", p.get("id"))) for p in row.get("players", []) if p.get("player_id", p.get("id")) is not None),
        ))
    for row in fixture.get("sidelined", []) or []:
        player = row.get("player", {}) or {}
        team = row.get("team_id") or row.get("team", {}).get("id") or "unknown"
        published = _parse_available(row.get("updated_at") or row.get("created_at"), published_default)
        output["injuries"].append(InjuryReport(
            team=str(team), player_id=str(player.get("id", row.get("player_id", "unknown"))),
            status=str((row.get("sideline") or {}).get("type", row.get("type", "unknown"))),
            source=source, published_at=published, available_at=published,
            expected_return=(row.get("sideline") or {}).get("expected_return"),
        ))
    for row in fixture.get("prematchNews", []) or []:
        published = _parse_available(row.get("published_at") or row.get("created_at"), published_default)
        output["news"].append(TeamNews(
            team=str(row.get("team_id", "match")), category="prematch_news",
            status=str(row.get("title", row.get("type", "published"))), source=source,
            published_at=published, available_at=published, note=row.get("description", row.get("content")),
        ))
    return output


def fetch_premium_odds_history(
    token: str, fixture_id: int | str, bookmaker_id: int | str,
) -> dict:
    """Fetch premium pre-match odds and provider update history."""
    return _get_json(
        token,
        f"odds/premium/fixtures/{fixture_id}/bookmakers/{bookmaker_id}",
        include="history",
    ).get("data", {})


def premium_odds_snapshots(
    payload: dict, *, match_id: str | None = None, bookmaker: str = "sportmonks",
) -> list[OddsSnapshot]:
    """Normalize premium odds history into immutable timestamped snapshots."""
    snapshots: list[OddsSnapshot] = []
    match_id = match_id or f"sportmonks:{payload.get('fixture_id', payload.get('fixture', 'unknown'))}"
    for odd in payload.get("odds", payload.get("data", [])) or []:
        selection = odd.get("label") or odd.get("name") or odd.get("selection")
        value = odd.get("value") or odd.get("odds")
        captured = odd.get("latest_bookmaker_update") or odd.get("created_at") or odd.get("captured_at")
        if selection is None or value is None or captured is None:
            continue
        snapshots.append(OddsSnapshot(
            match_id=match_id, bookmaker=bookmaker, market=str(odd.get("market_description", odd.get("market", "unknown"))),
            selection=str(selection), odds=float(value),
            captured_at=datetime.fromisoformat(str(captured).replace("Z", "+00:00")),
            is_closing=bool(odd.get("is_closing", False)),
        ))
    return snapshots


def parse_player_match_stats(
    fixture: dict, *, source: str = "sportmonks",
) -> list[PlayerMatchStat]:
    """Normalize fixture player statistics with conservative availability time."""
    kickoff = datetime.fromisoformat(fixture["starting_at"].replace("Z", "+00:00"))
    participants = {row.get("id"): row for row in fixture.get("participants", [])}
    rows: list[PlayerMatchStat] = []
    for lineup in fixture.get("lineups", []) or []:
        player_id = lineup.get("player_id") or lineup.get("player", {}).get("id")
        team_id = lineup.get("team_id")
        if player_id is None or team_id is None:
            continue
        opponent_id = next((key for key in participants if key != team_id), None)
        statistics = lineup.get("statistics", lineup.get("stats", {})) or {}
        def number(*keys: str) -> float:
            for key in keys:
                value = statistics.get(key, lineup.get(key))
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0
        rows.append(PlayerMatchStat(
            fixture_id=f"sportmonks:{fixture['id']}", player_id=str(player_id), team_id=str(team_id),
            opponent_id=str(opponent_id) if opponent_id is not None else None,
            position=str(lineup.get("position", lineup.get("detailed_position", ""))) or None,
            started=str(lineup.get("type", lineup.get("starter", ""))).casefold() in {"starting", "starter", "1", "true"},
            minutes=number("minutes", "minutes_played"), goals=number("goals"), assists=number("assists"),
            shots=number("shots", "total_shots"), shots_on_target=number("shots_on_target", "shots_on_target"),
            key_passes=number("key_passes", "key_passes_total"), tackles=number("tackles"),
            interceptions=number("interceptions"), clearances=number("clearances"), blocks=number("blocks"),
            errors=number("errors", "errors_leading_to_goal"), xg=statistics.get("xg"), xa=statistics.get("xa"),
            rating=statistics.get("rating"), observed_at=kickoff, available_at=kickoff, source=source,
        ))
    return rows


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
