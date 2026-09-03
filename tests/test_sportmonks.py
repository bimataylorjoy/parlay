from datetime import date

from parlay.data.sportmonks import extract_totals_25, fixture_to_match, totals_to_odds, parse_fixture_information, premium_odds_snapshots


def payload():
    return {
        "id": 123, "starting_at": "2026-09-05 14:00:00",
        "name": "Manchester City vs Coventry City",
        "participants": [
            {"name": "Coventry City", "meta": {"location": "away"}},
            {"name": "Manchester City", "meta": {"location": "home"}},
        ],
        "odds": [
            {"market_description": "Goal Line", "total": "2.5", "label": "Over", "value": "1.80", "stopped": False, "bookmaker_id": 2, "latest_bookmaker_update": "2026-09-01 12:00:00"},
            {"market_description": "Goal Line", "total": "2.5", "label": "Under", "value": "2.10", "stopped": False, "bookmaker_id": 2, "latest_bookmaker_update": "2026-09-01 12:00:00"},
        ],
    }


def test_sportmonks_payload_maps_fixture_and_totals():
    fixture = payload()
    match = fixture_to_match(fixture)
    assert match.match_id == "sportmonks:123"
    assert match.date == date(2026, 9, 5)
    assert match.home_goals is None and match.away_goals is None
    totals = extract_totals_25(fixture, bookmaker_id=2)
    assert totals["over_odds"] == 1.8
    odds = totals_to_odds(fixture, totals)
    assert {row.selection for row in odds} == {"over", "under"}


def test_sportmonks_enrichment_is_timestamped():
    fixture = {
        "id": 123,
        "starting_at": "2026-09-05T15:00:00Z",
        "lineups": [{"team_id": 1, "type": "starting", "updated_at": "2026-09-05T13:00:00Z", "players": [{"player_id": 9}]}],
        "sidelined": [{"team_id": 1, "player_id": 10, "type": "injury", "updated_at": "2026-09-05T12:00:00Z"}],
        "prematchNews": [{"team_id": 1, "title": "Manager update", "published_at": "2026-09-05T11:00:00Z"}],
    }
    parsed = parse_fixture_information(fixture)
    assert parsed["lineups"][0].available_at.isoformat().startswith("2026-09-05T13")
    assert parsed["injuries"][0].is_available_at(__import__("datetime").datetime(2026, 9, 5, 12, tzinfo=__import__("datetime").timezone.utc))
    assert parsed["news"][0].status == "Manager update"


def test_premium_odds_history_normalizes_updates():
    payload = {"fixture_id": 123, "odds": [
        {"market_description": "1x2", "label": "Home", "value": 2.0, "latest_bookmaker_update": "2026-09-05T10:00:00Z"},
        {"market_description": "1x2", "label": "Home", "value": 1.8, "latest_bookmaker_update": "2026-09-05T12:00:00Z"},
    ]}
    rows = premium_odds_snapshots(payload)
    assert len(rows) == 2
    assert rows[1].odds == 1.8
