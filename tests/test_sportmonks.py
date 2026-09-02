from datetime import date

from parlay.data.sportmonks import extract_totals_25, fixture_to_match, totals_to_odds


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
