import json
from datetime import datetime, timezone
from typing import ClassVar

import pytest

from parlay.data.loaders import load_football_data_fixtures, match_id_for
from parlay.data.odds_columns import EXCHANGE, MARKET_MAX
from parlay.data.the_odds_api import Quota, fetch_odds, parse_events

NOW = datetime(2030, 1, 4, 12, tzinfo=timezone.utc)

EVENT = {
    "id": "abc",
    "sport_key": "soccer_epl",
    "commence_time": "2030-01-05T15:00:00Z",
    "home_team": "Manchester City",
    "away_team": "Brighton and Hove Albion",
    "bookmakers": [
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2030-01-04T11:58:00Z",
                    "outcomes": [
                        {"name": "Manchester City", "price": 1.5},
                        {"name": "Brighton and Hove Albion", "price": 6.5},
                        {"name": "Draw", "price": 4.6},
                    ],
                },
                {
                    "key": "totals",
                    "last_update": "2030-01-04T11:58:00Z",
                    "outcomes": [
                        {"name": "Over", "price": 1.8, "point": 2.75},
                        {"name": "Under", "price": 2.05, "point": 2.75},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Manchester City", "price": 1.95, "point": -1.25},
                        {"name": "Brighton and Hove Albion", "price": 1.93, "point": 1.25},
                    ],
                },
            ],
        },
        {
            "key": "betfair_ex_eu",
            "title": "Betfair",
            "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester City", "price": 1.52}]}],
        },
        {
            "key": "williamhill",
            "title": "William Hill",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2030-01-04T13:00:00Z",  # in the future relative to retrieval: clamped
                    "outcomes": [
                        {"name": "Manchester City", "price": 1.53},
                        {"name": "Brighton and Hove Albion", "price": 6.0},
                        {"name": "Draw", "price": 4.5},
                    ],
                }
            ],
        },
    ],
}


def test_parse_events_maps_teams_markets_and_timestamps():
    matches, odds = parse_events([EVENT], "EPL", retrieved_at=NOW)
    match = matches[0]
    assert (match.home_team, match.away_team) == ("Man City", "Brighton")
    assert match.match_id == match_id_for(match.date, "Man City", "Brighton", "EPL")
    assert match.kickoff_at == datetime(2030, 1, 5, 15, tzinfo=timezone.utc)
    keys = {(o.bookmaker, o.market, o.selection): o.odds for o in odds}
    assert keys[("Pinnacle", "1x2", "draw")] == 4.6
    assert keys[("Pinnacle", "totals_2.75", "over")] == 1.8
    assert keys[("Pinnacle", "ah_-1.25", "home")] == 1.95
    assert keys[("Pinnacle", "ah_-1.25", "away")] == 1.93
    assert not any(o.bookmaker == EXCHANGE for o in odds)  # incomplete market dropped
    assert all(o.captured_at <= NOW for o in odds)


def test_fetch_odds_reads_quota_headers(monkeypatch):
    class Response:
        headers: ClassVar = {"x-requests-remaining": "498", "x-requests-used": "2", "x-requests-last": "2"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            return json.dumps([EVENT]).encode()

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("parlay.data.the_odds_api.urlopen", fake_urlopen)
    events, quota = fetch_odds("KEY", "soccer_epl", markets="h2h")
    assert events[0]["id"] == "abc"
    assert quota == Quota(498, 2, 2)
    assert "regions=eu" in captured["url"] and "apiKey=KEY" in captured["url"]
    with pytest.raises(ValueError):
        fetch_odds(" ", "soccer_epl")


FIXTURES = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,BFEH,BFED,BFEA,MaxH,MaxD,MaxA,B365>2.5,B365<2.5,AHh,B365AHH,B365AHA\n"
    "E0,05/01/2030,15:00,Man City,Brighton,1.5,4.5,6.5,1.52,4.7,7.0,1.55,4.8,7.2,1.7,2.15,-1.25,1.95,1.95\n"
    "SP1,05/01/2030,20:00,Barcelona,Girona,1.3,6,9,,,,,,,,,,,\n"
)


def test_fixtures_loader_keeps_requested_divisions_only(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text(FIXTURES, encoding="utf-8")
    matches, odds = load_football_data_fixtures(path, captured_at=NOW, leagues={"E0": "EPL"})
    assert [m.home_team for m in matches] == ["Man City"]
    assert matches[0].home_goals is None and matches[0].kickoff_at.hour == 15  # GMT in January
    books = {(o.bookmaker, o.market) for o in odds}
    assert {("Bet365", "1x2"), (EXCHANGE, "1x2"), (MARKET_MAX, "1x2"), ("Bet365", "totals_2.5")} <= books
    assert ("Bet365", "ah_-1.25") in books
    assert all(o.captured_at == NOW and not o.is_closing for o in odds)
