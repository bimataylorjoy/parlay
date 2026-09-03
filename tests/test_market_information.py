from datetime import datetime, timedelta, timezone

from parlay.data.market_information import (
    InjuryReport,
    LineupStatus,
    TeamNews,
    available_team_information,
    summarize_odds_movement,
)
from parlay.data.schemas import OddsSnapshot


def test_external_information_is_timestamp_safe():
    as_of = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    items = [
        TeamNews("A", "news", "questionable", "source", as_of - timedelta(hours=1), as_of - timedelta(minutes=1)),
        InjuryReport("A", "p1", "out", "source", as_of + timedelta(minutes=1), as_of + timedelta(minutes=1)),
        LineupStatus("m1", "A", "confirmed", "source", as_of, as_of),
    ]
    assert len(available_team_information(items, as_of)) == 2


def test_odds_movement_is_descriptive_and_cutoff_safe():
    base = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    rows = [
        OddsSnapshot("m1", "Book", "1x2", "home", 2.0, base),
        OddsSnapshot("m1", "Book", "1x2", "home", 1.8, base + timedelta(hours=1)),
        OddsSnapshot("m1", "Book", "1x2", "home", 1.5, base + timedelta(hours=2)),
    ]
    result = summarize_odds_movement(rows, as_of=base + timedelta(hours=1))
    assert len(result) == 1
    assert result[0].latest_odds == 1.8
    assert result[0].observations == 2
