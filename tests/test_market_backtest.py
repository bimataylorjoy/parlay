from datetime import date, datetime, timedelta, timezone

import pytest

from parlay.data.schemas import Match, OddsSnapshot
from parlay.evaluation.backtest import run_backtest
from parlay.prediction.markets import implied_probabilities, latest_odds_as_of


def test_latest_odds_respects_cutoff():
    rows = [
        OddsSnapshot("m", "book", "1x2", "home", 2.0, datetime(2024, 1, 1, tzinfo=timezone.utc)),
        OddsSnapshot("m", "book", "1x2", "home", 1.8, datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]
    selected = latest_odds_as_of(rows, datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
    assert selected[0].odds == 2.0


def test_implied_probabilities_remove_margin():
    rows = [OddsSnapshot("m", "book", "1x2", selection, odds, datetime(2024, 1, 1, tzinfo=timezone.utc))
            for selection, odds in (("home", 2.0), ("draw", 4.0), ("away", 4.0))]
    probabilities = implied_probabilities(rows)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["home_win"] == pytest.approx(0.5)


def test_backtest_populates_market_fields():
    start = date(2020, 1, 1)
    matches = [Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2) for i in range(8)]
    odds = [OddsSnapshot(str(i), "book", "1x2", selection, value, datetime.combine(start + timedelta(days=i), datetime.min.time(), timezone.utc))
            for i in range(8) for selection, value in (("home", 2.0), ("draw", 4.0), ("away", 4.0))]
    records, _ = run_backtest(matches, initial_train_days=3, test_days=2, odds=odds, bookmaker="book")
    assert records[0].market_home == pytest.approx(0.5)
    assert records[0].odds_home == 2.0


def test_backtest_uses_kickoff_minus_lead_for_odds():
    kickoff = datetime(2020, 1, 4, 20, tzinfo=timezone.utc)
    start = date(2020, 1, 1)
    matches = [Match(str(i), start + timedelta(days=i), "league", "2020", "A", "B", i % 3, (i + 1) % 2,
                      kickoff_at=kickoff if i == 3 else None) for i in range(8)]
    odds = [OddsSnapshot("3", "book", "1x2", selection, value, timestamp)
            for timestamp, value in ((datetime(2020, 1, 4, 18, 30, tzinfo=timezone.utc), 2.0),
                                     (datetime(2020, 1, 4, 19, 30, tzinfo=timezone.utc), 1.5))
            for selection in ("home", "draw", "away")]
    records, _ = run_backtest(matches, initial_train_days=3, test_days=2, odds=odds,
                               bookmaker="book", forecast_lead_minutes=60)
    target = next(row for row in records if row.match_id == "3")
    assert target.odds_home == 2.0
