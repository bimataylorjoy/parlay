from datetime import datetime, timedelta, timezone

from parlay.data.market_information import InjuryReport, TeamNews
from parlay.features.external import external_information_features


def test_external_features_exclude_future_information():
    as_of = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    items = [
        TeamNews("A", "news", "available", "source", as_of, as_of),
        InjuryReport("A", "p1", "out", "source", as_of + timedelta(minutes=1), as_of + timedelta(minutes=1)),
    ]
    features = external_information_features(items, as_of=as_of)
    assert features["available_information_count"] == 1
    assert features["prematch_news_count"] == 1
    assert features["injury_report_count"] == 0
