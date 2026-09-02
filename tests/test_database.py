from datetime import date, datetime, timezone
import json

from parlay.data.database import ResearchDatabase
from parlay.data.schemas import Match, OddsSnapshot


def test_database_persists_time_safe_snapshots():
    db = ResearchDatabase()
    db.insert_matches([Match("m1", date(2024, 1, 2), "league", "2024", "A", "B", 1, 0)])
    db.insert_odds([
        OddsSnapshot("m1", "book", "1x2", "home", 2.0, datetime(2024, 1, 1, tzinfo=timezone.utc)),
        OddsSnapshot("m1", "book", "1x2", "home", 1.8, datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ])
    db.insert_feature_snapshot("m1", datetime(2024, 1, 1, tzinfo=timezone.utc), "v1", json.dumps({"elo": 1500}))
    assert len(db.matches_as_of(date(2024, 1, 2))) == 1
    assert len(db.odds_as_of("m1", datetime(2024, 1, 1, 12, tzinfo=timezone.utc))) == 1
    assert json.loads(db.feature_as_of("m1", "v1", datetime(2024, 1, 1, 12, tzinfo=timezone.utc))["values_json"]) == {"elo": 1500}
    loaded = db.load_matches()
    assert loaded[0].home_team == "A"
    assert loaded[0].kickoff_at is None
    assert db.load_odds(match_id="m1")[0].odds == 2.0


def test_reingest_updates_match_without_deleting_children():
    db = ResearchDatabase()
    db.insert_matches([Match("m1", date(2024, 1, 2), "league", "2024", "A", "B", 1, 0)])
    db.insert_odds([OddsSnapshot("m1", "book", "1x2", "home", 2.0, datetime(2024, 1, 1, tzinfo=timezone.utc))])
    db.insert_matches([Match("m1", date(2024, 1, 2), "league", "2024", "A", "B", 2, 1)])
    assert db.load_matches()[0].home_goals == 2
    assert len(db.load_odds(match_id="m1")) == 1
