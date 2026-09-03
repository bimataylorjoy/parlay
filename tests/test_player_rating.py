from datetime import datetime, timezone, timedelta

from parlay.data.schemas import PlayerMatchStat
from parlay.features.player_rating import estimate_player_ratings


def stat(player, minutes, available):
    return PlayerMatchStat("f1", player, "A", "B", "FW", True, minutes, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 1.0, 1.0, 7.0, available, available, "test")


def test_player_rating_uses_shrinkage_and_cutoff():
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    future = old + timedelta(days=1)
    ratings = estimate_player_ratings([stat("p1", 90, old), stat("p2", 90, old), stat("p3", 90, future)], as_of=old)
    assert {row.player_id for row in ratings} == {"p1", "p2"}
    assert all(0 < row.reliability < 1 for row in ratings)
