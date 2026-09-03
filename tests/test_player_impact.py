from parlay.features.player_impact import PlayerRating, availability_adjustment


def test_player_impact_requires_ratings_and_is_capped():
    result = availability_adjustment([PlayerRating("p1", "A", 1.0, 0.8)], {"p1", "unknown"})
    assert result.status == "estimated"
    assert result.covered_players == 1
    assert result.missing_players == 1
    assert result.attack_delta == -0.35


def test_player_impact_is_unavailable_without_ratings():
    result = availability_adjustment([], {"p1"})
    assert result.status == "unavailable"
    assert result.attack_delta == 0.0
