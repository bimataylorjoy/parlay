import numpy as np
import pytest
from datetime import date
from parlay.data.schemas import Match
from parlay.models.corners import (
    CornerStrengthModel,
    corner_match_betting,
    corner_totals_probabilities,
    fit_corner_strength,
)


def _make_match(mid: str, h: str, a: str, hg: int, ag: int, hc: int, ac: int, days_ago: int = 0) -> Match:
    return Match(
        match_id=mid,
        date=date(2025, 1, 1),
        competition="E0",
        season="2425",
        home_team=h,
        away_team=a,
        home_goals=hg,
        away_goals=ag,
        home_corners=hc,
        away_corners=ac,
    )


def test_corner_strength_model_and_matrix():
    matches = [
        _make_match("m1", "Arsenal", "Chelsea", 2, 1, 8, 3),
        _make_match("m2", "Chelsea", "Arsenal", 0, 1, 4, 7),
        _make_match("m3", "Liverpool", "Chelsea", 3, 0, 9, 2),
        _make_match("m4", "Arsenal", "Liverpool", 2, 2, 6, 6),
    ]

    model = fit_corner_strength(matches)
    assert model.corner_attack["Arsenal"] > model.corner_attack["Chelsea"]

    matrix = model.corner_matrix("Arsenal", "Chelsea")
    assert np.isclose(matrix.sum(), 1.0)
    assert matrix.shape == (21, 21)

    # Expected corners
    exp_h, exp_a = model.expected_corners("Arsenal", "Chelsea")
    assert exp_h > exp_a
    assert 1.0 < exp_h < 15.0


def test_corner_markets():
    matches = [
        _make_match("m1", "Arsenal", "Chelsea", 2, 1, 7, 4),
        _make_match("m2", "Chelsea", "Arsenal", 1, 1, 5, 5),
    ]
    model = fit_corner_strength(matches)
    matrix = model.corner_matrix("Arsenal", "Chelsea")

    # Corner totals (10.5)
    tot = corner_totals_probabilities(matrix, 10.5)
    assert np.isclose(tot["over"] + tot["under"], 1.0)
    assert tot["push"] == 0.0

    # Corner totals quarter line (10.75)
    tot_quarter = corner_totals_probabilities(matrix, 10.75)
    assert np.isclose(sum(tot_quarter.values()), 1.0)

    # 1X2 Most corners
    match_1x2 = corner_match_betting(matrix)
    assert np.isclose(sum(match_1x2.values()), 1.0)
    assert match_1x2["home_most"] > 0
