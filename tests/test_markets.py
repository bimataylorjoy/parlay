import math
from datetime import datetime, timezone

import numpy as np
import pytest

from parlay.data.schemas import OddsSnapshot
from parlay.models.poisson import score_matrix
from parlay.prediction.markets import (
    asian_handicap,
    btts_probabilities,
    correct_score_probabilities,
    de_vig,
    expected_value,
    fair_odds,
    grouped_score_markets,
    implied_probabilities,
    kelly_criterion,
    latest_odds_as_of,
    most_likely_scores,
    outcome_probabilities,
    overround,
    team_totals_probabilities,
    totals_probabilities,
)
from parlay.prediction.parlay import btts_leg, over_leg, parlay_from_score_matrix


@pytest.fixture
def matrix():
    return score_matrix(1.5, 1.2)


def test_outcomes_and_complementary_markets(matrix):
    assert np.isclose(sum(outcome_probabilities(matrix).values()), 1.0)
    totals = totals_probabilities(matrix, 2.5)
    assert totals["push"] == 0 and np.isclose(totals["over"] + totals["under"], 1.0)
    assert np.isclose(sum(btts_probabilities(matrix).values()), 1.0)
    assert np.isclose(sum(asian_handicap(matrix, 0).values()), 1.0)


def test_lines_must_be_quarter_multiples(matrix):
    for bad in (float("nan"), 2.3):
        with pytest.raises(ValueError, match=r"multiple of 0\.25"):
            totals_probabilities(matrix, bad)
        with pytest.raises(ValueError, match=r"multiple of 0\.25"):
            asian_handicap(matrix, bad)


def test_integer_line_pushes_and_ev_returns_stake():
    point = np.zeros((5, 5))
    point[1, 2] = 1.0
    settlement = totals_probabilities(point, 3.0)
    assert settlement == {"over": 0.0, "push": 1.0, "under": 0.0}
    assert expected_value(settlement["over"], 2.0, settlement["under"]) == 0.0


def test_quarter_lines_average_adjacent_lines(matrix):
    ah = asian_handicap(matrix, -0.25)
    zero, half = asian_handicap(matrix, 0.0), asian_handicap(matrix, -0.5)
    for key in ah:
        assert ah[key] == pytest.approx(0.5 * (zero[key] + half[key]))
    t275, t25, t30 = (totals_probabilities(matrix, line) for line in (2.75, 2.5, 3.0))
    for key in t275:
        assert t275[key] == pytest.approx(0.5 * (t25[key] + t30[key]))


def test_team_totals_use_marginals(matrix):
    home = team_totals_probabilities(matrix, 0.5, "home")
    assert home["under"] == pytest.approx(matrix[0, :].sum())
    with pytest.raises(ValueError):
        team_totals_probabilities(matrix, 0.5, "draw")


def test_correct_score_grouped_and_top_scores(matrix):
    cs = correct_score_probabilities(matrix)
    assert np.isclose(sum(cs.values()), 1.0)
    truncated = correct_score_probabilities(matrix, max_home=2, max_away=2)
    assert "other" in truncated and np.isclose(sum(truncated.values()), 1.0)
    grouped = grouped_score_markets(matrix)
    assert grouped["scoreless_draw"] == pytest.approx(cs["0-0"])
    assert grouped["double_chance_1x"] > grouped["home_win_to_nil"]
    top = most_likely_scores(matrix, 3)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]


def test_fair_odds_and_expected_value():
    assert fair_odds(0.5) == 2.0
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        expected_value(0.5, 1.0)


def test_kelly_caps_and_handles_push():
    assert kelly_criterion(0.9, 10.0, fraction=1.0, max_stake=0.25) == pytest.approx(0.25)
    assert kelly_criterion(0.4, 2.0) == 0.0
    # With push mass the optimal stake is (b*p - q) / (b*(p+q))
    p, q, odds = 0.5, 0.3, 2.0
    assert kelly_criterion(p, odds, loss_probability=q) == pytest.approx((1.0 * p - q) / (1.0 * (p + q)))
    with pytest.raises(ValueError, match="fraction"):
        kelly_criterion(0.5, 2.0, fraction=1.1)


def test_de_vig_methods_sum_to_one_and_shin_corrects_longshots():
    raw = {"home": 1 / 1.5, "draw": 1 / 4.0, "away": 1 / 7.0}
    results = {method: de_vig(raw, method=method) for method in ("shin", "power", "multiplicative")}
    for probs in results.values():
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(0 < v < 1 for v in probs.values())
    assert results["shin"]["away"] < results["multiplicative"]["away"]
    assert de_vig({"a": 0.5, "b": 0.5})["a"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        de_vig(raw, method="magic")


def test_overround():
    assert overround([2.0, 2.0]) == pytest.approx(0.0)
    assert overround([1.9, 1.9]) > 0


def _snapshot(selection, odds, hour, closing=False):
    return OddsSnapshot(
        "m", "book", "1x2", selection, odds, datetime(2024, 1, 1, hour, tzinfo=timezone.utc), closing
    )


def test_latest_odds_respects_cutoff_and_ignores_closing():
    rows = [_snapshot("home", 2.0, 9), _snapshot("home", 1.8, 12), _snapshot("home", 1.5, 10, closing=True)]
    selected = latest_odds_as_of(rows, datetime(2024, 1, 1, 11, tzinfo=timezone.utc))
    assert [r.odds for r in selected] == [2.0]


def test_implied_probabilities_remove_margin():
    rows = [_snapshot(sel, odds, 9) for sel, odds in (("home", 2.0), ("draw", 4.0), ("away", 4.0))]
    probabilities = implied_probabilities(rows)
    assert probabilities["home_win"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        implied_probabilities(rows[:2])


def test_parlay_uses_joint_probability_not_product():
    matrix = score_matrix(1.8, 1.2, max_goals=12)
    result = parlay_from_score_matrix(matrix, [btts_leg(), over_leg(2.5)], decimal_odds=5.0)
    assert result.joint_probability != pytest.approx(result.independence_probability)
    assert "leg_dependence_present" in result.diagnostics
    assert result.expected_value == pytest.approx(result.joint_probability * 5.0 - 1.0)
    assert math.isfinite(result.dependence_adjustment)
