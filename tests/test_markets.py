import math
import numpy as np
import pytest

from parlay.models.poisson import (
    btts_probability,
    outcome_probabilities,
    score_matrix,
    totals_probability,
)
from parlay.prediction.markets import (
    asian_handicap,
    correct_score_probabilities,
    de_vig,
    fair_odds,
    grouped_score_markets,
    kelly_criterion,
    model_edge,
    totals_expected_value,
    totals_settlement_probabilities,
)


def test_score_matrix_and_outcomes_are_normalized():
    matrix = score_matrix(1.4, 1.0)
    assert np.isclose(matrix.sum(), 1.0)
    assert np.isclose(sum(outcome_probabilities(matrix).values()), 1.0)


def test_derived_markets_are_complementary():
    matrix = score_matrix(1.4, 1.0)
    totals = totals_probability(matrix, 2.5)
    btts = btts_probability(matrix)
    assert np.isclose(totals["over"] + totals["under"], 1.0)
    assert np.isclose(btts["yes"] + btts["no"], 1.0)
    assert np.isclose(sum(asian_handicap(matrix, 0).values()), 1.0)


def test_totals_line_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        totals_probability(score_matrix(1.0, 1.0), float("nan"))


def test_integer_totals_line_includes_push():
    matrix = np.zeros((5, 5))
    matrix[1, 2] = 1.0
    probabilities = totals_settlement_probabilities(matrix, 3.0)
    assert probabilities == {"over": 0.0, "push": 1.0, "under": 0.0}
    assert totals_expected_value(probabilities, 2.0) == 0.0


def test_market_helpers():
    assert fair_odds(0.5) == 2.0
    assert model_edge(0.5, 2.2) == pytest.approx(0.1)
    assert de_vig({"home": 0.5, "draw": 0.25, "away": 0.25})["home"] == pytest.approx(0.5)


def test_kelly_validates_fraction_and_caps_stake():
    assert kelly_criterion(0.9, 10.0, fraction=1.0, max_stake=0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="fraction"):
        kelly_criterion(0.5, 2.0, fraction=1.1)


def test_de_vig_methods():
    # 1X2 odds with overround: 1.50 (0.667), 4.00 (0.250), 7.00 (0.143) => sum = 1.06
    raw_probs = {"home": 1.0 / 1.50, "draw": 1.0 / 4.00, "away": 1.0 / 7.00}

    # Shin
    shin = de_vig(raw_probs, method="shin")
    assert np.isclose(sum(shin.values()), 1.0)
    assert all(0 < v < 1 for v in shin.values())

    # Power
    power = de_vig(raw_probs, method="power")
    assert np.isclose(sum(power.values()), 1.0)

    # Multiplicative
    mult = de_vig(raw_probs, method="multiplicative")
    assert np.isclose(sum(mult.values()), 1.0)

    # Shin adjusts for favorite-longshot bias: longshot probability is lower than multiplicative
    assert shin["away"] < mult["away"]


def test_quarter_asian_handicap_and_totals():
    matrix = score_matrix(1.5, 1.2)

    # Test Asian handicap quarter lines
    ah_minus_025 = asian_handicap(matrix, -0.25)
    assert np.isclose(sum(ah_minus_025.values()), 1.0)
    assert 0 < ah_minus_025["win"] < 1.0

    # Quarter line -0.25 is average of 0.0 and -0.5
    ah_0 = asian_handicap(matrix, 0.0)
    ah_minus_05 = asian_handicap(matrix, -0.5)
    assert np.isclose(ah_minus_025["win"], 0.5 * (ah_0["win"] + ah_minus_05["win"]))

    # Test Totals quarter line (e.g. 2.75 is average of 2.5 and 3.0)
    tot_275 = totals_settlement_probabilities(matrix, 2.75)
    tot_25 = totals_settlement_probabilities(matrix, 2.5)
    tot_30 = totals_settlement_probabilities(matrix, 3.0)
    assert np.isclose(sum(tot_275.values()), 1.0)
    assert np.isclose(tot_275["over"], 0.5 * (tot_25["over"] + tot_30["over"]))
    assert np.isclose(tot_275["under"], 0.5 * (tot_25["under"] + tot_30["under"]))

    # Under EV calculation
    ev_under = totals_expected_value(tot_25, 1.95, side="under")
    assert math.isfinite(ev_under)


def test_correct_score_and_grouped_markets():
    matrix = score_matrix(1.4, 1.1)

    # Correct score probabilities
    cs = correct_score_probabilities(matrix)
    assert np.isclose(sum(cs.values()), 1.0)
    assert "0-0" in cs and "1-0" in cs and "2-1" in cs
    assert cs["1-0"] > 0

    # Truncated correct score with "other"
    cs_trunc = correct_score_probabilities(matrix, max_home=2, max_away=2)
    assert "other" in cs_trunc
    assert np.isclose(sum(cs_trunc.values()), 1.0)

    # Grouped markets
    grouped = grouped_score_markets(matrix)
    assert np.isclose(grouped["btts_yes"] + grouped["btts_no"], 1.0)
    assert grouped["double_chance_1x"] > grouped["home_win_to_nil"]
    assert grouped["scoreless_draw"] == pytest.approx(cs["0-0"])
