from datetime import date

import pytest

from conftest import synthetic_league
from parlay.models.corners import fit_corner_strength
from parlay.models.team_strength import fit_team_strength
from parlay.prediction.fixture import Fixture, MarketQuote, PredictConfig, parse_odds_columns, predict_fixture


@pytest.fixture(scope="module")
def models():
    rows = synthetic_league(rounds=8, corners=True)
    return fit_team_strength(rows), fit_corner_strength(rows)


def test_parse_odds_columns_reads_all_supported_markets():
    row = {
        "B365H": "2.0", "B365D": "3.4", "B365A": "3.9",
        "B365>2.75": "2.1", "B365<2.75": "1.75",
        "B365BTTS_Y": "1.8", "B365BTTS_N": "",
        "B365CRN>10.5": "1.9", "B365C>9.5": "1.8", "B365C>2.5": "1.6",
        "AHh": "-0.25", "B365AHH": "1.95", "B365AHA": "1.95",
        "B3651X": "1.3", "PSH": "2.1",
    }  # fmt: skip
    odds = parse_odds_columns(row)
    assert odds[("1x2", "home", None)] == 2.0
    assert odds[("totals", "over", 2.75)] == 2.1
    assert odds[("btts", "yes", None)] == 1.8
    assert ("btts", "no", None) not in odds
    assert odds[("corners_totals", "over", 10.5)] == 1.9
    assert odds[("corners_totals", "over", 9.5)] == 1.8
    assert ("corners_totals", "over", 2.5) not in odds  # football-data closing goal total, not corners
    assert odds[("asian_handicap", "home", -0.25)] == 1.95
    assert odds[("double_chance", "1x", None)] == 1.3
    assert all(key[0] != "1x2" or value != 2.1 for key, value in odds.items())  # other prefixes ignored


def test_parse_odds_columns_converts_formats_and_prefixes():
    odds = parse_odds_columns({"PSH": "-1.25", "PSD": "2.4", "PSA": "3.0"}, prefix="PS", odds_format="indo")
    assert odds[("1x2", "home", None)] == pytest.approx(1.8)
    assert odds[("1x2", "draw", None)] == pytest.approx(3.4)


def test_market_quote_fair_odds_and_edge():
    quote = MarketQuote(
        "totals", "over", 3.0, win=0.45, push=0.2, loss=0.35, odds=2.0, market_probability=0.5
    )
    assert quote.fair_odds == pytest.approx(1 + 0.35 / 0.45)
    assert quote.expected_value == pytest.approx(0.45 - 0.35)
    assert quote.decisive_probability == pytest.approx(0.45 / 0.8)
    assert quote.edge == pytest.approx(0.45 / 0.8 - 0.5)
    assert quote.stake > 0
    assert quote.name == "totals:over 3"


def test_predict_fixture_covers_markets_and_selects_value(models):
    goals, corners = models
    fixture = Fixture(
        date(2024, 1, 1),
        "Lions",
        "Foxes",
        odds={("1x2", "home", None): 5.0, ("1x2", "draw", None): 4.0, ("1x2", "away", None): 1.6},
    )
    prediction = predict_fixture(fixture, goals, corners, PredictConfig(min_ev=0.02))
    assert prediction.ok
    assert sum(prediction.outcome.values()) == pytest.approx(1.0)
    assert set(prediction.totals) == set(PredictConfig().goal_lines)
    assert prediction.expected_corners is not None and prediction.corner_totals
    home_quote = next(q for q in prediction.quotes if q.market == "1x2" and q.selection == "home")
    assert home_quote.market_probability is not None and home_quote.edge is not None
    assert prediction.value_bets and prediction.value_bets[0].name == "1x2:home"
    assert all(q.expected_value >= 0.02 for q in prediction.value_bets)
    row = prediction.to_row()
    assert row["status"] == "ok" and row["recommendation"] == "1x2:home"
    assert row["over_2_5_prob"] + row["under_2_5_prob"] == pytest.approx(1.0, abs=1e-3)


def test_unknown_team_is_reported_not_raised(models):
    goals, corners = models
    prediction = predict_fixture(Fixture(date(2024, 1, 1), "Lions", "Nobody"), goals, corners)
    assert prediction.status == "unknown_team"
    assert "Nobody" in prediction.message
    assert prediction.to_row()["status"] == "unknown_team"


def test_no_value_bets_without_prices(models):
    goals, _ = models
    prediction = predict_fixture(Fixture(date(2024, 1, 1), "Hawks", "Bears"), goals)
    assert prediction.value_bets == []
    assert prediction.expected_corners is None
    assert "data_missingness" in prediction.diagnostics.anomaly_flags
