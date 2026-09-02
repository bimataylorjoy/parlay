from types import SimpleNamespace

import pytest

from parlay.evaluation.market import calibration_bins, evaluate_flat_stake


def record(actual, home=0.6, draw=0.2, away=0.2, odds_home=2.0, edge_home=0.1, ev_home=0.2):
    return SimpleNamespace(
        actual=actual, home_win=home, draw=draw, away_win=away,
        edge_home=edge_home, edge_draw=None, edge_away=None,
        ev_home=ev_home, ev_draw=None, ev_away=None,
        odds_home=odds_home, odds_draw=None, odds_away=None,
    )


def test_calibration_reports_empirical_frequency():
    rows = [record("home_win"), record("draw", home=0.4)]
    result = calibration_bins(rows, bins=5)
    home_bin = next(row for row in result if row["selection"] == "home" and row["bin_lower"] == 0.6)
    assert home_bin["count"] == 1
    assert home_bin["empirical_frequency"] == 1


def test_flat_stake_settles_win_and_loss():
    result = evaluate_flat_stake([record("home_win"), record("draw")], min_edge=0.05, min_ev=0.1)
    assert result["bets"] == 2
    assert result["profit"] == pytest.approx(0.0)
    assert result["yield"] == pytest.approx(0.0)


def test_flat_stake_returns_empty_when_threshold_not_met():
    result = evaluate_flat_stake([record("home", edge_home=0.01, ev_home=0.01)], min_edge=0.05)
    assert result["bets"] == 0


def test_evaluate_market_benchmark():
    from parlay.evaluation.market import evaluate_market_benchmark

    records = [
        SimpleNamespace(actual="home_win", market_home=0.5, market_draw=0.25, market_away=0.25),
        SimpleNamespace(actual="draw", market_home=0.4, market_draw=0.3, market_away=0.3),
        SimpleNamespace(actual="away_win", market_home=None, market_draw=None, market_away=None),
    ]

    mkt = evaluate_market_benchmark(records)
    assert mkt["n"] == 2.0
    assert mkt["log_loss"] > 0
    assert mkt["brier_score"] > 0
