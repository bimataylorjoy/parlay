from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import make_match, simple_series, synthetic_league
from parlay.data.database import ResearchDatabase
from parlay.data.schemas import OddsSnapshot
from parlay.evaluation.backtest import BacktestConfig, forecast_timestamp, run_backtest
from parlay.evaluation.compare import compare_models
from parlay.evaluation.metrics import (
    aggregate_scores,
    binary_log_loss,
    brier_score,
    log_loss,
    outcome_label,
    ranked_probability_score,
)
from parlay.evaluation.tuning import grid_search, nested_tuning

FAST = BacktestConfig(model="poisson", initial_train_days=5, test_days=2, bookmaker="book")


def test_scoring_rules():
    probabilities = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}
    assert outcome_label(2, 0) == "home_win" and outcome_label(0, 0) == "draw"
    assert log_loss(probabilities, "home_win") == pytest.approx(np.log(2))
    assert brier_score(probabilities, "home_win") == pytest.approx(0.375)
    # RPS penalizes "near misses" less: draw is closer to a home win than an away win
    assert ranked_probability_score(
        {"home_win": 0.0, "draw": 1.0, "away_win": 0.0}, "home_win"
    ) == pytest.approx(0.5)
    assert ranked_probability_score(
        {"home_win": 0.0, "draw": 0.0, "away_win": 1.0}, "home_win"
    ) == pytest.approx(1.0)
    assert binary_log_loss(0.8, False) == pytest.approx(-np.log(0.2))
    assert aggregate_scores([{"log_loss": 1.0, "brier_score": 0.5, "rps": None}])["n"] == 1.0
    with pytest.raises(ValueError):
        log_loss({"home_win": 0.5, "draw": 0.5}, "draw")


def test_backtest_produces_scored_records_and_persists_them():
    rows = simple_series(14)
    result = run_backtest(rows, config=FAST)
    assert result.records
    assert result.metrics["n"] == len(result.records) == sum(f.n for f in result.fold_metrics)
    assert all(np.isfinite(r.log_loss) and 0 <= r.rps <= 1 for r in result.records)
    for fold in result.fold_metrics:
        assert fold.train_end < fold.test_start
    db = ResearchDatabase()
    db.insert_matches(rows)
    db.insert_predictions(result.records)
    assert db.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == len(result.records)


def test_training_never_contains_the_forecast_match(monkeypatch):
    import parlay.evaluation.backtest as backtest

    seen = []
    original = backtest.fit_team_strength

    def spy(train, **kwargs):
        seen.append((max(m.date for m in train), kwargs["as_of"]))
        return original(train, **kwargs)

    monkeypatch.setattr(backtest, "fit_team_strength", spy)
    result = run_backtest(simple_series(14), config=FAST)
    first_test = min(r.match_date for r in result.records)
    assert all(latest <= as_of for latest, as_of in seen)
    assert max(latest for latest, _ in seen).isoformat() < max(r.match_date for r in result.records)
    assert seen[0][0].isoformat() < first_test


def _odds(match_id, day, prices, *, hour=0, bookmaker="book", closing=False, market="1x2"):
    captured = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=hour)
    selections = ("home", "draw", "away") if market == "1x2" else ("over", "under")
    return [
        OddsSnapshot(match_id, bookmaker, market, s, p, captured, closing)
        for s, p in zip(selections, prices, strict=True)
    ]


def test_market_fields_edges_and_closing_benchmark():
    rows = simple_series(10)
    odds = []
    for m in rows:
        odds += _odds(m.match_id, m.date, (2.0, 4.0, 4.0))
        odds += _odds(m.match_id, m.date, (1.8, 4.2, 4.5), hour=12, bookmaker="Pinnacle", closing=True)
        odds += _odds(m.match_id, m.date, (1.9, 1.9), market="totals_2.5")
    result = run_backtest(rows, odds, FAST)
    record = result.records[0]
    assert record.market_home == pytest.approx(0.5)
    assert record.odds_home == 2.0
    assert record.edge_home == pytest.approx(record.home_win - 0.5)
    assert record.ev_home == pytest.approx(record.home_win * 2.0 - 1.0)
    assert record.closing_home is not None and record.closing_home > 0.5
    assert record.market_over_2_5 == pytest.approx(0.5)
    for key in ("market_log_loss", "closing_log_loss", "market_over_2_5_log_loss", "over_2_5_log_loss"):
        assert key in result.metrics


def test_prices_after_the_forecast_cutoff_are_never_used():
    kickoff = datetime(2020, 1, 8, 20, tzinfo=timezone.utc)
    rows = [
        make_match(str(i), date(2020, 1, 1) + timedelta(days=i), "A", "B", i % 3, (i + 1) % 2,
                   kickoff_at=kickoff if i == 7 else None)
        for i in range(10)
    ]  # fmt: skip
    odds = _odds("7", date(2020, 1, 8), (2.0, 3.5, 4.0), hour=18) + _odds(
        "7", date(2020, 1, 8), (1.5, 4.0, 6.0), hour=19, closing=False
    )
    odds[-3:] = [replace(o, captured_at=datetime(2020, 1, 8, 19, 30, tzinfo=timezone.utc)) for o in odds[-3:]]
    result = run_backtest(rows, odds, replace(FAST, forecast_lead_minutes=60))
    target = next(r for r in result.records if r.match_id == "7")
    assert target.forecast_timestamp == "2020-01-08T19:00:00+00:00"
    assert target.odds_home == 2.0  # the 19:30 price is after kickoff - 60m


def test_closing_prices_are_not_used_as_pre_match_prices():
    rows = simple_series(10)
    odds = [o for m in rows for o in _odds(m.match_id, m.date, (1.8, 4.2, 4.5), closing=True)]
    result = run_backtest(rows, odds, FAST)
    assert all(r.odds_home is None for r in result.records)


def test_competition_filter_scores_only_selected_competition():
    rows = simple_series(14)
    rows = [replace(m, competition="Championship") if i % 2 else m for i, m in enumerate(rows)]
    result = run_backtest(rows, config=replace(FAST, competitions=("EPL",)))
    assert result.records and {r.competition for r in result.records} == {"EPL"}


def test_unseen_teams_are_counted_as_skipped():
    rows = [*simple_series(10), make_match("new", date(2020, 1, 9), "C", "A", 1, 1)]
    result = run_backtest(rows, config=FAST)
    assert result.metrics["skipped_unknown_team"] >= 1


def test_temperature_calibration_uses_only_earlier_forecasts():
    result = run_backtest(
        synthetic_league(rounds=4), config=replace(FAST, test_days=7, calibration_mode="temperature")
    )
    assert result.records
    assert result.metadata["calibration_window"] == "prior_predictions_only"


def test_forecast_timestamp_policy():
    assert forecast_timestamp(make_match(), 60).isoformat() == "2024-01-01T00:00:00+00:00"
    kicked = make_match(kickoff_at=datetime(2024, 1, 1, 15, tzinfo=timezone.utc))
    assert forecast_timestamp(kicked, 90).hour == 13


def test_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(forecast_lead_minutes=-1)
    with pytest.raises(ValueError):
        BacktestConfig(evaluation_mode="random")
    with pytest.raises(ValueError):
        BacktestConfig(calibration_mode="isotonic")


def test_regime_metadata_is_reported():
    result = run_backtest(synthetic_league(rounds=4), config=replace(FAST, test_days=7))
    regimes = result.metadata["regimes"]
    assert "EPL" in regimes["competition"]
    assert any(key.startswith("established") or key.startswith("limited") for key in regimes["team_history"])


def test_compare_runs_all_models_on_identical_folds():
    rows = simple_series(14)
    odds = [o for m in rows for o in _odds(m.match_id, m.date, (2.0, 4.0, 4.0))]
    results, summary = compare_models(rows, odds, FAST)
    assert set(results) == {"poisson", "dixon_coles", "negative_binomial"}
    folds = {model: [f.test_start for f in r.fold_metrics] for model, r in results.items()}
    assert len({tuple(v) for v in folds.values()}) == 1
    assert summary[-1]["model"] == "market_book"
    for row in summary[:-1]:
        stats = row["stability_log_loss"]
        assert stats["min"] <= stats["mean"] <= stats["max"] and row["per_fold"]


def test_grid_search_and_nested_tuning():
    rows = simple_series(40)
    grid = grid_search(rows, FAST, half_lives=(5.0, None), l2_grid=(0.0, 5.0))
    assert len(grid) == 4
    assert grid[0]["log_loss"] <= grid[-1]["log_loss"]
    report = nested_tuning(rows, FAST, half_lives=(5.0, None), l2_grid=(1.0,), holdout_days=10)
    assert report["holdout_metrics"]["n"] > 0
    assert report["evaluated_on"].startswith("holdout >")
    assert set(report["hyperparameters"]) == {"half_life_days", "l2_reg"}


def test_records_serialize_to_flat_dicts():
    result = run_backtest(simple_series(10), config=FAST)
    row = asdict(result.records[0])
    assert {"match_id", "home_win", "rps", "over_2_5", "closing_home"} <= set(row)
