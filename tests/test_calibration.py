from types import SimpleNamespace

import pytest
import numpy as np

from parlay.evaluation.calibration import temperature_scale, find_optimal_temperature, apply_calibration, temporally_safe_calibration


def test_temperature_scale_one_is_identity():
    probs = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}
    scaled = temperature_scale(probs, 1.0)
    assert scaled["home_win"] == pytest.approx(0.6)
    assert scaled["draw"] == pytest.approx(0.2)
    assert scaled["away_win"] == pytest.approx(0.2)


def test_temperature_scale_high_softens_probs():
    probs = {"home_win": 0.8, "draw": 0.1, "away_win": 0.1}
    scaled = temperature_scale(probs, 2.0)
    # The dominant probability should decrease
    assert scaled["home_win"] < 0.8
    # The uniform distribution is 0.333, so it moves towards that
    assert scaled["draw"] > 0.1
    assert np.isclose(sum(scaled.values()), 1.0)


def test_temperature_scale_low_sharpens_probs():
    probs = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}
    scaled = temperature_scale(probs, 0.5)
    # The dominant probability should increase
    assert scaled["home_win"] > 0.5
    # The others should decrease
    assert scaled["draw"] < 0.25
    assert np.isclose(sum(scaled.values()), 1.0)


def test_find_optimal_temperature_reduces_loss():
    # Simulate an overconfident model: it predicts 90% but only wins 60% of the time.
    # It predicts 90% home win, but actual is away win often.
    records = []
    # 6 times home wins, but we gave it 0.9
    for _ in range(6):
        records.append(SimpleNamespace(home_win=0.9, draw=0.05, away_win=0.05, actual="home_win"))
    # 4 times home loses, but we still gave it 0.9
    for _ in range(4):
        records.append(SimpleNamespace(home_win=0.9, draw=0.05, away_win=0.05, actual="away_win"))
        
    t_opt = find_optimal_temperature(records)
    # Since model is overconfident, T should be > 1 to soften probs
    assert t_opt > 1.0
    
    # Test apply calibration
    calibrated = apply_calibration(records, t_opt)
    
    from parlay.evaluation.metrics import aggregate_scores, log_loss
    orig_loss = aggregate_scores([
        {"log_loss": log_loss({"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual),
         "brier_score": 0.0} 
        for r in records
    ])["log_loss"]
    
    calib_loss = aggregate_scores(calibrated)["log_loss"]
    
    # Calibration should strictly reduce or maintain loss
    assert calib_loss < orig_loss


def test_temporal_calibration_excludes_test_window():
    rows = []
    for i in range(15):
        rows.append(SimpleNamespace(
            forecast_timestamp=f"2024-01-{i + 1:02d}T00:00:00+00:00",
            home_win=0.8, draw=0.1, away_win=0.1,
            actual="home_win" if i % 2 == 0 else "away_win",
        ))
    temperature, calibrated, metrics = temporally_safe_calibration(rows, train_end="2024-01-05T00:00:00+00:00", calibrate_end="2024-01-10T00:00:00+00:00")
    assert temperature > 0
    assert calibrated
    assert metrics["n"] == len(calibrated)
    assert all(row["forecast_timestamp"] > "2024-01-10T00:00:00+00:00" for row in calibrated)
