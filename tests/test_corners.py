import numpy as np
import pytest

from conftest import make_match, synthetic_league
from parlay.evaluation.corners import run_corners_backtest, settle_total
from parlay.models.corners import fit_corner_strength
from parlay.prediction.markets import match_winner_counts, totals_probabilities


def test_corner_model_fits_overdispersed_counts():
    rows = synthetic_league(rounds=10, corners=True)
    model = fit_corner_strength(rows)
    assert model.fit_converged
    assert model.dispersion < 100  # simulated with NB dispersion 8
    assert model.attack["Lions"] > model.attack["Foxes"]
    matrix = model.corner_matrix("Lions", "Foxes")
    assert np.isclose(matrix.sum(), 1.0)
    home, away = model.expected_corners("Lions", "Foxes")
    assert home > away > 0


def test_corner_markets_and_quarter_lines():
    model = fit_corner_strength(synthetic_league(rounds=4, corners=True))
    matrix = model.corner_matrix("Hawks", "Bears")
    half = totals_probabilities(matrix, 10.5)
    assert half["push"] == 0.0 and np.isclose(half["over"] + half["under"], 1.0)
    quarter = totals_probabilities(matrix, 10.75)
    assert np.isclose(sum(quarter.values()), 1.0)
    winner = match_winner_counts(matrix)
    assert np.isclose(sum(winner.values()), 1.0)


def test_unseen_team_gets_league_average_and_no_data_falls_back():
    model = fit_corner_strength(synthetic_league(rounds=3, corners=True))
    assert model.expected_corners("Newcomer", "Lions")[0] > 0
    fallback = fit_corner_strength([make_match()])
    assert not fallback.fit_converged and "fallback" in fallback.fit_warning


def test_fixed_dispersion_changes_the_distribution():
    rows = synthetic_league(rounds=5, corners=True)
    poisson_like = fit_corner_strength(rows, dispersion=900.0).corner_matrix("Lions", "Foxes")
    dispersed = fit_corner_strength(rows, dispersion=3.0).corner_matrix("Lions", "Foxes")
    assert not np.allclose(poisson_like, dispersed, atol=1e-3)


@pytest.mark.parametrize(
    "total,line,expected", [(11, 10.5, "over"), (10, 10.0, "push"), (9, 10.5, "under"), (10, 10.25, "split")]
)
def test_settle_total(total, line, expected):
    assert settle_total(total, line) == expected


def test_corners_backtest_is_walk_forward():
    rows = synthetic_league(rounds=6, corners=True)
    result = run_corners_backtest(rows, line=10.5, initial_train_days=60, test_days=14)
    assert result.records
    assert result.metadata["market"] == "corners_total"
    assert result.metrics["records"] == len(result.records)
    assert result.metrics["n"] + result.metrics["pushes"] + result.metrics["split_settlements"] == len(
        result.records
    )
