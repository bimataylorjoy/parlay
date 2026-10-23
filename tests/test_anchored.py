import math
from datetime import date, timedelta

import numpy as np

from parlay.betting.anchored import FEATURES, AnchoredRow, build_rows, evaluate_anchored, fit_weights
from parlay.data.schemas import Match, OddsSnapshot


def _synthetic_rows(model_informative: bool, n_per_season=1500, seasons=4, seed=0):
    """True log-odds = market signal + hidden signal; the 'model' feature sees the hidden part (or noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(seasons):
        for i in range(n_per_season):
            base = np.array([0.45, 0.27, 0.28]) * np.exp(rng.normal(0, 0.3, 3))
            hidden = rng.normal(0, 0.35, 3)
            true = base * np.exp(hidden)
            true /= true.sum()
            market = base / base.sum()
            seen = hidden if model_informative else rng.normal(0, 0.35, 3)
            model = market * np.exp(seen)
            model /= model.sum()
            x = np.zeros((3, len(FEATURES)))
            x[:, 0] = np.log(model) - np.log(market)
            x[:, 4] = np.log(market)
            actual = int(rng.choice(3, p=true))
            best = 1.02 / market
            rows.append(
                AnchoredRow(
                    f"{s}-{i}", f"20{20 + s}-{21 + s}", date(2020 + s, 9, 1), actual, market, x, best, true
                )
            )
    return rows


def test_anchored_detects_real_information_and_rejects_noise():
    informative = evaluate_anchored(_synthetic_rows(True), l2=1.0)
    assert informative.pooled["improvement"] > 3 * informative.pooled["improvement_se"]
    assert all(w["model_edge"] > 0.5 for w in informative.weights.values())
    assert informative.pooled["anchored_mean_clv"] > informative.pooled["market_mean_clv"]

    noise = evaluate_anchored(_synthetic_rows(False), l2=1.0)
    assert noise.pooled["improvement"] < 2 * noise.pooled["improvement_se"]
    assert all(abs(w["model_edge"]) < 0.15 for w in noise.weights.values())


def test_zero_weights_reproduce_the_market():
    rows = _synthetic_rows(False, n_per_season=50, seasons=1)
    w = fit_weights(rows, l2=1e9)
    assert np.allclose(w, 0.0, atol=1e-4)


def test_build_rows_from_database_objects():
    matches, odds = [], []
    model = {}
    for i in range(3):
        day = date(2024, 1, 1) + timedelta(days=7 * i)
        m = Match(f"m{i}", day, "EPL", "2023-24", "A", "B", 1, 0)
        matches.append(m)
        model[m.match_id] = (0.5, 0.3, 0.2)
        at = __import__("datetime").datetime(
            2024, 1, 1, tzinfo=__import__("datetime").timezone.utc
        ) + timedelta(days=7 * i)
        for book, prices in {"Pinnacle": (2.0, 3.6, 4.0), "Bet365": (2.1, 3.3, 3.8)}.items():
            odds += [
                OddsSnapshot(m.match_id, book, "1x2", s, p, at)
                for s, p in zip(("home", "draw", "away"), prices, strict=True)
            ]
    rows = build_rows(matches, odds, model)
    assert len(rows) == 3
    row = rows[0]
    assert math.isclose(row.market.sum(), 1.0)
    assert row.best_odds[0] == 2.1 and row.actual == 0
    assert row.features.shape == (3, len(FEATURES))
