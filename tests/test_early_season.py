from datetime import date

from parlay.data.schemas import Match
from parlay.models.team_strength import fit_team_strength


def test_early_season_shrinkage_reduces_extreme_strength():
    rows = [
        Match(str(i), date(2026, 8, 21 + i), "EPL", "2026-27", "A", "B", 5, 0)
        for i in range(2)
    ]
    raw = fit_team_strength(rows, half_life_days=None, early_season_shrinkage=0.0)
    shrunk = fit_team_strength(rows, half_life_days=None, early_season_shrinkage=0.75)
    raw_gap = abs(raw.attack["A"] - raw.attack["B"])
    shrunk_gap = abs(shrunk.attack["A"] - shrunk.attack["B"])
    assert shrunk_gap < raw_gap
    assert shrunk.shrinkage_weight == 0.25
