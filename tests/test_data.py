from datetime import date, datetime, timezone

import pytest

from parlay.data.normalization import Team, TeamRegistry
from parlay.data.schemas import Match, OddsSnapshot
from parlay.data.validation import validate_matches, validate_odds
from parlay.data.loaders import _match_id


def match(**kwargs):
    values = dict(
        match_id="m1", date=date(2025, 1, 1), competition="league",
        season="2024", home_team="A", away_team="B", home_goals=1,
        away_goals=0,
    )
    values.update(kwargs)
    return Match(**values)


def test_team_alias_resolves_to_canonical_id():
    registry = TeamRegistry([Team("a", "Alpha")], {"A FC": "a"})
    assert registry.resolve("a fc") == "a"


def test_validation_rejects_duplicate_and_negative_score():
    with pytest.raises(ValueError, match="duplicate"):
        validate_matches([match(home_goals=-1), match(home_goals=-1)])


def test_odds_cannot_be_after_forecast_date():
    row = OddsSnapshot("m1", "book", "1x2", "home", 2.0,
                       datetime(2025, 1, 2, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="after forecast"):
        validate_odds([row], as_of=date(2025, 1, 1))


def test_match_id_is_independent_of_source_path():
    row = {"Date": "01/01/25", "HomeTeam": "A", "AwayTeam": "B"}
    assert _match_id(row, "league", "2024") == _match_id(row, "league", "2024")
