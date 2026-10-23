from datetime import date, datetime, timezone

import pytest

from conftest import make_match
from parlay.data.loaders import load_football_data_csv, match_id_for, season_for
from parlay.data.normalization import Team, TeamRegistry
from parlay.data.schemas import OddsSnapshot
from parlay.data.teams import canonical_team_name
from parlay.data.validation import validate_matches, validate_odds

HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,HC,AC,"
    "B365H,B365D,B365A,PSCH,PSCD,PSCA,B365>2.5,B365<2.5"
)


def test_team_registry_resolves_aliases_to_ids():
    registry = TeamRegistry([Team("a", "Alpha")], {"A FC": "a"})
    assert registry.resolve("a fc") == "a"
    assert registry.canonical_name("A FC") == "Alpha"
    with pytest.raises(KeyError):
        registry.resolve("Beta")
    with pytest.raises(ValueError):
        TeamRegistry([Team("a", "Alpha")], {"x": "missing"})


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("Manchester City", "Man City"),
        ("  manchester   united ", "Man United"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("Brighton & Hove Albion", "Brighton"),
        ("Wolverhampton Wanderers", "Wolves"),
        ("Queens Park Rangers", "QPR"),
        ("Arsenal", "Arsenal"),
    ],
)
def test_canonical_team_names(raw, canonical):
    assert canonical_team_name(raw) == canonical


def test_validation_rejects_duplicates_negative_scores_and_bad_odds():
    with pytest.raises(ValueError, match="duplicate"):
        validate_matches([make_match(home_goals=-1), make_match(home_goals=-1)])
    with pytest.raises(ValueError, match="identical"):
        validate_matches([make_match(home="A", away="A")])
    late = OddsSnapshot("m1", "book", "1x2", "home", 2.0, datetime(2025, 1, 2, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="after forecast"):
        validate_odds([late], as_of=date(2025, 1, 1))


def test_season_label_and_stable_match_ids():
    assert season_for(date(2025, 8, 16)) == "2025-26"
    assert season_for(date(2026, 5, 10)) == "2025-26"
    first = match_id_for(date(2025, 1, 1), "Arsenal", "Chelsea", "EPL")
    assert first == match_id_for(date(2025, 1, 1), "arsenal", "chelsea", "EPL")
    assert first != match_id_for(date(2025, 1, 1), "Arsenal", "Chelsea", "Championship")


def test_loader_reads_results_stats_and_odds(tmp_path):
    path = tmp_path / "E0.csv"
    path.write_text(
        HEADER
        + "\nE0,16/08/2025,20:00,Man City,Nottingham Forest,2,1,14,9,6,3,7,2,"
        + "1.30,5.5,9.0,1.28,6.0,10.0,1.6,2.3\n",
        encoding="utf-8",
    )
    matches, odds = load_football_data_csv(path, competition="EPL")
    match = matches[0]
    assert (match.home_team, match.away_team) == ("Man City", "Nott'm Forest")
    assert match.season == "2025-26"
    assert (match.home_sot, match.away_corners, match.home_shots) == (6, 2, 14)
    assert match.kickoff_at == datetime(2025, 8, 16, 20, tzinfo=timezone.utc)
    by_key = {(o.bookmaker, o.market, o.is_closing) for o in odds}
    assert by_key == {("Bet365", "1x2", False), ("Pinnacle", "1x2", True), ("Bet365", "totals_2.5", False)}
    closing = [o for o in odds if o.is_closing]
    assert all(o.captured_at == match.kickoff_at for o in closing)
    assert all(o.captured_at < match.kickoff_at for o in odds if not o.is_closing)


def test_loader_skips_malformed_prices_instead_of_failing(tmp_path, caplog):
    path = tmp_path / "E1.csv"
    path.write_text(
        HEADER
        + "\nE1,02/02/2019,15:00,Brentford,Hull,1,1,,,,,,,0,3.4,3.2,,,,1.9,1.9\n,,,,,,,,,,,,,,,,,,,,\n",
        encoding="utf-8",
    )
    matches, odds = load_football_data_csv(path, competition="Championship")
    assert len(matches) == 1
    assert {o.market for o in odds} == {"totals_2.5"}
    assert "skipped 1" in caplog.text


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,HomeTeam,AwayTeam\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_football_data_csv(path)
