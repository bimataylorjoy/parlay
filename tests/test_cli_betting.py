from datetime import date, datetime, timedelta, timezone

import pytest

from parlay.cli import main
from parlay.data.database import ResearchDatabase
from parlay.data.loaders import match_id_for
from parlay.data.schemas import Match, OddsSnapshot
from test_odds_feeds import EVENT, FIXTURES


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "data.sqlite"
    ResearchDatabase(str(path)).close()
    return path


def test_odds_sync_from_football_data_and_the_odds_api(tmp_path, database, monkeypatch, capsys):
    import parlay.data.odds_sync as odds_sync
    import parlay.data.the_odds_api as api

    fixtures = tmp_path / "fixtures.csv"
    fixtures.write_text(FIXTURES, encoding="utf-8")
    monkeypatch.setattr(odds_sync, "acquire_csv", lambda url, directory: (fixtures, {}))
    assert main(["odds", "sync", "--database", str(database)]) == 0
    assert "no Pinnacle prices" in capsys.readouterr().out

    future = dict(EVENT, commence_time=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat())
    past = dict(EVENT, id="old", commence_time="2001-01-01T15:00:00Z")
    monkeypatch.setattr(api, "fetch_odds", lambda *a, **k: ([future, past], api.Quota(490, 10, 2)))
    monkeypatch.setenv("THE_ODDS_API_KEY", "KEY")
    assert (
        main(
            ["odds", "sync", "--source", "the-odds-api", "--competition", "EPL", "--database", str(database)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "credits remaining: 490" in out and "Stored 1 upcoming" in out
    db = ResearchDatabase(str(database))
    assert {o.bookmaker for o in db.load_odds()} >= {"Pinnacle", "Bet365", "Betfair Exchange"}


def test_odds_sync_without_api_key_is_a_clear_error(database, monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="THE_ODDS_API_KEY"):
        main(["odds", "sync", "--source", "the-odds-api", "--database", str(database)])


def _seed_upcoming(path, kickoff):
    db = ResearchDatabase(str(path))
    match_id = match_id_for(kickoff.date(), "A", "B", "EPL")
    db.insert_future_matches(
        [Match(match_id, kickoff.date(), "EPL", "2030-31", "A", "B", None, None, kickoff_at=kickoff)]
    )
    at = kickoff - timedelta(hours=5)
    prices = {"Pinnacle": (2.0, 3.6, 4.0), "Bet365": (2.25, 3.3, 3.5)}
    db.insert_odds(
        [
            OddsSnapshot(match_id, book, "1x2", sel, price, at)
            for book, values in prices.items()
            for sel, price in zip(("home", "draw", "away"), values, strict=True)
        ]
    )
    db.close()
    return match_id


def test_value_paper_bets_add_update_report(tmp_path, database, capsys):
    kickoff = datetime(2030, 1, 5, 15, tzinfo=timezone.utc)
    match_id = _seed_upcoming(database, kickoff)
    as_of = (kickoff - timedelta(hours=2)).isoformat()
    out_csv = tmp_path / "value.csv"
    assert (
        main(["value", "--database", str(database), "--as-of", as_of, "--paper", "--output", str(out_csv)])
        == 0
    )
    out = capsys.readouterr().out
    assert "1 value bets" in out and "Recorded 1 new paper bets" in out and "A vs B: A" in out
    key = out_csv.read_text().splitlines()[1].split(",")[0]
    assert key.startswith("vb:")

    assert main(["bets", "add", key, "--stake", "10", "--odds", "2.2", "--database", str(database)]) == 0
    assert "Recorded real bet" in capsys.readouterr().out
    manual = ["--match-id", match_id, "--market", "1x2", "--selection", "draw", "--bookmaker", "Bet365"]
    assert main(["bets", "add", "--stake", "5", "--database", str(database), *manual, "--odds", "3.3"]) == 0
    with pytest.raises(SystemExit, match="--market"):
        main(["bets", "add", "--stake", "5", "--database", str(database), "--match-id", match_id])

    db = ResearchDatabase(str(database))
    db.insert_odds([OddsSnapshot(match_id, "Pinnacle", "1x2", s, p, kickoff, True)
                    for s, p in zip(("home", "draw", "away"), (1.9, 3.7, 4.4), strict=True)])  # fmt: skip
    db.insert_matches([Match(match_id, kickoff.date(), "EPL", "2030-31", "A", "B", 1, 0, kickoff_at=kickoff)])
    db.close()
    assert main(["bets", "update", "--database", str(database)]) == 0
    assert "Closing prices attached to 3 bets; 3 bets settled" in capsys.readouterr().out
    assert main(["bets", "report", "--database", str(database), "--kind", "real"]) == 0
    report = capsys.readouterr().out
    assert "Closing-line value" in report and "Not enough bets" in report
    assert main(["bets", "list", "--database", str(database)]) == 0
    assert "win" in capsys.readouterr().out


def test_value_warns_when_reference_is_not_pinnacle(database, capsys):
    _seed_upcoming(database, datetime(2030, 1, 5, 15, tzinfo=timezone.utc))
    assert (
        main(
            ["value", "--database", str(database), "--reference", "consensus", "--as-of", "2030-01-05T12:00"]
        )
        == 0
    )
    assert "low confidence" in capsys.readouterr().out


def test_backtest_value_cli(tmp_path, database, capsys):
    db = ResearchDatabase(str(database))
    for i in range(4):
        day = date(2024, 1, 1) + timedelta(days=i)
        m = Match(f"g{i}", day, "EPL", "2023-24", "A", "B", 1, 0)
        db.insert_matches([m])
        pre = datetime.combine(day, datetime.min.time(), timezone.utc)
        db.insert_odds([OddsSnapshot(m.match_id, book, "1x2", s, p, pre)
                        for book, values in {"Pinnacle": (2.0, 3.6, 4.0), "Bet365": (2.3, 3.3, 3.5)}.items()
                        for s, p in zip(("home", "draw", "away"), values, strict=True)])  # fmt: skip
    db.close()
    out = tmp_path / "vbt"
    assert main(["backtest-value", "--database", str(database), "--output-dir", str(out)]) == 0
    assert "4 bets on 4 matches" in capsys.readouterr().out
    assert (out / "bets.csv").exists() and (out / "metrics.json").exists()


def test_paper_logging_records_each_selection_once(database, capsys):
    kickoff = datetime(2030, 1, 5, 15, tzinfo=timezone.utc)
    match_id = _seed_upcoming(database, kickoff)
    argv = ["value", "--database", str(database), "--as-of", "2030-01-05T12:00", "--paper"]
    assert main(argv) == 0
    db = ResearchDatabase(str(database))  # the price moves and the engine runs again
    db.insert_odds([OddsSnapshot(match_id, "Bet365", "1x2", s, p, kickoff - timedelta(hours=4))
                    for s, p in zip(("home", "draw", "away"), (2.3, 3.3, 3.4), strict=True)])  # fmt: skip
    db.close()
    assert main(argv) == 0
    assert "Recorded 0 new paper bets" in capsys.readouterr().out


def test_daily_runs_every_step_and_reports(tmp_path, database, monkeypatch, capsys):
    import parlay.data.odds_sync as odds_sync

    fixtures = tmp_path / "fixtures.csv"
    fixtures.write_text(FIXTURES, encoding="utf-8")
    monkeypatch.setattr(odds_sync, "acquire_csv", lambda url, directory: (fixtures, {}))
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    assert main(["daily", "--database", str(database), "--skip-results", "--league", "E0"]) == 0
    out = capsys.readouterr().out
    for title in (
        "Prices: football-data",
        "Closing-line value and settlement",
        "Value bets",
        "Paper-trading report",
    ):
        assert title in out
    assert "THE_ODDS_API_KEY is not set" in out
