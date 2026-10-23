from datetime import datetime, timedelta, timezone

import pytest

from parlay.betting.value import ValueConfig
from parlay.betting.watch import WatchConfig, Watcher, format_slot_message, kickoff_slots
from parlay.cli import main
from parlay.data.database import ResearchDatabase
from parlay.data.odds_sync import SyncResult, credit_cost
from parlay.data.schemas import Match, OddsSnapshot
from parlay.notify import TelegramNotifier

KICKOFF = datetime(2030, 1, 5, 15, tzinfo=timezone.utc)


def _match(match_id, kickoff, competition="EPL", home="A", away="B", goals=None):
    return Match(
        match_id, kickoff.date(), competition, "2029-30", home, away, goals, goals, kickoff_at=kickoff
    )


def _prices(match_id, at, soft_home=2.25):
    rows = []
    for book, values in {"Pinnacle": (2.0, 3.6, 4.0), "1xBet": (soft_home, 3.3, 3.5)}.items():
        rows += [
            OddsSnapshot(match_id, book, "1x2", s, p, at)
            for s, p in zip(("home", "draw", "away"), values, strict=True)
        ]
    return rows


class FakeSync:
    """Stands in for The Odds API: stores fresh prices and reports remaining credits."""

    def __init__(self, credits=100, soft_home=2.25, fail=False):
        self.calls = []
        self.credits = credits
        self.soft_home = soft_home
        self.fail = fail

    def __call__(self, db, api_key, competitions, *, now, regions, markets):
        if self.fail:
            raise OSError("network down")
        self.calls.append((tuple(competitions), now))
        self.credits -= credit_cost(markets, regions, len(competitions))
        ids = [m.match_id for m in db.load_matches() if m.home_goals is None]
        db.insert_odds([row for mid in ids for row in _prices(mid, now, self.soft_home)])
        return SyncResult(len(ids), 0, ("1xBet", "Pinnacle"), self.credits)


class Collect:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)


@pytest.fixture
def db():
    database = ResearchDatabase()
    database.insert_future_matches(
        [
            _match("m1", KICKOFF),
            _match("m2", KICKOFF, "Championship", "C", "D"),
            _match("m3", KICKOFF + timedelta(hours=2)),
            _match("la", KICKOFF, "La Liga", "E", "F"),
        ]
    )
    return database


def test_slots_group_by_kickoff_and_filter_competitions(db):
    slots = kickoff_slots(
        db.load_matches(), after=KICKOFF - timedelta(days=1), competitions=("EPL", "Championship")
    )
    assert [s.kickoff for s in slots] == [KICKOFF, KICKOFF + timedelta(hours=2)]
    assert slots[0].competitions == ("Championship", "EPL") and len(slots[0].matches) == 2
    assert slots[0].trigger(50) == KICKOFF - timedelta(minutes=50)


def test_watcher_runs_each_due_slot_once_and_logs_paper_bets(db):
    sync, notes = FakeSync(), Collect()
    watcher = Watcher(
        db, "KEY", WatchConfig(value=ValueConfig(bookmakers=("1xBet",))), notifier=notes, sync=sync
    )
    assert watcher.run_due(KICKOFF - timedelta(minutes=60)) == []  # too early
    assert watcher.next_trigger(KICKOFF - timedelta(minutes=60)) == KICKOFF - timedelta(minutes=50)
    reports = watcher.run_due(KICKOFF - timedelta(minutes=45))
    assert len(reports) == 1 and reports[0].status == "ok"
    assert sync.calls[0][0] == ("Championship", "EPL")
    assert {b.match_id for b in reports[0].bets} == {"m1", "m2"}  # only the slot's matches
    assert len(watcher.ledger.bets(kind="paper")) == 2
    assert "2 value bets" in notes.messages[0] and "1xBet" in notes.messages[0]
    assert watcher.run_due(KICKOFF - timedelta(minutes=40)) == []  # already processed
    assert watcher.run_due(KICKOFF - timedelta(minutes=5)) == []  # second slot not due yet
    assert len(sync.calls) == 1


def test_watcher_skips_late_slots_and_protects_credits(db):
    sync = FakeSync(credits=23)
    watcher = Watcher(db, "KEY", WatchConfig(credit_reserve=20), sync=sync)
    assert watcher.run_due(KICKOFF - timedelta(minutes=5)) == []  # inside min-lead: too late to act
    first = watcher.run_due(KICKOFF - timedelta(minutes=30))[0]
    assert first.credits_remaining == 19
    second = watcher.run_due(KICKOFF + timedelta(hours=2) - timedelta(minutes=30))[0]
    assert second.status == "skipped_credits" and len(sync.calls) == 1


def test_sync_failure_is_reported_not_raised(db):
    notes = Collect()
    watcher = Watcher(db, "KEY", WatchConfig(), notifier=notes, sync=FakeSync(fail=True))
    report = watcher.run_due(KICKOFF - timedelta(minutes=30))[0]
    assert report.status == "sync_failed" and "network down" in notes.messages[0]


def test_empty_slots_are_silent_unless_requested(db):
    notes = Collect()
    watcher = Watcher(db, "KEY", WatchConfig(), notifier=notes, sync=FakeSync(soft_home=1.9))
    assert watcher.run_due(KICKOFF - timedelta(minutes=30))[0].bets == []
    assert notes.messages == []
    report = watcher.run_due(KICKOFF + timedelta(hours=2) - timedelta(minutes=30))[0]
    assert "0 value bets" in format_slot_message(report, 1000)


def test_watch_config_validation():
    with pytest.raises(ValueError):
        WatchConfig(lead_minutes=10, min_lead_minutes=20)


def test_telegram_notifier_posts_message(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        captured["url"], captured["data"] = request.full_url, request.data
        return Response()

    monkeypatch.setattr("parlay.notify.urlopen", fake_urlopen)
    TelegramNotifier("TOKEN", "42").send("hello")
    assert captured["url"].endswith("/botTOKEN/sendMessage") and b"chat_id=42" in captured["data"]
    with pytest.raises(ValueError):
        TelegramNotifier("", "42")


def test_watch_cli_next_and_once(tmp_path, monkeypatch, capsys):
    path = tmp_path / "data.sqlite"
    database = ResearchDatabase(str(path))
    soon = datetime.now(timezone.utc) + timedelta(minutes=30)
    database.insert_future_matches([_match("m1", soon)])
    database.close()
    assert main(["watch", "--next", "--database", str(path)]) == 0
    assert "1 slots in the next 7 days" in capsys.readouterr().out

    import parlay.betting.watch as watch_module

    fake = FakeSync()
    monkeypatch.setattr(watch_module, "sync_the_odds_api", fake)
    monkeypatch.setenv("THE_ODDS_API_KEY", "KEY")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert main(["watch", "--once", "--database", str(path), "--books", "1xBet"]) == 0
    out = capsys.readouterr().out
    assert "terminal only" in out and "1 value bets" in out and len(fake.calls) == 1
    monkeypatch.delenv("THE_ODDS_API_KEY")
    with pytest.raises(SystemExit, match="THE_ODDS_API_KEY"):
        main(["watch", "--once", "--database", str(path)])
