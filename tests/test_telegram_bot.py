import os
from datetime import datetime, timedelta, timezone

import pytest

from parlay.betting.watch import Slot, SlotReport
from parlay.config import load_env_file, read_env_file, set_env_value
from parlay.data.database import ResearchDatabase
from parlay.data.loaders import match_id_for
from parlay.data.schemas import Match, OddsSnapshot
from parlay.telegram.api import keyboard, split_message
from parlay.telegram.bot import ParlayBot, parse_amount, quick_amounts
from parlay.telegram.texts import describe_bet, money, when


class FakeAPI:
    def __init__(self, updates=None):
        self.sent = []
        self.updates = updates or []
        self.answered = []

    def send_message(self, chat_id, text, buttons=None):
        self.sent.append((str(chat_id), text, buttons))

    def answer_callback(self, callback_id, text=None):
        self.answered.append(callback_id)

    def get_updates(self, offset=None, timeout=25):
        batch, self.updates = self.updates, []
        return batch

    def get_me(self):
        return {"username": "testbot"}

    def set_commands(self, commands):
        self.commands = commands

    def texts(self):
        return "\n".join(text for _, text, _ in self.sent)

    def buttons(self):
        return [data for _, _, rows in self.sent if rows for row in rows for _, data in row]


def message(update_id, text, chat_id=42):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id, "type": "private", "first_name": "Bima"}, "text": text},
    }


def button(update_id, data, chat_id=42):
    return {
        "update_id": update_id,
        "callback_query": {"id": f"cb{update_id}", "data": data, "message": {"chat": {"id": chat_id}}},
    }


def seeded_db(path, n=2, soft_home=2.25):
    db = ResearchDatabase(str(path))
    now = datetime.now(timezone.utc)
    matches, odds = [], []
    for i in range(n):
        kickoff = now + timedelta(hours=5 + i)
        mid = match_id_for(kickoff.date(), f"Home{i}", f"Away{i}", "EPL")
        matches.append(
            Match(mid, kickoff.date(), "EPL", "x", f"Home{i}", f"Away{i}", None, None, kickoff_at=kickoff)
        )
        for book, values in {"Pinnacle": (2.0, 3.6, 4.0), "1xBet": (soft_home + 0.05 * i, 3.3, 3.5)}.items():
            for selection, price in zip(("home", "draw", "away"), values, strict=True):
                odds.append(OddsSnapshot(mid, book, "1x2", selection, price, now - timedelta(minutes=5)))
    db.insert_future_matches(matches)
    db.insert_odds(odds)
    db.close()
    return matches


@pytest.fixture
def bot(tmp_path):
    seeded_db(tmp_path / "db.sqlite")
    return ParlayBot(FakeAPI(), {"42"}, database=str(tmp_path / "db.sqlite"), books="1xBet,Pinnacle")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("50000", (50000.0, None)),
        ("50.000", (50000.0, None)),
        ("50rb", (50000.0, None)),
        ("1jt", (1_000_000.0, None)),
        ("1,5jt", (1_500_000.0, None)),
        ("50rb@2,15", (50000.0, 2.15)),
        ("20000 @ 7.9", (20000.0, 7.9)),
        ("abc", None),
        ("50rb@1", None),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


def test_texts_are_plain_indonesian():
    assert money(1234567.4) == "Rp1.234.567" and money(-5000) == "-Rp5.000"
    assert describe_bet("Arsenal", "Everton", "1x2", "draw") == ("Seri", "1X2 · hasil akhir 90 menit")
    pick, rule = describe_bet("A", "B", "totals_2.5", "over")
    assert pick == "Over 2,5 gol" and "≥ 3" in rule
    pick, rule = describe_bet("A", "B", "totals_3", "under")
    assert "≤ 2" in rule and "uang kembali" in rule
    pick, rule = describe_bet("Arsenal", "Everton", "ah_-1.5", "home")
    assert pick == "Arsenal -1,5" and "selisih ≥ 2" in rule
    pick, rule = describe_bet("Arsenal", "Everton", "ah_-1.5", "away")
    assert pick == "Everton +1,5" and "kalah maksimal 1" in rule
    assert "WIB" in when(datetime(2030, 1, 4, 19, tzinfo=timezone.utc)) and "Sabtu" in when(
        datetime(2030, 1, 4, 19, tzinfo=timezone.utc)
    )


def test_quick_amounts_respect_minimum(monkeypatch):
    monkeypatch.setenv("PARLAY_MIN_STAKE", "5000")
    assert quick_amounts(316) == [5000, 10000, 50000, 100000]
    assert quick_amounts(23_400)[0] == 10000 and 23000 in quick_amounts(23_400)


def test_only_authorized_chats_and_callbacks_answered(bot):
    bot.api.updates = [message(1, "/start", chat_id=999), message(2, "/start"), button(3, "m:help")]
    assert bot.poll_once() == 3
    assert {chat for chat, _, _ in bot.api.sent} == {"42"}
    assert "Halo" in bot.api.sent[0][1] and "m:value" in bot.api.buttons()
    assert bot.api.answered == ["cb3"] and bot.offset == 4


def test_value_cards_and_placing_a_bet(bot):
    bot.on_button("42", "m:value")
    cards = [text for _, text, _ in bot.api.sent if "Pasang:" in text]
    assert len(cards) == 2
    assert "Menang Home1" in cards[0] and "Peluang menang" in cards[0] and "EV" in cards[0]
    key = next(d for d in bot.api.buttons() if d.startswith("p:"))[2:]
    bot.on_button("42", f"p:{key}")
    assert "Berapa taruhanmu" in bot.api.sent[-1][1]
    bot.on_text("42", "25rb@2,4")
    assert "Dicatat" in bot.api.sent[-1][1]
    real = bot.ledger.bets(kind="real")
    assert len(real) == 1 and real[0].stake == 25000 and real[0].odds == 2.4
    bot.on_button("42", f"b:{key}:10000")
    assert "sudah tercatat" in bot.api.sent[-1][1]


def test_simulation_bet(bot):
    bot.on_button("42", "m:value")
    key = next(d for d in bot.api.buttons() if d.startswith("s:"))[2:]
    bot.on_button("42", f"s:{key}")
    assert "simulasi" in bot.api.sent[-1][1] and len(bot.ledger.bets(kind="paper")) == 1


def test_parlay_slip_flow(bot):
    bot.on_button("42", "m:slip")
    assert "minimal 2" in bot.api.sent[-1][1]
    bot.on_button("42", "m:value")
    keys = [d[2:] for d in bot.api.buttons() if d.startswith("a:")]
    for key in keys:
        bot.on_button("42", f"a:{key}")
    bot.on_button("42", f"a:{keys[0]}")
    assert "sudah ada di slip" in bot.api.sent[-1][1]
    bot.on_button("42", "m:slip")
    card = bot.api.sent[-1][1]
    assert "Mix parlay 2 tim" in card and "Peluang tiket menang" in card
    assert "t:" in [d for _, _, rows in bot.api.sent[-1:] for row in rows for _, d in row]
    bot.on_button("42", "t:")
    bot.on_text("42", "20rb")
    assert "Tiket parlay #1 dicatat" in bot.api.sent[-1][1]
    assert bot.state.slip("42") == [] and len(bot.ledger.tickets(kind="real")) == 1


def test_negative_slip_says_do_not_bet(tmp_path):
    seeded_db(tmp_path / "db.sqlite", soft_home=1.8)
    bot = ParlayBot(FakeAPI(), {"42"}, database=str(tmp_path / "db.sqlite"), books="1xBet")
    bot.ledger.save_candidates([])
    from parlay.betting.value import ValueConfig, find_value_bets

    now = datetime.now(timezone.utc)
    bets = find_value_bets(
        bot.db.load_matches(), bot.db.load_odds(), ValueConfig(min_ev=-1, max_ev=1), as_of=now
    )
    bets = [b for b in bets if b.selection == "home" and b.bookmaker == "1xBet"]
    bot.ledger.save_candidates(bets)
    for b in bets:
        bot.state.add_to_slip("42", b.key)
    bot.on_button("42", "m:slip")
    assert "Jangan dipasang" in bot.api.sent[-1][1]
    assert "t:" not in [d for _, _, rows in bot.api.sent[-1:] for row in rows for _, d in row]


def test_bankroll_report_schedule_and_unknown_input(bot):
    bot.on_text("42", "/modal 2jt")
    assert bot.state.bankroll("42") == 2_000_000
    bot.on_button("42", "m:bankroll")
    bot.on_text("42", "500rb")
    assert bot.state.bankroll("42") == 500_000
    bot.on_button("42", "m:report")
    assert "Laporan taruhan" in bot.api.sent[-1][1]
    bot.on_button("42", "m:next")
    assert "Jadwal 7 hari" in bot.api.sent[-1][1] and "Home0" in bot.api.sent[-1][1]
    bot.on_text("42", "halo")
    assert "Pilih dari menu" in bot.api.sent[-1][1]
    bot.api.updates = [button(9, "p:vb:deadbeef")]
    bot.poll_once()
    assert "kedaluwarsa" in bot.api.sent[-1][1] and "m:value" in bot.api.buttons()


def test_sync_reports_credits(bot):
    bot.run = lambda argv: "Stored 2 upcoming matches\nThe Odds API credits remaining: 480"
    bot.on_button("42", "m:sync")
    assert any("Sisa kuota API bulan ini: 480" in t for _, t, _ in bot.api.sent)
    bot.run = lambda argv: "error: HTTP 401"
    bot.on_button("42", "m:sync")
    assert "Gagal memperbarui" in bot.api.sent[-1][1]


def test_watcher_alerts_become_cards(bot):
    from parlay.betting.value import ValueConfig, find_value_bets

    now = datetime.now(timezone.utc)
    matches = bot.db.load_matches()
    bets = find_value_bets(matches, bot.db.load_odds(), ValueConfig(bookmakers=("1xBet",)), as_of=now)
    report = SlotReport(Slot(matches[0].kickoff_at, tuple(matches)), "ok", bets, 480)
    bot.send_slot_report(report)
    assert "Harga terbaru sebelum kickoff" in bot.api.texts() and "p:" in "".join(bot.api.buttons())


def test_daily_maintenance_runs_once_per_day(bot):
    calls = []
    bot.run = lambda argv: calls.append(argv[0]) or ""
    morning = datetime(2030, 1, 5, 1, tzinfo=timezone.utc)  # 08:00 WIB
    assert bot.daily_maintenance(morning) is True
    assert bot.daily_maintenance(morning + timedelta(hours=2)) is False
    assert calls[0] == "fetch"
    assert (
        bot.daily_maintenance(datetime(2030, 1, 5, 23, 30, tzinfo=timezone.utc)) is True
    )  # 06:30 WIB next day


def test_keyboard_and_split():
    assert '"callback_data": "m:value"' in keyboard([[("Value", "m:value")]])
    assert keyboard(None) is None
    assert len(split_message("x\n" * 5000, limit=1000)) > 1


def test_env_file_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "env"
    path.write_text(
        '# secrets\nexport THE_ODDS_API_KEY=abc\nPARLAY_BOOKS="1xBet,Pinnacle"\n', encoding="utf-8"
    )
    assert read_env_file(path) == {"THE_ODDS_API_KEY": "abc", "PARLAY_BOOKS": "1xBet,Pinnacle"}
    monkeypatch.setenv("THE_ODDS_API_KEY", "already-set")
    load_env_file(path)
    assert os.environ["THE_ODDS_API_KEY"] == "already-set"
    set_env_value("TELEGRAM_CHAT_ID", "42", path)
    set_env_value("TELEGRAM_CHAT_ID", "43", path)
    assert read_env_file(path)["TELEGRAM_CHAT_ID"] == "43" and path.read_text().count("TELEGRAM_CHAT_ID") == 1
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_setup_cli(tmp_path, monkeypatch, capsys):
    import parlay.telegram.api as api_module
    from parlay.cli import main

    env = tmp_path / "env"
    monkeypatch.setenv("PARLAY_ENV_FILE", str(env))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    fake = FakeAPI([message(5, "/start")])
    monkeypatch.setattr(api_module, "TelegramAPI", lambda token: fake)
    assert main(["telegram", "--setup"]) == 0
    assert read_env_file(env)["TELEGRAM_CHAT_ID"] == "42" and "Halo" in fake.sent[0][1]
    monkeypatch.setattr(api_module, "TelegramAPI", lambda token: FakeAPI())
    assert main(["telegram", "--setup"]) == 1
    assert "send /start to @testbot" in capsys.readouterr().out


def test_bot_requires_token():
    from parlay.cli import main

    with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN"):
        main(["telegram"])
