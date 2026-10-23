import os
from datetime import datetime, timedelta, timezone

import pytest

from parlay.config import load_env_file, read_env_file, set_env_value
from parlay.data.database import ResearchDatabase
from parlay.data.loaders import match_id_for
from parlay.data.schemas import Match, OddsSnapshot
from parlay.telegram_bot import ParlayBot, find_chat_ids, parse_ticket_args, split_message


class FakeAPI:
    def __init__(self, updates=None):
        self.sent = []
        self.updates = updates or []

    def send_message(self, chat_id, text, *, preformatted=False):
        self.sent.append((str(chat_id), text, preformatted))

    def get_updates(self, offset=None, timeout=25):
        batch, self.updates = self.updates, []
        return batch

    def get_me(self):
        return {"username": "testbot"}


def update(update_id, text, chat_id=42, chat_type="private"):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id, "type": chat_type, "first_name": "Bima"}, "text": text},
    }


def make_bot(tmp_path, api=None, runs=None):
    runs = runs if runs is not None else []

    def run(argv):
        runs.append(list(argv))
        return "=== Mix parlays from value bets\nTicket at 1xBet" if argv[0] == "value" else "ok output"

    return ParlayBot(
        api or FakeAPI(), {"42"}, database=str(tmp_path / "db.sqlite"), books="1xBet", run_command=run
    ), runs


def test_env_file_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "env"
    path.write_text(
        '# secrets\nexport THE_ODDS_API_KEY=abc\nPARLAY_BOOKS="1xBet,Pinnacle"\n', encoding="utf-8"
    )
    assert read_env_file(path) == {"THE_ODDS_API_KEY": "abc", "PARLAY_BOOKS": "1xBet,Pinnacle"}
    monkeypatch.setenv("THE_ODDS_API_KEY", "already-set")
    load_env_file(path)
    assert os.environ["THE_ODDS_API_KEY"] == "already-set"  # never overridden
    assert os.environ["PARLAY_BOOKS"] == "1xBet,Pinnacle"
    set_env_value("TELEGRAM_CHAT_ID", "42", path)
    set_env_value("TELEGRAM_CHAT_ID", "43", path)
    assert read_env_file(path)["TELEGRAM_CHAT_ID"] == "43"
    assert path.read_text().count("TELEGRAM_CHAT_ID") == 1
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_split_message_escapes_and_chunks():
    chunks = split_message("a<b>\n" * 3000, preformatted=True, limit=1000)
    assert len(chunks) > 1 and all(c.startswith("<pre>") and len(c) < 1100 for c in chunks)
    assert "&lt;b&gt;" in chunks[0]


def test_parse_ticket_args():
    argv = parse_ticket_args("vb:aaaa vb:bbbb boost=0.1 stake=5")
    assert argv == ["vb:aaaa", "vb:bbbb", "--boost", "0.1", "--stake", "5", "--record", "real"]
    legs = parse_ticket_args("Arsenal vs Everton:1x2:1; Hull vs Brentford:totals_2.5:over fav=4")
    assert legs == ["--favourites", "--max-legs", "4", "--leg", "Arsenal vs Everton:1x2:1",
                    "--leg", "Hull vs Brentford:totals_2.5:over"]  # fmt: skip


def test_only_authorized_chats_get_answers(tmp_path):
    api = FakeAPI([update(1, "/help", chat_id=999), update(2, "/help")])
    bot, _ = make_bot(tmp_path, api)
    assert bot.poll_once() == 2
    assert [chat for chat, _, _ in api.sent] == ["42"]
    assert "/value" in api.sent[0][1] and bot.offset == 3


def test_commands_map_onto_cli(tmp_path):
    api = FakeAPI()
    bot, runs = make_bot(tmp_path, api)
    for text in ("/bet vb:abcd1234 10 2.1", "/paper vb:abcd1234 5", "/open", "/report", "/next",
                 "/ticket vb:a vb:b boost=0.1", "/parlay 2"):  # fmt: skip
        bot.handle_update(update(1, text))
    db = ["--database", str(tmp_path / "db.sqlite")]
    assert runs == [
        ["bets", "add", "vb:abcd1234", "--stake", "10", "--odds", "2.1", *db],
        ["bets", "add", "vb:abcd1234", "--stake", "5", "--paper", *db],
        ["bets", "list", "--open", *db],
        ["bets", "report", *db],
        ["watch", "--next", *db],
        ["ticket", "vb:a", "vb:b", "--boost", "0.1", *db],
        ["value", "--parlay", "--max-legs", "2", "--days", "3", "--books", "1xBet", *db],
    ]
    assert all(pre for _, _, pre in api.sent)
    bot.handle_update(update(2, "/bet nonsense"))
    bot.handle_update(update(3, "/dance"))
    assert "Usage" in api.sent[-2][1] and "Unknown command" in api.sent[-1][1]


def test_value_command_lists_keys(tmp_path):
    path = tmp_path / "db.sqlite"
    kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
    db = ResearchDatabase(str(path))
    mid = match_id_for(kickoff.date(), "A", "B", "EPL")
    db.insert_future_matches(
        [Match(mid, kickoff.date(), "EPL", "x", "A", "B", None, None, kickoff_at=kickoff)]
    )
    at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.insert_odds([OddsSnapshot(mid, b, "1x2", s, p, at)
                    for b, vals in {"Pinnacle": (2.0, 3.6, 4.0), "1xBet": (2.25, 3.3, 3.5)}.items()
                    for s, p in zip(("home", "draw", "away"), vals, strict=True)])  # fmt: skip
    db.close()
    api = FakeAPI()
    bot = ParlayBot(api, {"42"}, database=str(path), books="1xBet")
    bot.handle_update(update(1, "/value"))
    text = api.sent[0][1]
    assert "1 value bets" in text and "<code>vb:" in text and "1xBet" in text


def test_find_chat_ids_and_setup_cli(tmp_path, monkeypatch, capsys):
    api = FakeAPI([update(1, "/start"), update(2, "hi", chat_id=-100, chat_type="group")])
    assert find_chat_ids(api) == [("42", "Bima")]

    import parlay.telegram_bot as module
    from parlay.cli import main

    env = tmp_path / "env"
    monkeypatch.setenv("PARLAY_ENV_FILE", str(env))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    fake = FakeAPI([update(5, "/start")])
    monkeypatch.setattr(module, "TelegramAPI", lambda token: fake)
    assert main(["telegram", "--setup"]) == 0
    assert read_env_file(env)["TELEGRAM_CHAT_ID"] == "42"
    assert "Connected" in fake.sent[0][1]
    assert "Saved TELEGRAM_CHAT_ID=42" in capsys.readouterr().out


def test_setup_without_messages_explains_what_to_do(monkeypatch, capsys):
    import parlay.telegram_bot as module
    from parlay.cli import main

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.setattr(module, "TelegramAPI", lambda token: FakeAPI())
    assert main(["telegram", "--setup"]) == 1
    assert "send /start to @testbot" in capsys.readouterr().out


def test_bot_requires_token_and_chat(monkeypatch):
    from parlay.cli import main

    with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN"):
        main(["telegram"])
