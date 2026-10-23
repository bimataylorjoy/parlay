"""Interactive Telegram bot: value bets, tickets and the ledger from your phone.

The bot long-polls Telegram (no public server needed), answers only the
chat ids in ``TELEGRAM_CHAT_ID`` (comma-separated), and can run the
lineup-window watcher in the same process so slot alerts arrive as messages.
"""

from __future__ import annotations

import contextlib
import html
import io
import json
import shlex
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_MESSAGE = 3800

HELP = """<b>parlay bot</b>
/value [days] — value bets vs Pinnacle (default 3 days)
/parlay [legs] — mix-parlay tickets built from value bets
/ticket vb:aaaa vb:bbbb — price a parlay; legs also as
   <code>Arsenal vs Everton:1x2:1; Hull vs Brentford:totals_2.5:over</code>
   <code>boost=0.1</code> · <code>fav=6</code> (next 6 favourites) · <code>stake=5</code> (record as real)
/bet vb:aaaa 10 [odds] — record a real single bet (/paper for a paper bet)
/open — open bets and tickets
/report — profit, yield and closing-line value
/next — upcoming kickoff slots and credit cost
/sync — refresh prices now (uses The Odds API credits)
/credits — remaining The Odds API credits"""


class TelegramAPI:
    """Minimal Bot API client (getUpdates long polling, sendMessage)."""

    def __init__(self, token: str, *, timeout: int = 40):
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        self.base = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout

    def _call(self, method: str, params: dict[str, Any], *, timeout: int | None = None) -> Any:
        data = urlencode(params).encode()
        with urlopen(
            Request(f"{self.base}/{method}", data=data), timeout=timeout or self.timeout
        ) as response:
            payload = json.load(response)
        if not payload.get("ok"):
            raise OSError(f"Telegram {method} failed: {payload.get('description')}")
        return payload["result"]

    def get_me(self) -> dict[str, Any]:
        return self._call("getMe", {})

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": json.dumps(["message"])}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", params, timeout=timeout + 15)

    def send_message(self, chat_id: int | str, text: str, *, preformatted: bool = False) -> None:
        for chunk in split_message(text, preformatted=preformatted):
            self._call(
                "sendMessage",
                {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": "true"},
            )


def split_message(text: str, *, preformatted: bool = False, limit: int = MAX_MESSAGE) -> list[str]:
    """Split on line boundaries; preformatted text is HTML-escaped and wrapped in <pre>."""
    lines = text.splitlines() or [""]
    chunks, current = [], ""
    for line in lines:
        piece = html.escape(line) if preformatted else line
        if current and len(current) + len(piece) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{piece}" if current else piece
    chunks.append(current)
    return [f"<pre>{c}</pre>" if preformatted else c for c in chunks if c.strip()]


def capture_cli(argv: Sequence[str]) -> str:
    """Run a ``parlay`` command and return what it printed (errors included)."""
    from parlay.cli import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            main(list(argv))
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(exc.code if isinstance(exc.code, str) else f"exit status {exc.code}")
        except Exception as exc:
            print(f"error: {exc}")
    return buffer.getvalue().strip() or "(no output)"


def parse_ticket_args(text: str) -> list[str]:
    """``/ticket`` arguments → ``parlay ticket`` argv (keys, ``;``-separated legs, key=value options)."""
    argv: list[str] = []
    options = {"boost": "--boost", "stake": "--stake", "odds": "--ticket-odds", "book": "--book"}
    record = False
    legs: list[str] = []
    for part in [p.strip() for p in text.replace("\n", ";").split(";") if p.strip()]:
        words = []
        for token in shlex.split(part):
            key, eq, value = token.partition("=")
            if eq and key.lower() in options:
                argv += [options[key.lower()], value]
                record = record or key.lower() == "stake"
            elif eq and key.lower() == "fav":
                argv += ["--favourites", "--max-legs", value]
            elif token.lower().startswith("vb:"):
                argv.append(token)
            else:
                words.append(token)
        if words:
            legs.append(" ".join(words))
    for leg in legs:
        argv += ["--leg", leg]
    if record:
        argv += ["--record", "real"]
    return argv


class ParlayBot:
    def __init__(
        self,
        api: TelegramAPI,
        allowed_chats: set[str],
        *,
        database: str,
        books: str | None,
        bankroll: float = 1000.0,
        run_command: Callable[[Sequence[str]], str] = capture_cli,
    ):
        self.api = api
        self.allowed = allowed_chats
        self.database = database
        self.books = books
        self.bankroll = bankroll
        self.run = run_command
        self.offset: int | None = None

    # -- dispatch -----------------------------------------------------------
    def handle_update(self, update: dict[str, Any]) -> None:
        self.offset = update["update_id"] + 1
        message = update.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id", ""))
        text = (message.get("text") or "").strip()
        if not text or chat_id not in self.allowed:
            return
        command, _, rest = text.partition(" ")
        command = command.split("@")[0].lower()
        handler = {
            "/start": self.help,
            "/help": self.help,
            "/value": self.value,
            "/parlay": self.parlay,
            "/ticket": self.ticket,
            "/bet": lambda c, r: self.bet(c, r, kind="real"),
            "/paper": lambda c, r: self.bet(c, r, kind="paper"),
            "/open": lambda c, r: self.cli(c, ["bets", "list", "--open"]),
            "/report": lambda c, r: self.cli(c, ["bets", "report"]),
            "/next": lambda c, r: self.cli(c, ["watch", "--next"]),
            "/sync": self.sync,
            "/credits": self.credits,
        }.get(command)
        if handler is None:
            self.api.send_message(chat_id, "Unknown command. /help lists what I can do.")
            return
        handler(chat_id, rest.strip())

    def poll_once(self, timeout: int = 25) -> int:
        updates = self.api.get_updates(self.offset, timeout=timeout)
        for update in updates:
            try:
                self.handle_update(update)
            except (OSError, ValueError) as exc:
                print(f"failed to handle update {update.get('update_id')}: {exc}")
        return len(updates)

    # -- commands -----------------------------------------------------------
    def _db(self) -> list[str]:
        return ["--database", self.database]

    def _books(self) -> list[str]:
        return ["--books", self.books] if self.books else []

    def cli(self, chat_id: str, argv: list[str]) -> None:
        self.api.send_message(chat_id, self.run([*argv, *self._db()]), preformatted=True)

    def help(self, chat_id: str, _: str) -> None:
        self.api.send_message(chat_id, HELP)

    def value(self, chat_id: str, rest: str) -> None:
        from parlay.betting import Ledger, ValueConfig, find_value_bets
        from parlay.data.database import ResearchDatabase

        days = int(rest) if rest.isdigit() else 3
        db = ResearchDatabase(self.database)
        now = datetime.now(timezone.utc)
        matches = [
            m
            for m in db.load_matches()
            if m.home_goals is None and m.kickoff_at and now < m.kickoff_at <= now + timedelta(days=days)
        ]
        odds = [o for m in matches for o in db.load_odds(match_id=m.match_id)]
        books = tuple(b.strip() for b in self.books.split(",")) if self.books else None
        bets = find_value_bets(matches, odds, ValueConfig(bookmakers=books), as_of=now)
        Ledger(db).save_candidates(bets)
        db.close()
        if not bets:
            self.api.send_message(
                chat_id, f"No value bets in the next {days} days ({len(matches)} matches priced)."
            )
            return
        lines = [f"<b>{len(bets)} value bets</b> · next {days} days · vs Pinnacle"]
        for bet in bets:
            kickoff = f"{bet.kickoff_at:%a %H:%M}" if bet.kickoff_at else bet.match_date.isoformat()
            lines.append(
                f"\n<code>{bet.key}</code> · {kickoff} UTC\n{html.escape(bet.description)}\n"
                f"@ <b>{bet.odds:.2f}</b> {html.escape(bet.bookmaker)} · fair {bet.fair_odds:.2f} · "
                f"EV {bet.expected_value:+.1%} · stake {bet.stake_fraction * self.bankroll:.2f}"
            )
        lines.append("\nPlaced one? <code>/bet KEY STAKE [ODDS]</code>")
        self.api.send_message(chat_id, "\n".join(lines))

    def parlay(self, chat_id: str, rest: str) -> None:
        legs = rest if rest.isdigit() else "3"
        argv = ["value", "--parlay", "--max-legs", legs, "--days", "3", *self._books()]
        output = self.run([*argv, *self._db()])
        marker = "=== Mix parlays"
        text = output[output.index(marker) :] if marker in output else output
        self.api.send_message(chat_id, text, preformatted=True)

    def ticket(self, chat_id: str, rest: str) -> None:
        if not rest:
            self.api.send_message(chat_id, "Usage: /ticket vb:aaaa vb:bbbb [boost=0.1] [stake=5]")
            return
        self.cli(chat_id, ["ticket", *parse_ticket_args(rest)])

    def bet(self, chat_id: str, rest: str, *, kind: str) -> None:
        parts = rest.split()
        if len(parts) < 2 or not parts[0].startswith("vb:"):
            self.api.send_message(
                chat_id, f"Usage: /{'bet' if kind == 'real' else 'paper'} vb:aaaa STAKE [ODDS]"
            )
            return
        argv = ["bets", "add", parts[0], "--stake", parts[1]]
        if len(parts) > 2:
            argv += ["--odds", parts[2]]
        if kind == "paper":
            argv.append("--paper")
        self.cli(chat_id, argv)

    def sync(self, chat_id: str, _: str) -> None:
        self.api.send_message(chat_id, "Syncing prices…")
        output = self.run(["odds", "sync", "--source", "the-odds-api", *self._db()])
        self.api.send_message(chat_id, output, preformatted=True)
        self.value(chat_id, "3")

    def credits(self, chat_id: str, _: str) -> None:
        import os

        key = os.environ.get("THE_ODDS_API_KEY")
        if not key:
            self.api.send_message(chat_id, "THE_ODDS_API_KEY is not set.")
            return
        try:
            with urlopen(f"https://api.the-odds-api.com/v4/sports?apiKey={key}", timeout=20) as response:
                remaining = response.headers.get("x-requests-remaining")
        except URLError as exc:
            self.api.send_message(chat_id, f"Could not reach The Odds API: {exc}")
            return
        self.api.send_message(chat_id, f"The Odds API credits remaining: <b>{remaining}</b> (resets monthly)")


def find_chat_ids(api: TelegramAPI) -> list[tuple[str, str]]:
    """Private chats that recently messaged the bot: ``[(chat_id, name)]``."""
    seen: dict[str, str] = {}
    for update in api.get_updates(timeout=0):
        chat = (update.get("message") or {}).get("chat") or {}
        if chat.get("type") == "private":
            seen[str(chat["id"])] = chat.get("username") or chat.get("first_name") or ""
    return list(seen.items())


def run_bot(bot: ParlayBot, *, watcher: Any = None, stop_at: datetime | None = None) -> None:
    """Long-poll forever (or until ``stop_at``), running due watcher slots between polls."""
    while stop_at is None or datetime.now(timezone.utc) < stop_at:
        try:
            bot.poll_once(timeout=25)
        except (OSError, URLError) as exc:
            print(f"telegram poll failed: {exc}; retrying in 10 s")
            time.sleep(10)
        if watcher is not None:
            for report in watcher.run_due():
                slot = f"{report.slot.kickoff:%Y-%m-%d %H:%M} UTC"
                print(f"slot {slot}: {report.status}, {len(report.bets)} value bets")
