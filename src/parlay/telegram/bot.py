"""Button-driven Telegram bot for beginners (Indonesian).

Every value bet arrives as a card — what to bet, the market in plain words,
odds, win probability, EV and a stake in money — with buttons to log it as
placed, simulate it, or add it to a parlay slip. The bot answers only the
configured chat ids and also runs a daily maintenance job (results, closing
prices, settlement, fresh odds) so the ledger stays current without commands.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.error import URLError

from parlay.betting import Ledger, ValueConfig, find_value_bets
from parlay.betting.tickets import build_ticket, price_leg
from parlay.betting.value import ValueBet, commission_for
from parlay.betting.watch import SlotReport, kickoff_slots
from parlay.data.database import ResearchDatabase

from .api import Buttons, TelegramAPI
from .state import BotState
from .texts import (
    HELP,
    WELCOME,
    describe_bet,
    esc,
    local_zone,
    min_stake,
    money,
    report_text,
    ticket_card,
    value_card,
    when,
)

MENU: Buttons = [
    [("🎯 Value bet", "m:value"), ("🎟 Slip parlay", "m:slip")],
    [("📊 Laporan", "m:report"), ("🗓 Jadwal", "m:next")],
    [("🔄 Update harga", "m:sync"), ("⚙️ Modal", "m:bankroll")],
    [("ℹ️ Bantuan", "m:help")],
]
COMMANDS = [
    ("menu", "Menu utama"),
    ("value", "Value bet terbaru"),
    ("slip", "Slip mix parlay"),
    ("laporan", "Untung/rugi dan kualitas harga"),
    ("jadwal", "Jadwal laga dan pengecekan harga"),
    ("modal", "Atur modal, contoh: /modal 1jt"),
    ("bantuan", "Cara pakai"),
]
CARDS_PER_PAGE = 5
QUICK_STAKES = (10_000.0, 50_000.0, 100_000.0)
_AMOUNT = re.compile(r"^\s*([\d.,]+)\s*(rb|ribu|k|jt|juta)?\s*(?:@\s*([\d.,]+))?\s*$", re.IGNORECASE)


def parse_amount(text: str) -> tuple[float, float | None] | None:
    """``"50rb"`` → 50 000; ``"1,5jt"`` → 1 500 000; ``"50000@2.15"`` → (50000, 2.15)."""
    match = _AMOUNT.match(text)
    if not match:
        return None
    number, unit, odds = match.groups()
    if unit:
        value = float(number.replace(".", "").replace(",", ".")) if "," in number else float(number)
        value *= 1_000 if unit.lower() in ("rb", "ribu", "k") else 1_000_000
    else:
        value = float(number.replace(".", "").replace(",", ""))
    price = float(odds.replace(",", ".")) if odds else None
    if value <= 0 or (price is not None and price <= 1):
        return None
    return value, price


def quick_amounts(suggested: float) -> list[float]:
    """Stake buttons: the suggestion (never below the bookmaker minimum) plus round amounts."""
    floor = min_stake()
    first = max(round(suggested, -3), floor)
    return sorted({first, *(a for a in QUICK_STAKES if a >= floor)})[:4]


def capture_cli(argv: Sequence[str]) -> str:
    from parlay.cli import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            main(list(argv))
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(exc.code)
        except Exception as exc:
            print(f"error: {exc}")
    return buffer.getvalue()


class ParlayBot:
    def __init__(
        self,
        api: TelegramAPI,
        allowed_chats: set[str],
        *,
        database: str,
        books: str | None,
        competitions: tuple[str, ...] = ("EPL", "Championship"),
        run_command: Callable[[Sequence[str]], str] = capture_cli,
    ):
        self.api = api
        self.allowed = allowed_chats
        self.database = database
        self.db = ResearchDatabase(database)
        self.ledger = Ledger(self.db)
        self.state = BotState(self.db)
        self.books = tuple(b.strip() for b in books.split(",") if b.strip()) if books else None
        self.competitions = competitions
        self.run = run_command
        self.offset: int | None = None
        self.pending: dict[str, dict[str, Any]] = {}
        self.last_maintenance: date | None = None

    # ------------------------------------------------------------------ io
    def send(self, chat_id: str, text: str, buttons: Buttons | None = None) -> None:
        self.api.send_message(chat_id, text, buttons=buttons)

    def poll_once(self, timeout: int = 25) -> int:
        updates = self.api.get_updates(self.offset, timeout=timeout)
        for update in updates:
            self.offset = update["update_id"] + 1
            chat = self._chat_of(update)
            try:
                self.handle_update(update)
            except KeyError:
                if chat in self.allowed:
                    self.pending.pop(chat, None)
                    self.menu(
                        chat, "⌛ Taruhan ini sudah kedaluwarsa. Tekan 🎯 Value bet untuk daftar terbaru."
                    )
            except (OSError, ValueError) as exc:
                if chat in self.allowed:
                    self.send(chat, f"⚠️ Terjadi kesalahan: {esc(str(exc))}")
        return len(updates)

    @staticmethod
    def _chat_of(update: dict[str, Any]) -> str:
        if "callback_query" in update:
            return str(update["callback_query"]["message"]["chat"]["id"])
        return str(((update.get("message") or {}).get("chat") or {}).get("id", ""))

    def handle_update(self, update: dict[str, Any]) -> None:
        chat = self._chat_of(update)
        if chat not in self.allowed:
            return
        if "callback_query" in update:
            query = update["callback_query"]
            self.api.answer_callback(query["id"])
            self.on_button(chat, query.get("data", ""))
        else:
            self.on_text(chat, ((update.get("message") or {}).get("text") or "").strip())

    # ------------------------------------------------------------- routing
    def on_text(self, chat: str, text: str) -> None:
        if not text:
            return
        if text.startswith("/"):
            command, _, rest = text[1:].partition(" ")
            command = command.split("@")[0].lower()
            routes: dict[str, Callable[[], None]] = {
                "start": lambda: self.menu(chat, WELCOME),
                "menu": lambda: self.menu(chat),
                "value": lambda: self.show_value(chat),
                "slip": lambda: self.show_slip(chat),
                "laporan": lambda: self.show_report(chat),
                "report": lambda: self.show_report(chat),
                "jadwal": lambda: self.show_schedule(chat),
                "bantuan": lambda: self.menu(chat, HELP),
                "help": lambda: self.menu(chat, HELP),
                "modal": lambda: self.set_bankroll(chat, rest) if rest else self.ask_bankroll(chat),
            }
            routes.get(command, lambda: self.menu(chat, "Perintah tidak dikenal. Pilih dari menu 👇"))()
            return
        pending = self.pending.get(chat)
        if pending:
            self.on_amount(chat, text, pending)
        else:
            self.menu(chat, "Pilih dari menu 👇")

    def on_button(self, chat: str, data: str) -> None:
        kind, _, arg = data.partition(":")
        if kind == "m":
            {
                "value": lambda: self.show_value(chat),
                "value_all": lambda: self.show_value(chat, everything=True),
                "slip": lambda: self.show_slip(chat),
                "clear": lambda: self.clear_slip(chat),
                "report": lambda: self.show_report(chat),
                "next": lambda: self.show_schedule(chat),
                "sync": lambda: self.sync_prices(chat),
                "bankroll": lambda: self.ask_bankroll(chat),
                "help": lambda: self.menu(chat, HELP),
                "menu": lambda: self.menu(chat),
            }.get(arg, lambda: self.menu(chat))()
        elif kind == "p":
            self.ask_stake(chat, arg)
        elif kind == "b":
            key, _, amount = arg.rpartition(":")
            self.record_bet(chat, key, float(amount), None, kind="real")
        elif kind == "s":
            self.record_simulation(chat, arg)
        elif kind == "a":
            self.add_to_slip(chat, arg)
        elif kind == "t":
            self.ask_ticket_stake(chat)
        elif kind == "tb":
            self.record_ticket(chat, float(arg), None)
        elif kind == "x":
            self.pending.pop(chat, None)
            self.menu(chat, "Dibatalkan.")

    def menu(self, chat: str, text: str = "Menu utama 👇") -> None:
        self.send(chat, text, MENU)

    # -------------------------------------------------------------- value
    def value_bets(self, days: int = 3) -> list[ValueBet]:
        now = datetime.now(timezone.utc)
        matches = [
            m
            for m in self.db.load_matches()
            if m.home_goals is None and m.kickoff_at and now < m.kickoff_at <= now + timedelta(days=days)
        ]
        odds = [o for m in matches for o in self.db.load_odds(match_id=m.match_id)]
        bets = find_value_bets(matches, odds, ValueConfig(bookmakers=self.books), as_of=now)
        self.ledger.save_candidates(bets)
        return bets

    @staticmethod
    def card_buttons(bet: ValueBet) -> Buttons:
        return [
            [("✅ Saya pasang", f"p:{bet.key}"), ("🧪 Simulasi", f"s:{bet.key}")],
            [("➕ Tambah ke parlay", f"a:{bet.key}")],
        ]

    def show_value(self, chat: str, *, everything: bool = False) -> None:
        bets = self.value_bets()
        if not bets:
            self.menu(
                chat,
                "😴 <b>Belum ada value bet</b> untuk 3 hari ke depan.\n"
                "Artinya odds bookie kamu belum ada yang lebih tinggi dari harga wajar. "
                "Saya akan mengecek lagi otomatis sebelum setiap kickoff.",
            )
            return
        bankroll = self.state.bankroll(chat)
        shown = bets if everything else bets[:CARDS_PER_PAGE]
        self.send(
            chat,
            f"🎯 <b>{len(bets)} value bet</b> ditemukan (3 hari ke depan), urut dari EV tertinggi.\n"
            f"Modal: {money(bankroll)} · /modal untuk mengubah.",
        )
        for bet in shown:
            self.send(chat, value_card(bet, bankroll), self.card_buttons(bet))
        if len(shown) < len(bets):
            self.send(chat, f"Masih ada {len(bets) - len(shown)} lagi.", [[("Lihat semua", "m:value_all")]])

    # ------------------------------------------------------------ betting
    def _candidate(self, key: str) -> dict[str, Any]:
        data = self.ledger.candidate(key)
        data["key"] = key
        return data

    def ask_stake(self, chat: str, key: str) -> None:
        data = self._candidate(key)
        pick, _ = describe_bet(data["home_team"], data["away_team"], data["market"], data["selection"])
        suggested = float(data["stake_fraction"]) * self.state.bankroll(chat)
        self.pending[chat] = {"type": "bet", "key": key}
        amounts = quick_amounts(suggested)
        buttons: Buttons = [[(money(a), f"b:{key}:{a:.0f}") for a in amounts], [("Batal", "x:")]]
        self.send(
            chat,
            f"💬 Berapa taruhanmu untuk <b>{esc(pick)}</b> @ {float(data['odds']):.2f} "
            f"({esc(data['bookmaker'])})?\n\n"
            f"Saran: <b>{money(suggested)}</b>. Tekan tombol atau ketik nominal, misal <code>50rb</code>.\n"
            f"Kalau odds di bookie sudah berubah, ketik <code>50rb@2,15</code>.",
            buttons,
        )

    def on_amount(self, chat: str, text: str, pending: dict[str, Any]) -> None:
        parsed = parse_amount(text)
        if parsed is None:
            self.send(
                chat,
                "Format tidak dikenali. Contoh: <code>50000</code>, <code>50rb</code>, <code>1jt</code>.",
            )
            return
        amount, odds = parsed
        if pending["type"] == "bet":
            self.record_bet(chat, pending["key"], amount, odds, kind="real")
        elif pending["type"] == "ticket":
            self.record_ticket(chat, amount, odds)
        elif pending["type"] == "bankroll":
            self.set_bankroll(chat, text)

    def record_bet(self, chat: str, key: str, stake: float, odds: float | None, *, kind: str) -> None:
        self.pending.pop(chat, None)
        data = self._candidate(key)
        bet_id = self.ledger.record_value_bet(
            data, kind=kind, stake=stake, odds=odds, commission=commission_for(data["bookmaker"])
        )
        pick, _ = describe_bet(data["home_team"], data["away_team"], data["market"], data["selection"])
        if bet_id is None:
            self.menu(chat, "ℹ️ Taruhan ini sudah tercatat sebelumnya.")
            return
        label = "Dicatat" if kind == "real" else "Disimpan sebagai simulasi"
        self.menu(
            chat,
            f"✅ {label}: <b>{esc(pick)}</b> @ {odds or float(data['odds']):.2f}, "
            f"taruhan {money(stake)}.\n"
            "Saya akan mengecek hasil dan kualitas harganya otomatis.",
        )

    def record_simulation(self, chat: str, key: str) -> None:
        data = self._candidate(key)
        stake = max(float(data["stake_fraction"]) * self.state.bankroll(chat), 1.0)
        self.record_bet(chat, key, stake, None, kind="paper")

    # -------------------------------------------------------------- slip
    def add_to_slip(self, chat: str, key: str) -> None:
        data = self._candidate(key)
        current = [self._candidate(k) for k in self.state.slip(chat)]
        if any(c["match_id"] == data["match_id"] for c in current):
            self.send(chat, "Laga ini sudah ada di slip (satu tim per laga).", [[("🎟 Lihat slip", "m:slip")]])
            return
        self.state.add_to_slip(chat, key)
        count = len(current) + 1
        self.send(
            chat,
            f"➕ Ditambahkan ke slip parlay ({count} tim).",
            [[("🎟 Lihat slip", "m:slip"), ("🗑 Kosongkan", "m:clear")]],
        )

    def slip_ticket(self, chat: str) -> Any:
        now = datetime.now(timezone.utc)
        legs, books = [], []
        for key in self.state.slip(chat):
            data = self._candidate(key)
            match = next((m for m in self.db.load_matches() if m.match_id == data["match_id"]), None)
            if match is None or (match.kickoff_at and match.kickoff_at <= now):
                self.state.remove_from_slip(chat, key)
                continue
            books.append(data["bookmaker"])
            legs.append(
                price_leg(
                    match,
                    data["market"],
                    data["selection"],
                    self.db.load_odds(match_id=match.match_id),
                    bookmaker=data["bookmaker"],
                    as_of=now,
                )
            )
        if len(legs) < 2:
            return None
        return build_ticket(legs, max(set(books), key=books.count))

    def show_slip(self, chat: str) -> None:
        count = len(self.state.slip(chat))
        ticket = self.slip_ticket(chat) if count >= 2 else None
        if ticket is None:
            self.menu(
                chat,
                f"🎟 Slip parlay berisi {len(self.state.slip(chat))} tim. Tambahkan minimal 2 value bet "
                "dengan tombol <b>➕ Tambah ke parlay</b>.",
            )
            return
        buttons: Buttons = [[("🗑 Kosongkan slip", "m:clear")]]
        if (ticket.expected_value or 0) > 0:
            buttons.insert(0, [("✅ Saya pasang tiket ini", "t:")])
        self.send(chat, ticket_card(ticket, self.state.bankroll(chat)), buttons)

    def clear_slip(self, chat: str) -> None:
        self.state.clear_slip(chat)
        self.menu(chat, "🗑 Slip parlay dikosongkan.")

    def ask_ticket_stake(self, chat: str) -> None:
        ticket = self.slip_ticket(chat)
        if ticket is None:
            self.menu(chat, "Slip kosong atau kurang dari 2 tim.")
            return
        suggested = ticket.stake_fraction() * self.state.bankroll(chat)
        self.pending[chat] = {"type": "ticket"}
        amounts = quick_amounts(suggested)
        self.send(
            chat,
            f"💬 Berapa taruhan untuk tiket @ {ticket.combined_odds:.2f}? Saran {money(suggested)}.\n"
            "Ketik nominal atau tekan tombol. Kalau total odds di bookie berbeda, "
            "ketik <code>20rb@7,9</code>.",
            [[(money(a), f"tb:{a:.0f}") for a in amounts], [("Batal", "x:")]],
        )

    def record_ticket(self, chat: str, stake: float, odds: float | None) -> None:
        self.pending.pop(chat, None)
        ticket = self.slip_ticket(chat)
        if ticket is None:
            self.menu(chat, "Slip kosong atau kurang dari 2 tim.")
            return
        ticket_id = self.ledger.record_ticket(ticket, kind="real", stake=stake, odds=odds)
        self.state.clear_slip(chat)
        self.menu(
            chat,
            f"✅ Tiket parlay #{ticket_id} dicatat: {money(stake)} @ {odds or ticket.combined_odds:.2f}.",
        )

    # ----------------------------------------------------- other screens
    def show_report(self, chat: str) -> None:
        self.ledger.update()
        self.ledger.update_tickets()
        self.menu(chat, report_text(self.ledger.report(kind="real"), self.ledger.report(kind="paper")))

    def show_schedule(self, chat: str) -> None:
        now = datetime.now(timezone.utc)
        slots = [s for s in kickoff_slots(self.db.load_matches(), after=now, competitions=self.competitions)]
        slots = [s for s in slots if s.kickoff <= now + timedelta(days=7)][:10]
        if not slots:
            self.menu(chat, "🗓 Belum ada jadwal tersimpan. Tekan <b>🔄 Update harga</b>.")
            return
        lines = ["🗓 <b>Jadwal 7 hari</b> (harga dicek ulang ±50 menit sebelum kickoff)", ""]
        for slot in slots:
            names = ", ".join(f"{m.home_team}–{m.away_team}" for m in slot.matches[:4])
            more = f" +{len(slot.matches) - 4}" if len(slot.matches) > 4 else ""
            lines.append(f"• <b>{when(slot.kickoff)}</b>\n   {esc(names)}{more}")
        self.menu(chat, "\n".join(lines))

    def sync_prices(self, chat: str) -> None:
        self.send(chat, "🔄 Mengambil harga terbaru…")
        output = self.run(["odds", "sync", "--source", "the-odds-api", "--database", self.database])
        credits = re.search(r"credits remaining: (\d+)", output)
        if credits is None:
            self.send(chat, f"⚠️ Gagal memperbarui harga:\n<code>{esc(output.strip()[-500:])}</code>")
            return
        self.send(chat, f"✅ Harga diperbarui. Sisa kuota API bulan ini: {credits.group(1)}.")
        self.show_value(chat)

    def ask_bankroll(self, chat: str) -> None:
        self.pending[chat] = {"type": "bankroll"}
        self.send(
            chat,
            f"⚙️ Modal sekarang: <b>{money(self.state.bankroll(chat))}</b>.\n"
            "Ketik modal barumu, misal <code>2jt</code> atau <code>500rb</code>.",
            [[("Batal", "x:")]],
        )

    def set_bankroll(self, chat: str, text: str) -> None:
        parsed = parse_amount(text)
        if parsed is None:
            self.send(chat, "Format tidak dikenali. Contoh: <code>/modal 1jt</code>.")
            return
        self.pending.pop(chat, None)
        self.state.set_bankroll(chat, parsed[0])
        self.menu(chat, f"✅ Modal diatur ke <b>{money(parsed[0])}</b>. Saran taruhan menyesuaikan.")

    # -------------------------------------------------------- automation
    def send_slot_report(self, report: SlotReport) -> None:
        """Watcher hook: one card per value bet found after lineups."""
        for chat in self.allowed:
            if report.status != "ok":
                self.send(chat, f"⚠️ Pengecekan harga {when(report.slot.kickoff)} gagal: {report.message}")
                continue
            if not report.bets:
                continue
            self.send(chat, f"🔔 <b>Harga terbaru sebelum kickoff {when(report.slot.kickoff)}</b>")
            for bet in report.bets:
                self.send(chat, value_card(bet, self.state.bankroll(chat)), self.card_buttons(bet))

    def daily_maintenance(self, now: datetime | None = None) -> bool:
        """Once a day after 06:00 local time: results, closing prices, settlement, fresh odds."""
        now = now or datetime.now(timezone.utc)
        local = now.astimezone(local_zone())
        if local.hour < 6 or self.last_maintenance == local.date():
            return False
        self.last_maintenance = local.date()
        before = {b.bet_id for b in self.ledger.bets() if b.result is not None}
        season = local.year if local.month >= 7 else local.year - 1
        self.run(["fetch", "--league", "E0", "E1", "--from-season", str(season), "--to-season", str(season),
                  "--database", self.database])  # fmt: skip
        if os.environ.get("THE_ODDS_API_KEY"):
            self.run(["odds", "sync", "--source", "the-odds-api", "--database", self.database])
        self.ledger.update()
        self.ledger.update_tickets()
        settled = [
            b for b in self.ledger.bets(kind="real") if b.result is not None and b.bet_id not in before
        ]
        if settled:
            lines = ["🏁 <b>Hasil taruhanmu</b>"]
            for bet in settled:
                pick, _ = describe_bet(bet.home_team or "", bet.away_team or "", bet.market, bet.selection)
                icon = "✅" if (bet.pnl or 0) > 0 else ("➖" if (bet.pnl or 0) == 0 else "❌")
                teams = f"{esc(bet.home_team or '')} vs {esc(bet.away_team or '')}"
                lines.append(f"{icon} {teams}: {esc(pick)} → {money(bet.pnl or 0)}")
            for chat in self.allowed:
                self.menu(chat, "\n".join(lines))
        return True


class BotNotifier:
    """Watcher notifier that sends value-bet cards with buttons through the bot."""

    def __init__(self, bot: ParlayBot):
        self.bot = bot

    def send(self, text: str) -> None:
        for chat in self.bot.allowed:
            self.bot.send(chat, esc(text))

    def send_report(self, report: SlotReport, bankroll: float) -> None:
        self.bot.send_slot_report(report)


def run_bot(bot: ParlayBot, *, watcher: Any = None) -> None:
    while True:
        try:
            bot.poll_once(timeout=25)
        except (OSError, URLError) as exc:
            print(f"telegram poll failed: {exc}; retrying in 10 s")
            time.sleep(10)
        bot.daily_maintenance()
        if watcher is not None:
            for report in watcher.run_due():
                slot = f"{report.slot.kickoff:%Y-%m-%d %H:%M} UTC"
                print(f"slot {slot}: {report.status}, {len(report.bets)} value bets")
