"""Lineup-window watcher: re-price each kickoff slot shortly before it starts.

Confirmed lineups are published about an hour before kickoff and the sharp
price (Pinnacle) moves within minutes; slower bookmakers can keep stale
prices for a while. ``Watcher`` syncs The Odds API ``lead_minutes`` before
each kickoff slot — only for the competitions playing in that slot — looks
for value in those matches, logs paper bets and sends a notification. Each
slot is processed once (recorded in ``watch_runs``) and syncs stop when the
remaining credits would fall below a reserve.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from parlay.data.database import ResearchDatabase
from parlay.data.odds_sync import SyncResult, credit_cost, sync_the_odds_api
from parlay.data.schemas import Match
from parlay.notify import Notifier

from .ledger import Ledger
from .value import ValueBet, ValueConfig, commission_for, find_value_bets

SCHEMA = """
CREATE TABLE IF NOT EXISTS watch_runs (
    slot TEXT NOT NULL,
    lead_minutes INTEGER NOT NULL,
    ran_at TEXT NOT NULL,
    status TEXT NOT NULL,
    matches INTEGER NOT NULL DEFAULT 0,
    value_bets INTEGER NOT NULL DEFAULT 0,
    credits_remaining INTEGER,
    PRIMARY KEY (slot, lead_minutes)
);
"""

SyncFunction = Callable[..., SyncResult]


@dataclass(frozen=True, slots=True)
class Slot:
    kickoff: datetime
    matches: tuple[Match, ...]

    @property
    def competitions(self) -> tuple[str, ...]:
        return tuple(sorted({m.competition for m in self.matches}))

    def trigger(self, lead_minutes: int) -> datetime:
        return self.kickoff - timedelta(minutes=lead_minutes)


@dataclass(frozen=True, slots=True)
class WatchConfig:
    competitions: tuple[str, ...] = ("EPL", "Championship")
    lead_minutes: int = 50
    min_lead_minutes: int = 10
    regions: str = "eu"
    markets: str = "h2h,totals"
    credit_reserve: int = 20
    bankroll: float = 1000.0
    paper: bool = True
    notify_empty: bool = False
    value: ValueConfig = field(default_factory=ValueConfig)

    def __post_init__(self) -> None:
        if not 0 < self.min_lead_minutes < self.lead_minutes:
            raise ValueError("need 0 < min_lead_minutes < lead_minutes")


@dataclass(frozen=True, slots=True)
class SlotReport:
    slot: Slot
    status: str
    bets: list[ValueBet] = field(default_factory=list)
    credits_remaining: int | None = None
    message: str = ""


def kickoff_slots(matches: Iterable[Match], *, after: datetime, competitions: Iterable[str]) -> list[Slot]:
    """Upcoming matches with a kickoff time, grouped by kickoff."""
    wanted = set(competitions)
    grouped: dict[datetime, list[Match]] = defaultdict(list)
    for m in matches:
        if (
            m.home_goals is None
            and m.kickoff_at is not None
            and m.kickoff_at > after
            and m.competition in wanted
        ):
            grouped[m.kickoff_at].append(m)
    return [
        Slot(kickoff, tuple(sorted(ms, key=lambda m: m.match_id))) for kickoff, ms in sorted(grouped.items())
    ]


def format_slot_message(report: SlotReport, bankroll: float) -> str:
    slot = report.slot
    head = (
        f"⚽ {slot.kickoff:%a %d %b %H:%M} UTC · {len(slot.matches)} matches · {len(report.bets)} value bets"
    )
    lines = [head]
    for bet in report.bets:
        lines.append(
            f"• {bet.description} @ {bet.odds:.2f} ({bet.bookmaker}) fair {bet.fair_odds:.2f} "
            f"EV {bet.expected_value:+.1%} stake {bet.stake_fraction:.2%} "
            f"≈ {bet.stake_fraction * bankroll:.2f} · {bet.key}"
        )
    if report.message:
        lines.append(report.message)
    return "\n".join(lines)


class Watcher:
    def __init__(
        self,
        database: ResearchDatabase,
        api_key: str,
        config: WatchConfig | None = None,
        *,
        notifier: Notifier | None = None,
        sync: SyncFunction | None = None,
    ):
        self.db = database
        self.api_key = api_key
        self.config = config or WatchConfig()
        self.notifier = notifier
        self.sync = sync
        self.ledger = Ledger(database)
        self.db.connection.executescript(SCHEMA)

    # -- schedule -----------------------------------------------------------
    def slots(self, now: datetime) -> list[Slot]:
        return kickoff_slots(self.db.load_matches(), after=now, competitions=self.config.competitions)

    def _done(self, slot: Slot) -> bool:
        row = self.db.connection.execute(
            "SELECT 1 FROM watch_runs WHERE slot = ? AND lead_minutes = ?",
            (slot.kickoff.isoformat(), self.config.lead_minutes),
        ).fetchone()
        return row is not None

    def due_slots(self, now: datetime) -> list[Slot]:
        lead, min_lead = self.config.lead_minutes, self.config.min_lead_minutes
        return [
            s
            for s in self.slots(now)
            if s.trigger(lead) <= now < s.kickoff - timedelta(minutes=min_lead) and not self._done(s)
        ]

    def next_trigger(self, now: datetime) -> datetime | None:
        lead = self.config.lead_minutes
        upcoming = [s.trigger(lead) for s in self.slots(now) if not self._done(s)]
        future = [t for t in upcoming if t > now]
        return min(future) if future else None

    def last_known_credits(self) -> int | None:
        row = self.db.connection.execute(
            "SELECT credits_remaining FROM watch_runs WHERE credits_remaining IS NOT NULL "
            "ORDER BY ran_at DESC LIMIT 1"
        ).fetchone()
        return int(row["credits_remaining"]) if row else None

    # -- running ------------------------------------------------------------
    def _record(self, slot: Slot, now: datetime, report: SlotReport) -> None:
        self.db.connection.execute(
            """INSERT OR REPLACE INTO watch_runs
            (slot, lead_minutes, ran_at, status, matches, value_bets, credits_remaining)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                slot.kickoff.isoformat(),
                self.config.lead_minutes,
                now.isoformat(),
                report.status,
                len(slot.matches),
                len(report.bets),
                report.credits_remaining,
            ),
        )
        self.db.connection.commit()

    def run_slot(self, slot: Slot, now: datetime) -> SlotReport:
        cost = credit_cost(self.config.markets, self.config.regions, len(slot.competitions))
        known = self.last_known_credits()
        if known is not None and known - cost < self.config.credit_reserve:
            report = SlotReport(
                slot,
                "skipped_credits",
                credits_remaining=known,
                message=f"only {known} credits left; sync skipped",
            )
        else:
            try:
                result = (self.sync or sync_the_odds_api)(
                    self.db,
                    self.api_key,
                    slot.competitions,
                    now=now,
                    regions=self.config.regions,
                    markets=self.config.markets,
                )
            except (OSError, ValueError) as exc:
                report = SlotReport(
                    slot, "sync_failed", credits_remaining=known, message=f"sync failed: {exc}"
                )
            else:
                ids = {m.match_id for m in slot.matches}
                current = [m for m in self.db.load_matches() if m.match_id in ids]
                odds = [o for m in current for o in self.db.load_odds(match_id=m.match_id)]
                bets = find_value_bets(current, odds, self.config.value, as_of=now)
                self.ledger.save_candidates(bets)
                if self.config.paper:
                    for bet in bets:
                        if not self.ledger.has_open_bet("paper", bet.match_id, bet.market, bet.selection):
                            self.ledger.record_value_bet(
                                bet,
                                kind="paper",
                                stake=max(bet.stake_fraction * self.config.bankroll, 0.01),
                                commission=commission_for(
                                    bet.bookmaker, self.config.value.exchange_commission
                                ),
                            )
                report = SlotReport(slot, "ok", bets, result.credits_remaining)
        self._record(slot, now, report)
        if self.notifier and (report.bets or self.config.notify_empty or report.status != "ok"):
            try:
                send_report = getattr(self.notifier, "send_report", None)
                if send_report is not None:  # rich notifiers (the Telegram bot) render their own cards
                    send_report(report, self.config.bankroll)
                else:
                    self.notifier.send(format_slot_message(report, self.config.bankroll))
            except OSError as exc:
                print(f"notification failed: {exc}")
        return report

    def run_due(self, now: datetime | None = None) -> list[SlotReport]:
        now = now or datetime.now(timezone.utc)
        return [self.run_slot(slot, now) for slot in self.due_slots(now)]
