"""Bet ledger: record paper/real bets, attach closing-line value, settle, and report.

Closing-line value (CLV) — the taken price relative to the fair closing
price, ``effective_odds × p_close − 1`` — is the fastest reliable evidence of
edge: its noise is far smaller than that of profit and loss, so a few hundred
bets say more about CLV than thousands say about yield.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any

from parlay.data.database import ResearchDatabase
from parlay.data.odds_columns import parse_market
from parlay.data.schemas import Match, OddsSnapshot

from .reference import fair_price
from .settlement import selections_for, settle
from .tickets import Ticket
from .value import ValueBet

SCHEMA = """
CREATE TABLE IF NOT EXISTS value_candidates (
    key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bets (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('paper', 'real')),
    placed_at TEXT NOT NULL,
    match_id TEXT NOT NULL,
    competition TEXT,
    match_date TEXT,
    kickoff_at TEXT,
    home_team TEXT,
    away_team TEXT,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    odds REAL NOT NULL,
    commission REAL NOT NULL DEFAULT 0,
    stake REAL NOT NULL,
    fair_probability REAL,
    reference TEXT,
    candidate_key TEXT,
    closing_probability REAL,
    closing_source TEXT,
    clv REAL,
    result TEXT,
    pnl REAL,
    settled_at TEXT,
    note TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bets_candidate
    ON bets(kind, candidate_key) WHERE candidate_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_bets_match ON bets(match_id);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('paper', 'real')),
    placed_at TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    odds REAL NOT NULL,
    stake REAL NOT NULL,
    boost REAL NOT NULL DEFAULT 0,
    fair_probability REAL,
    expected_value REAL,
    ticket_key TEXT,
    legs INTEGER NOT NULL,
    closing_probability REAL,
    clv REAL,
    result TEXT,
    pnl REAL,
    settled_at TEXT
);
CREATE TABLE IF NOT EXISTS ticket_legs (
    ticket_id INTEGER NOT NULL REFERENCES tickets(ticket_id),
    leg_no INTEGER NOT NULL,
    match_id TEXT NOT NULL,
    competition TEXT,
    match_date TEXT,
    kickoff_at TEXT,
    home_team TEXT,
    away_team TEXT,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds REAL NOT NULL,
    fair_probability REAL,
    closing_probability REAL,
    closing_source TEXT,
    result TEXT,
    PRIMARY KEY (ticket_id, leg_no)
);
"""

STOP_MIN_BETS = 50
EDGE_MIN_BETS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class BetRow:
    bet_id: int
    kind: str
    placed_at: str
    match_id: str
    competition: str | None
    match_date: str | None
    kickoff_at: str | None
    home_team: str | None
    away_team: str | None
    market: str
    selection: str
    bookmaker: str
    odds: float
    commission: float
    stake: float
    fair_probability: float | None
    reference: str | None
    candidate_key: str | None
    closing_probability: float | None
    closing_source: str | None
    clv: float | None
    result: str | None
    pnl: float | None
    settled_at: str | None
    note: str | None

    @property
    def effective_odds(self) -> float:
        return 1.0 + (self.odds - 1.0) * (1.0 - self.commission)


class Ledger:
    def __init__(self, database: ResearchDatabase):
        self.db = database
        self.db.connection.executescript(SCHEMA)

    # -- candidates ---------------------------------------------------------
    def save_candidates(self, bets: list[ValueBet]) -> None:
        self.db.connection.executemany(
            "INSERT OR REPLACE INTO value_candidates(key, created_at, payload_json) VALUES (?, ?, ?)",
            [(bet.key, _now(), json.dumps(asdict(bet), default=str)) for bet in bets],
        )
        self.db.connection.commit()

    def candidate(self, key: str) -> dict[str, Any]:
        row = self.db.connection.execute(
            "SELECT payload_json FROM value_candidates WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown value bet {key!r}; run `parlay value` first")
        return json.loads(row["payload_json"])

    # -- recording ----------------------------------------------------------
    def record(
        self,
        *,
        kind: str,
        match_id: str,
        market: str,
        selection: str,
        bookmaker: str,
        odds: float,
        stake: float,
        commission: float = 0.0,
        fair_probability: float | None = None,
        reference: str | None = None,
        candidate_key: str | None = None,
        match: Match | None = None,
        note: str | None = None,
    ) -> int | None:
        """Insert one bet; returns its id, or ``None`` if this candidate was already recorded."""
        if kind not in ("paper", "real"):
            raise ValueError("kind must be 'paper' or 'real'")
        if odds <= 1 or stake <= 0:
            raise ValueError("odds must exceed 1 and stake must be positive")
        selections_for(market)  # validates market
        if selection not in selections_for(market):
            raise ValueError(f"selection must be one of {selections_for(market)}")
        cursor = self.db.connection.execute(
            """INSERT OR IGNORE INTO bets (kind, placed_at, match_id, competition, match_date, kickoff_at,
                home_team, away_team, market, selection, bookmaker, odds, commission, stake,
                fair_probability, reference, candidate_key, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kind,
                _now(),
                match_id,
                match.competition if match else None,
                match.date.isoformat() if match else None,
                match.kickoff_at.isoformat() if match and match.kickoff_at else None,
                match.home_team if match else None,
                match.away_team if match else None,
                market,
                selection,
                bookmaker,
                odds,
                commission,
                stake,
                fair_probability,
                reference,
                candidate_key,
                note,
            ),
        )
        self.db.connection.commit()
        return cursor.lastrowid if cursor.rowcount else None

    def record_value_bet(
        self,
        bet: ValueBet | dict[str, Any],
        *,
        kind: str,
        stake: float,
        odds: float | None = None,
        commission: float = 0.0,
    ) -> int | None:
        data = asdict(bet) if isinstance(bet, ValueBet) else bet
        key = bet.key if isinstance(bet, ValueBet) else data.get("key")
        kickoff = data.get("kickoff_at")
        match = Match(
            match_id=data["match_id"],
            date=date.fromisoformat(str(data["match_date"])),
            competition=data["competition"],
            season="",
            home_team=data["home_team"],
            away_team=data["away_team"],
            home_goals=None,
            away_goals=None,
            kickoff_at=datetime.fromisoformat(str(kickoff)) if kickoff and kickoff != "None" else None,
        )
        return self.record(
            kind=kind,
            match_id=data["match_id"],
            market=data["market"],
            selection=data["selection"],
            bookmaker=data["bookmaker"],
            odds=odds or float(data["odds"]),
            stake=stake,
            commission=commission,
            fair_probability=float(data["fair_probability"]),
            reference=data["reference"],
            candidate_key=key,
            match=match,
        )

    # -- reading ------------------------------------------------------------
    def has_open_bet(self, kind: str, match_id: str, market: str, selection: str) -> bool:
        row = self.db.connection.execute(
            "SELECT 1 FROM bets WHERE kind = ? AND match_id = ? AND market = ? AND selection = ? "
            "AND result IS NULL",
            (kind, match_id, market, selection),
        ).fetchone()
        return row is not None

    def bets(self, *, kind: str | None = None, open_only: bool = False) -> list[BetRow]:
        query = "SELECT * FROM bets"
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if open_only:
            clauses.append("result IS NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY placed_at, bet_id"
        return [BetRow(**dict(row)) for row in self.db.connection.execute(query, params)]

    # -- closing line value and settlement ------------------------------------
    def update(self, *, method: str = "shin") -> dict[str, int]:
        """Attach closing prices/CLV and settle bets whose results are known."""
        matches = {m.match_id: m for m in self.db.load_matches()}
        open_rows = [b for b in self.bets() if b.clv is None or b.result is None]
        odds_by_match: dict[str, list[OddsSnapshot]] = {}
        counts = {"clv": 0, "settled": 0}
        for bet in open_rows:
            match = matches.get(bet.match_id)
            if bet.match_id not in odds_by_match:
                odds_by_match[bet.match_id] = self.db.load_odds(match_id=bet.match_id)
            updates: dict[str, Any] = {}
            if bet.clv is None:
                closing = closing_fair_probability(odds_by_match[bet.match_id], bet, match, method=method)
                if closing is not None:
                    probability, source = closing
                    updates.update(
                        closing_probability=probability,
                        closing_source=source,
                        clv=bet.effective_odds * probability - 1.0,
                    )
                    counts["clv"] += 1
            if bet.result is None and match is not None and match.home_goals is not None:
                outcome = settle(bet.market, bet.selection, match.home_goals, match.away_goals)  # type: ignore[arg-type]
                updates.update(
                    result=outcome.label, pnl=outcome.profit(bet.stake, bet.effective_odds), settled_at=_now()
                )
                counts["settled"] += 1
            if updates:
                assignments = ", ".join(f"{column} = ?" for column in updates)
                self.db.connection.execute(
                    f"UPDATE bets SET {assignments} WHERE bet_id = ?", (*updates.values(), bet.bet_id)
                )
        self.db.connection.commit()
        return counts

    def report(self, *, kind: str | None = None) -> dict[str, Any]:
        report = summarize_bets(self.bets(kind=kind))
        report["parlays"] = summarize_tickets(self.tickets(kind=kind))
        return report

    # -- parlay tickets -----------------------------------------------------
    def record_ticket(
        self, ticket: Ticket, *, kind: str, stake: float, odds: float | None = None
    ) -> int | None:
        """Store a ticket and its legs; ``None`` if an identical open ticket of this kind exists."""
        if kind not in ("paper", "real") or stake <= 0:
            raise ValueError("kind must be 'paper' or 'real' and stake positive")
        taken = odds or ticket.combined_odds
        if kind == "paper" and self.has_open_ticket(kind, ticket.key):
            return None
        p = ticket.win_probability
        cursor = self.db.connection.execute(
            """INSERT INTO tickets (kind, placed_at, bookmaker, odds, stake, boost, fair_probability,
                expected_value, ticket_key, legs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kind,
                _now(),
                ticket.bookmaker,
                taken,
                stake,
                ticket.boost,
                p,
                taken * p - 1.0 if p is not None else None,
                ticket.key,
                len(ticket.legs),
            ),
        )
        ticket_id = cursor.lastrowid
        self.db.connection.executemany(
            """INSERT INTO ticket_legs (ticket_id, leg_no, match_id, competition, match_date, kickoff_at,
                home_team, away_team, market, selection, odds, fair_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ticket_id,
                    number,
                    leg.match_id,
                    leg.competition,
                    leg.match_date.isoformat(),
                    leg.kickoff_at.isoformat() if leg.kickoff_at else None,
                    leg.home_team,
                    leg.away_team,
                    leg.market,
                    leg.selection,
                    leg.odds,
                    leg.fair_probability,
                )
                for number, leg in enumerate(ticket.legs, start=1)
            ],
        )
        self.db.connection.commit()
        return ticket_id

    def has_open_ticket(self, kind: str, key: str) -> bool:
        row = self.db.connection.execute(
            "SELECT 1 FROM tickets WHERE kind = ? AND ticket_key = ? AND result IS NULL", (kind, key)
        ).fetchone()
        return row is not None

    def tickets(self, *, kind: str | None = None, open_only: bool = False) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM tickets", []
        clauses = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if open_only:
            clauses.append("result IS NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = [dict(r) for r in self.db.connection.execute(query + " ORDER BY placed_at, ticket_id", params)]
        for row in rows:
            row["leg_rows"] = [
                dict(r)
                for r in self.db.connection.execute(
                    "SELECT * FROM ticket_legs WHERE ticket_id = ? ORDER BY leg_no", (row["ticket_id"],)
                )
            ]
        return rows

    def update_tickets(self, *, method: str = "shin") -> dict[str, int]:
        """Closing prices per leg, ticket CLV once every leg has one, settlement once every leg is final."""
        matches = {m.match_id: m for m in self.db.load_matches()}
        counts = {"clv": 0, "settled": 0}
        for ticket in self.tickets():
            if ticket["clv"] is not None and ticket["result"] is not None:
                continue
            legs = ticket["leg_rows"]
            for leg in legs:
                match = matches.get(leg["match_id"])
                if leg["closing_probability"] is None:
                    closing = closing_probability(
                        self.db.load_odds(match_id=leg["match_id"]),
                        leg["market"],
                        leg["selection"],
                        ticket["placed_at"],
                        match,
                        method=method,
                    )
                    if closing is not None:
                        leg["closing_probability"], leg["closing_source"] = closing
                if (
                    leg["result"] is None
                    and match
                    and match.home_goals is not None
                    and match.away_goals is not None
                ):
                    leg["result"] = settle(
                        leg["market"], leg["selection"], match.home_goals, match.away_goals
                    ).label
                self.db.connection.execute(
                    """UPDATE ticket_legs SET closing_probability = ?, closing_source = ?, result = ?
                    WHERE ticket_id = ? AND leg_no = ?""",
                    (
                        leg["closing_probability"],
                        leg["closing_source"],
                        leg["result"],
                        ticket["ticket_id"],
                        leg["leg_no"],
                    ),
                )
            updates: dict[str, Any] = {}
            if ticket["clv"] is None and all(leg["closing_probability"] is not None for leg in legs):
                p_close = math.prod(leg["closing_probability"] for leg in legs)
                updates.update(closing_probability=p_close, clv=ticket["odds"] * p_close - 1.0)
                counts["clv"] += 1
            if ticket["result"] is None and all(leg["result"] is not None for leg in legs):
                result, pnl = settle_ticket(ticket["odds"], ticket["stake"], legs)
                updates.update(result=result, pnl=pnl, settled_at=_now())
                counts["settled"] += 1
            if updates:
                assignments = ", ".join(f"{column} = ?" for column in updates)
                self.db.connection.execute(
                    f"UPDATE tickets SET {assignments} WHERE ticket_id = ?",
                    (*updates.values(), ticket["ticket_id"]),
                )
        self.db.connection.commit()
        return counts


def closing_fair_probability(
    odds: list[OddsSnapshot], bet: BetRow, match: Match | None, *, method: str = "shin"
) -> tuple[float, str] | None:
    """Fair closing probability of a recorded bet's selection."""
    return closing_probability(odds, bet.market, bet.selection, bet.placed_at, match, method=method)


def closing_probability(
    odds: list[OddsSnapshot],
    market: str,
    selection: str,
    placed_at: str,
    match: Match | None,
    *,
    method: str = "shin",
) -> tuple[float, str] | None:
    """Fair closing probability of one selection.

    Uses closing snapshots (Pinnacle → Betfair Exchange → consensus). Without
    closing data, the latest pre-kickoff snapshot captured after the bet is a
    proxy (labelled ``*-latest``), but only once kickoff has passed.
    """
    selections = selections_for(market)
    rows = [o for o in odds if o.market == market]
    closing = [o for o in rows if o.is_closing]
    label = ""
    if not closing:
        kickoff = match.kickoff_at if match else None
        if kickoff is None or kickoff > datetime.now(timezone.utc):
            return None
        placed = datetime.fromisoformat(placed_at)
        closing = [o for o in rows if placed <= o.captured_at <= kickoff]
        label = "-latest"
    if not closing:
        return None
    quotes: dict[str, dict[str, float]] = defaultdict(dict)
    latest: dict[str, datetime] = {}
    for row in closing:
        latest[row.bookmaker] = max(latest.get(row.bookmaker, row.captured_at), row.captured_at)
    for row in closing:
        if row.captured_at == latest[row.bookmaker]:
            quotes[row.bookmaker][row.selection] = row.odds
    fair = fair_price(quotes, selections, "auto", method=method)
    if fair is None:
        return None
    return fair.probabilities[selection], fair.source + label


LEG_RETURN: dict[str, float | None] = {"win": None, "push": 1.0, "loss": 0.0, "half_loss": 0.5}


def settle_ticket(ticket_odds: float, stake: float, legs: list[dict[str, Any]]) -> tuple[str, float]:
    """Accumulator settlement: each leg returns odds (win), 1 (push), (1 + odds)/2 (half win),
    0.5 (half loss) or 0 (loss); the payout scales the ticket price by the product of returns
    relative to the product of leg odds (so boosts and quoted ticket prices carry through)."""
    product_odds = math.prod(leg["odds"] for leg in legs)
    returns = []
    for leg in legs:
        result, odds = leg["result"], leg["odds"]
        if result == "win":
            returns.append(odds)
        elif result == "half_win":
            returns.append((1.0 + odds) / 2.0)
        else:
            returns.append(LEG_RETURN[result] or 0.0)
    multiple = math.prod(returns)
    payout = stake * multiple * ticket_odds / product_odds
    if multiple == 0:
        label = "loss"
    elif all(leg["result"] == "push" for leg in legs):
        label = "void"
    else:
        label = "win"
    return label, payout - stake


def summarize_tickets(tickets: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [t for t in tickets if t["result"] is not None]
    staked = sum(t["stake"] for t in settled)
    profit = sum(t["pnl"] or 0.0 for t in settled)
    clv_mean, clv_se = _mean_se([t["clv"] for t in tickets if t["clv"] is not None])
    _, yield_se = _mean_se([(t["pnl"] or 0.0) / t["stake"] for t in settled])
    return {
        "tickets": len(tickets),
        "open": len(tickets) - len(settled),
        "settled": len(settled),
        "won": sum(1 for t in settled if t["result"] == "win"),
        "staked": staked,
        "profit": profit,
        "yield": profit / staked if staked else 0.0,
        "yield_se": yield_se,
        "clv_n": sum(1 for t in tickets if t["clv"] is not None),
        "mean_clv": clv_mean,
        "clv_se": clv_se,
    }


def _mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)


def summarize_bets(rows: list[BetRow]) -> dict[str, Any]:
    """Totals, yield, CLV with standard error, per-market breakdown and a verdict."""
    settled = [b for b in rows if b.result is not None]
    with_clv = [b for b in rows if b.clv is not None]
    staked = sum(b.stake for b in settled)
    pnl = sum(b.pnl or 0.0 for b in settled)
    clv_mean, clv_se = _mean_se([b.clv for b in with_clv])  # type: ignore[misc]
    returns = [(b.pnl or 0.0) / b.stake for b in settled]
    _, yield_se = _mean_se(returns)
    by_market: dict[str, dict[str, float]] = {}
    for kind in ("1x2", "totals", "ah"):
        subset = [b for b in rows if parse_market(b.market)[0] == kind]
        if not subset:
            continue
        sub_settled = [b for b in subset if b.result is not None]
        sub_clv = [b.clv for b in subset if b.clv is not None]
        m, se = _mean_se(sub_clv)
        sub_staked = sum(b.stake for b in sub_settled)
        by_market[kind] = {
            "bets": float(len(subset)),
            "settled": float(len(sub_settled)),
            "yield": sum(b.pnl or 0.0 for b in sub_settled) / sub_staked if sub_staked else 0.0,
            "clv_n": float(len(sub_clv)),
            "mean_clv": m,
            "clv_se": se,
        }
    t_stat = clv_mean / clv_se if clv_se > 0 else 0.0
    if len(with_clv) >= STOP_MIN_BETS and clv_mean + 2 * clv_se < 0:
        verdict = (
            "STOP: closing-line value is significantly negative — the prices you take are worse than fair."
        )
    elif len(with_clv) >= EDGE_MIN_BETS and t_stat > 2:
        verdict = "Evidence of edge: closing-line value is significantly positive."
    elif len(with_clv) < STOP_MIN_BETS:
        verdict = (
            f"Not enough bets with closing prices yet ({len(with_clv)}/{STOP_MIN_BETS}); keep paper trading."
        )
    else:
        verdict = "Inconclusive: CLV is not yet significantly different from zero."
    return {
        "bets": len(rows),
        "open": len(rows) - len(settled),
        "settled": len(settled),
        "staked": staked,
        "profit": pnl,
        "yield": pnl / staked if staked else 0.0,
        "yield_se": yield_se,
        "clv_n": len(with_clv),
        "mean_clv": clv_mean,
        "clv_se": clv_se,
        "clv_t": t_stat,
        "by_market": by_market,
        "verdict": verdict,
    }
