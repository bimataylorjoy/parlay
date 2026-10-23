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
        return summarize_bets(self.bets(kind=kind))


def closing_fair_probability(
    odds: list[OddsSnapshot], bet: BetRow, match: Match | None, *, method: str = "shin"
) -> tuple[float, str] | None:
    """Fair closing probability of the bet's selection.

    Uses closing snapshots (Pinnacle → Betfair Exchange → consensus). Without
    closing data, the latest pre-kickoff snapshot captured after the bet is a
    proxy (labelled ``*-latest``), but only once kickoff has passed.
    """
    selections = selections_for(bet.market)
    rows = [o for o in odds if o.market == bet.market]
    closing = [o for o in rows if o.is_closing]
    label = ""
    if not closing:
        kickoff = match.kickoff_at if match else None
        if kickoff is None or kickoff > datetime.now(timezone.utc):
            return None
        placed = datetime.fromisoformat(bet.placed_at)
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
    return fair.probabilities[bet.selection], fair.source + label


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
