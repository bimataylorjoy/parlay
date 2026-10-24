"""Per-chat bot state stored next to the ledger: bankroll and the parlay slip."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from parlay.data.database import ResearchDatabase

SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_settings (
    chat_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (chat_id, key)
);
CREATE TABLE IF NOT EXISTS bot_slip_legs (
    chat_id TEXT NOT NULL,
    match_id TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, match_id)
);
"""

DEFAULT_BANKROLL = 1_000_000.0


@dataclass(frozen=True, slots=True)
class SlipLeg:
    match_id: str
    market: str
    selection: str
    bookmaker: str


class BotState:
    def __init__(self, database: ResearchDatabase):
        self.db = database
        self.db.connection.executescript(SCHEMA)
        self._migrate_old_slips()

    def _migrate_old_slips(self) -> None:
        """Earlier versions stored value-bet keys in ``bot_slips``; convert them to legs once."""
        exists = self.db.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'bot_slips'"
        ).fetchone()
        if not exists:
            return
        for row in self.db.connection.execute(
            """SELECT s.chat_id, c.payload_json FROM bot_slips s
            JOIN value_candidates c ON c.key = s.candidate_key ORDER BY s.added_at"""
        ).fetchall():
            data = json.loads(row["payload_json"])
            self.add_leg(
                row["chat_id"],
                SlipLeg(data["match_id"], data["market"], data["selection"], data["bookmaker"]),
            )
        self.db.connection.execute("DROP TABLE bot_slips")
        self.db.connection.commit()

    # -- bankroll -----------------------------------------------------------
    def bankroll(self, chat_id: str) -> float:
        row = self.db.connection.execute(
            "SELECT value FROM bot_settings WHERE chat_id = ? AND key = 'bankroll'", (chat_id,)
        ).fetchone()
        if row:
            return float(row["value"])
        return float(os.environ.get("PARLAY_BANKROLL", DEFAULT_BANKROLL))

    def set_bankroll(self, chat_id: str, amount: float) -> None:
        if amount <= 0:
            raise ValueError("bankroll must be positive")
        self.db.connection.execute(
            "INSERT OR REPLACE INTO bot_settings(chat_id, key, value) VALUES (?, 'bankroll', ?)",
            (chat_id, str(amount)),
        )
        self.db.connection.commit()

    # -- parlay slip (one leg per match) -------------------------------------
    def slip(self, chat_id: str) -> list[SlipLeg]:
        rows = self.db.connection.execute(
            "SELECT match_id, market, selection, bookmaker FROM bot_slip_legs WHERE chat_id = ? "
            "ORDER BY added_at, match_id",
            (chat_id,),
        )
        return [SlipLeg(r["match_id"], r["market"], r["selection"], r["bookmaker"]) for r in rows]

    def add_leg(self, chat_id: str, leg: SlipLeg) -> bool:
        """Add (or replace) the leg for this match; returns False when the match was already in the slip."""
        existed = self.db.connection.execute(
            "SELECT 1 FROM bot_slip_legs WHERE chat_id = ? AND match_id = ?", (chat_id, leg.match_id)
        ).fetchone()
        self.db.connection.execute(
            """INSERT OR REPLACE INTO bot_slip_legs(chat_id, match_id, market, selection, bookmaker)
            VALUES (?, ?, ?, ?, ?)""",
            (chat_id, leg.match_id, leg.market, leg.selection, leg.bookmaker),
        )
        self.db.connection.commit()
        return existed is None

    def remove_leg(self, chat_id: str, match_id: str) -> None:
        self.db.connection.execute(
            "DELETE FROM bot_slip_legs WHERE chat_id = ? AND match_id = ?", (chat_id, match_id)
        )
        self.db.connection.commit()

    def clear_slip(self, chat_id: str) -> None:
        self.db.connection.execute("DELETE FROM bot_slip_legs WHERE chat_id = ?", (chat_id,))
        self.db.connection.commit()
