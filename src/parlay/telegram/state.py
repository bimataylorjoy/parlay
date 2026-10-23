"""Per-chat bot state stored next to the ledger: bankroll and the parlay slip."""

from __future__ import annotations

import os

from parlay.data.database import ResearchDatabase

SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_settings (
    chat_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (chat_id, key)
);
CREATE TABLE IF NOT EXISTS bot_slips (
    chat_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, candidate_key)
);
"""

DEFAULT_BANKROLL = 1_000_000.0


class BotState:
    def __init__(self, database: ResearchDatabase):
        self.db = database
        self.db.connection.executescript(SCHEMA)

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

    def slip(self, chat_id: str) -> list[str]:
        rows = self.db.connection.execute(
            "SELECT candidate_key FROM bot_slips WHERE chat_id = ? ORDER BY added_at, candidate_key",
            (chat_id,),
        )
        return [row["candidate_key"] for row in rows]

    def add_to_slip(self, chat_id: str, key: str) -> None:
        self.db.connection.execute(
            "INSERT OR IGNORE INTO bot_slips(chat_id, candidate_key) VALUES (?, ?)", (chat_id, key)
        )
        self.db.connection.commit()

    def remove_from_slip(self, chat_id: str, key: str) -> None:
        self.db.connection.execute(
            "DELETE FROM bot_slips WHERE chat_id = ? AND candidate_key = ?", (chat_id, key)
        )
        self.db.connection.commit()

    def clear_slip(self, chat_id: str) -> None:
        self.db.connection.execute("DELETE FROM bot_slips WHERE chat_id = ?", (chat_id,))
        self.db.connection.commit()
