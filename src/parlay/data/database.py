"""SQLite persistence for immutable, time-aware research data."""

from datetime import date, datetime
import sqlite3
from typing import Iterable

from .schemas import Match, OddsSnapshot


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    match_date TEXT NOT NULL,
    kickoff_at TEXT,
    competition TEXT NOT NULL,
    season TEXT NOT NULL,
    home_team_id TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id TEXT NOT NULL REFERENCES teams(team_id),
    home_goals INTEGER,
    away_goals INTEGER,
    home_sot INTEGER,
    away_sot INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_shots INTEGER,
    away_shots INTEGER,
    neutral INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    source_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds REAL NOT NULL,
    captured_at TEXT NOT NULL,
    is_closing INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (match_id, bookmaker, market, selection, captured_at)
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    as_of TEXT NOT NULL,
    feature_set TEXT NOT NULL,
    values_json TEXT NOT NULL,
    PRIMARY KEY (match_id, as_of, feature_set)
);

CREATE TABLE IF NOT EXISTS information_snapshots (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    available_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (match_id, kind, source, available_at)
);

CREATE TABLE IF NOT EXISTS predictions (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    forecast_timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    home_team_id TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id TEXT NOT NULL REFERENCES teams(team_id),
    home_win REAL NOT NULL,
    draw REAL NOT NULL,
    away_win REAL NOT NULL,
    actual TEXT NOT NULL,
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    log_loss REAL NOT NULL,
    brier_score REAL NOT NULL,
    market_home REAL,
    market_draw REAL,
    market_away REAL,
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,
    edge_home REAL,
    edge_draw REAL,
    edge_away REAL,
    ev_home REAL,
    ev_draw REAL,
    ev_away REAL,
    PRIMARY KEY (match_id, forecast_timestamp, model_version)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_odds_match_time ON odds_snapshots(match_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_features_match_time ON feature_snapshots(match_id, as_of);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version, forecast_timestamp);
"""


def _dt(value: date | datetime | str | None) -> str | None:
    return value.isoformat() if value is not None else None


class ResearchDatabase:
    """Small repository abstraction; the schema can later be moved to Postgres."""

    def __init__(self, path: str = ":memory:"):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def upsert_team(self, team_id: str, canonical_name: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO teams(team_id, canonical_name) VALUES (?, ?)",
            (team_id, canonical_name),
        )
        self.connection.commit()

    def insert_matches(self, matches: Iterable[Match], team_ids: dict[str, str] | None = None) -> None:
        rows = list(matches)
        team_ids = team_ids or {name: name for row in rows for name in (row.home_team, row.away_team)}
        for name, team_id in team_ids.items():
            self.upsert_team(team_id, name)
        missing = {
            team
            for row in rows
            for team in (row.home_team, row.away_team)
            if team not in team_ids
        }
        if missing:
            raise ValueError(f"Missing team IDs for: {sorted(missing)}")
        self.connection.executemany(
            """INSERT INTO matches
            (match_id, match_date, competition, season, home_team_id, away_team_id,
             home_goals, away_goals, home_sot, away_sot, home_corners, away_corners,
             home_shots, away_shots, neutral, source, source_timestamp, kickoff_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
              match_date=excluded.match_date,
              kickoff_at=excluded.kickoff_at,
              competition=excluded.competition,
              season=excluded.season,
              home_team_id=excluded.home_team_id,
              away_team_id=excluded.away_team_id,
              home_goals=excluded.home_goals,
              away_goals=excluded.away_goals,
              home_sot=excluded.home_sot,
              away_sot=excluded.away_sot,
              home_corners=excluded.home_corners,
              away_corners=excluded.away_corners,
              home_shots=excluded.home_shots,
              away_shots=excluded.away_shots,
              neutral=excluded.neutral,
              source=excluded.source,
              source_timestamp=excluded.source_timestamp""",
            [(
                row.match_id, _dt(row.date), row.competition, row.season,
                team_ids[row.home_team], team_ids[row.away_team], row.home_goals,
                row.away_goals, row.home_sot, row.away_sot, row.home_corners, row.away_corners,
                row.home_shots, row.away_shots, int(row.neutral), row.source, _dt(row.source_timestamp), _dt(row.kickoff_at),
            ) for row in rows],
        )
        self.connection.commit()

    def insert_odds(self, odds: Iterable[OddsSnapshot]) -> None:
        self.connection.executemany(
            """INSERT OR REPLACE INTO odds_snapshots
            (match_id, bookmaker, market, selection, odds, captured_at, is_closing)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(
                row.match_id, row.bookmaker, row.market, row.selection, row.odds,
                _dt(row.captured_at), int(row.is_closing),
            ) for row in odds],
        )
        self.connection.commit()

    def insert_future_matches(self, matches: Iterable[Match], team_ids: dict[str, str] | None = None) -> None:
        """Insert scheduled matches without inventing a final result."""
        rows = list(matches)
        if any(row.home_goals is not None or row.away_goals is not None for row in rows):
            raise ValueError("future matches must not contain a final score")
        self.insert_matches(rows, team_ids)

    def insert_feature_snapshot(self, match_id: str, as_of: datetime,
                                feature_set: str, values_json: str) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO feature_snapshots
            (match_id, as_of, feature_set, values_json) VALUES (?, ?, ?, ?)""",
            (match_id, _dt(as_of), feature_set, values_json),
        )
        self.connection.commit()

    def insert_information_snapshot(
        self, match_id: str, kind: str, source: str, available_at: datetime,
        payload_json: str,
    ) -> None:
        """Persist raw enrichment without making it part of model inputs."""
        self.connection.execute(
            """INSERT OR REPLACE INTO information_snapshots
            (match_id, kind, source, available_at, payload_json)
            VALUES (?, ?, ?, ?, ?)""",
            (match_id, kind, source, _dt(available_at), payload_json),
        )
        self.connection.commit()

    def information_as_of(self, match_id: str, as_of: datetime) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            """SELECT * FROM information_snapshots
            WHERE match_id = ? AND available_at <= ? ORDER BY available_at""",
            (match_id, _dt(as_of)),
        ))

    def matches_as_of(self, as_of: date) -> list[sqlite3.Row]:
        """Return matches whose result was known by the supplied date."""
        return list(self.connection.execute(
            "SELECT * FROM matches WHERE match_date <= ? ORDER BY match_date, match_id",
            (_dt(as_of),),
        ))

    def load_matches(self, *, as_of: date | datetime | None = None, information_set=None) -> list[Match]:
        """Rehydrate domain matches. Supports legacy `as_of: date` or `InformationSet`."""
        # Backward compat: if information_set provided, use it to filter knowable results
        if information_set is not None:
            # Load all then filter via InformationSet (centralized eligibility)
            all_matches = self.load_matches()
            return information_set.filter_knowable(all_matches)
        query = """SELECT m.*, h.canonical_name AS home_name,
                   a.canonical_name AS away_name
                   FROM matches m
                   JOIN teams h ON h.team_id = m.home_team_id
                   JOIN teams a ON a.team_id = m.away_team_id"""
        params: tuple[object, ...] = ()
        if as_of is not None:
            # If as_of is datetime, filter via kickoff_aware logic conservatively via date
            if isinstance(as_of, datetime):
                # For datetime, still filter by date <= as_of.date() as conservative pre-filter,
                # then caller should apply InformationSet for exact kickoff+lag
                query += " WHERE m.match_date <= ?"
                params = (_dt(as_of.date()),)
            else:
                query += " WHERE m.match_date <= ?"
                params = (_dt(as_of),)
        query += " ORDER BY m.match_date, m.match_id"
        rows = self.connection.execute(query, params)
        return [Match(
            match_id=row["match_id"],
            date=date.fromisoformat(row["match_date"]),
            competition=row["competition"],
            season=row["season"],
            home_team=row["home_name"],
            away_team=row["away_name"],
            home_goals=row["home_goals"],
            away_goals=row["away_goals"],
            home_sot=row["home_sot"] if "home_sot" in row.keys() else None,
            away_sot=row["away_sot"] if "away_sot" in row.keys() else None,
            home_corners=row["home_corners"] if "home_corners" in row.keys() else None,
            away_corners=row["away_corners"] if "away_corners" in row.keys() else None,
            home_shots=row["home_shots"] if "home_shots" in row.keys() else None,
            away_shots=row["away_shots"] if "away_shots" in row.keys() else None,
            neutral=bool(row["neutral"]),
            source=row["source"],
            source_timestamp=datetime.fromisoformat(row["source_timestamp"]) if row["source_timestamp"] else None,
            kickoff_at=datetime.fromisoformat(row["kickoff_at"]) if row["kickoff_at"] else None,
        ) for row in rows]

    def load_odds(self, *, match_id: str | None = None) -> list[OddsSnapshot]:
        query = "SELECT * FROM odds_snapshots"
        params: tuple[object, ...] = ()
        if match_id is not None:
            query += " WHERE match_id = ?"
            params = (match_id,)
        query += " ORDER BY captured_at, match_id, selection"
        rows = self.connection.execute(query, params)
        return [OddsSnapshot(
            match_id=row["match_id"],
            bookmaker=row["bookmaker"],
            market=row["market"],
            selection=row["selection"],
            odds=float(row["odds"]),
            captured_at=datetime.fromisoformat(row["captured_at"]),
            is_closing=bool(row["is_closing"]),
        ) for row in rows]

    def odds_as_of(self, match_id: str, as_of: datetime) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            """SELECT * FROM odds_snapshots
            WHERE match_id = ? AND captured_at <= ? ORDER BY captured_at""",
            (match_id, _dt(as_of)),
        ))

    def feature_as_of(self, match_id: str, feature_set: str, as_of: datetime) -> sqlite3.Row | None:
        return self.connection.execute(
            """SELECT * FROM feature_snapshots
            WHERE match_id = ? AND feature_set = ? AND as_of <= ?
            ORDER BY as_of DESC LIMIT 1""",
            (match_id, feature_set, _dt(as_of)),
        ).fetchone()

    def insert_predictions(self, records: Iterable[object], team_ids: dict[str, str] | None = None) -> None:
        rows = list(records)
        team_ids = team_ids or {
            name: name for row in rows for name in (row.home_team, row.away_team)
        }
        for name, team_id in team_ids.items():
            self.upsert_team(team_id, name)
        self.connection.executemany(
            """INSERT OR REPLACE INTO predictions
            (match_id, forecast_timestamp, model_version, home_team_id, away_team_id,
            home_win, draw, away_win, actual, home_goals, away_goals, log_loss, brier_score,
            market_home, market_draw, market_away, odds_home, odds_draw, odds_away,
            edge_home, edge_draw, edge_away, ev_home, ev_draw, ev_away)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                row.match_id, row.forecast_timestamp, row.model_version,
                team_ids[row.home_team], team_ids[row.away_team], row.home_win,
                row.draw, row.away_win, row.actual, row.home_goals, row.away_goals,
                row.log_loss, row.brier_score, row.market_home, row.market_draw,
                row.market_away, row.odds_home, row.odds_draw, row.odds_away,
                row.edge_home, row.edge_draw, row.edge_away, row.ev_home, row.ev_draw,
                row.ev_away,
            ) for row in rows],
        )
        self.connection.commit()
