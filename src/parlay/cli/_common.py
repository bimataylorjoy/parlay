"""Shared CLI helpers: argument groups and history loading."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from parlay.data.database import ResearchDatabase
from parlay.data.loaders import load_many
from parlay.data.schemas import Match, OddsSnapshot
from parlay.data.sources import acquire_csv
from parlay.models.team_strength import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_L2_REG,
    DEFAULT_SOT_BLEND,
    ESTIMATORS,
    GOAL_MODELS,
)

DEFAULT_DATABASE = "data/parlay.sqlite"
DEFAULT_SNAPSHOT_DIR = "data/raw/snapshots"


def half_life(value: str) -> float | None:
    """``none``/``inf`` disables time decay."""
    if value.strip().lower() in {"none", "inf", "off"}:
        return None
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("half-life must be positive (or 'none')")
    return number


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def add_database(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database", default=DEFAULT_DATABASE, help=f"SQLite database (default: {DEFAULT_DATABASE})"
    )


def add_model_options(parser: argparse.ArgumentParser, *, model: bool = True) -> None:
    group = parser.add_argument_group("model")
    if model:
        group.add_argument(
            "--model", choices=GOAL_MODELS, default="dixon_coles", help="goal model (default: dixon_coles)"
        )
    group.add_argument("--estimator", choices=ESTIMATORS, default="mle", help="estimator (default: mle)")
    group.add_argument(
        "--half-life-days",
        type=half_life,
        default=DEFAULT_HALF_LIFE_DAYS,
        help=f"time-decay half-life in days, or 'none' (default: {DEFAULT_HALF_LIFE_DAYS:g})",
    )
    group.add_argument(
        "--l2-reg",
        type=float,
        default=DEFAULT_L2_REG,
        help=f"precision of the N(0, 1/l2) prior on team strengths (default: {DEFAULT_L2_REG:g})",
    )
    group.add_argument(
        "--sot-blend",
        type=float,
        default=DEFAULT_SOT_BLEND,
        help=f"weight of shots-on-target strengths, 0 = goals only (default: {DEFAULT_SOT_BLEND:g})",
    )


def add_history_source(parser: argparse.ArgumentParser) -> None:
    add_database(parser)
    parser.add_argument(
        "csv", nargs="*", type=Path, help="football-data CSV files to use instead of the database (optional)"
    )
    parser.add_argument(
        "--csv-competition", default="EPL", help="competition label for CSV inputs (default: EPL)"
    )


def load_history(args: argparse.Namespace) -> tuple[list[Match], list[OddsSnapshot]]:
    """Completed matches and odds from CSV arguments, or else from the database."""
    if args.csv:
        snapshots = [acquire_csv(source, DEFAULT_SNAPSHOT_DIR)[0] for source in args.csv]
        matches, odds = load_many(snapshots, competition=args.csv_competition)
    else:
        if not Path(args.database).exists():
            raise SystemExit(
                f"Database {args.database} not found; run `parlay fetch` or `parlay ingest` first"
            )
        database = ResearchDatabase(args.database)
        matches, odds = database.load_matches(), database.load_odds()
        database.close()
    completed = [m for m in matches if m.home_goals is not None and m.away_goals is not None]
    if not completed:
        raise SystemExit("No completed matches available")
    return completed, odds
