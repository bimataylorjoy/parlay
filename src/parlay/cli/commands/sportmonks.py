"""``parlay sync-sportmonks`` and ``parlay predict-sportmonks``."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from parlay.cli._common import add_database, add_model_options, iso_date
from parlay.cli.commands.predict import predict_fixtures
from parlay.cli.render import fixture_summary
from parlay.data.database import ResearchDatabase
from parlay.evaluation.serialization import write_rows
from parlay.prediction.fixture import Fixture, PredictConfig


def register(commands: argparse._SubParsersAction) -> None:
    sync = commands.add_parser("sync-sportmonks", help="sync upcoming fixtures, O/U 2.5 odds and enrichment")
    add_database(sync)
    sync.add_argument("--token", default=None, help="API token (prefer the SPORTMONKS_API_TOKEN variable)")
    sync.add_argument("--start-date", type=iso_date, required=True)
    sync.add_argument("--end-date", type=iso_date, required=True)
    sync.add_argument("--competition", choices=("EPL", "Championship"), default="EPL")
    sync.add_argument("--bookmaker-id", type=int, default=2)
    sync.set_defaults(handler=run_sync)

    live = commands.add_parser("predict-sportmonks", help="predict synced Sportmonks fixtures")
    add_database(live)
    add_model_options(live)
    live.add_argument("--min-ev", type=float, default=0.02)
    live.add_argument("--kelly-fraction", type=float, default=0.25)
    live.add_argument("--max-stake", type=float, default=0.005)
    live.add_argument("--output", type=Path, default=Path("outputs/sportmonks_predictions.csv"))
    live.set_defaults(handler=run_predict_sportmonks)


def run_sync(args: argparse.Namespace) -> int:
    from parlay.data.sportmonks import (
        LEAGUE_IDS,
        extract_totals_25,
        fetch_fixture_enrichment,
        fetch_fixtures_with_odds,
        fixture_to_match,
        parse_fixture_information,
        totals_to_odds,
    )

    token = args.token or os.environ.get("SPORTMONKS_API_TOKEN")
    if not token:
        raise SystemExit("Set SPORTMONKS_API_TOKEN or pass --token")
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must not be before --start-date")
    days = tuple(
        date.fromordinal(d) for d in range(args.start_date.toordinal(), args.end_date.toordinal() + 1)
    )
    fixtures = fetch_fixtures_with_odds(token, days, league_id=LEAGUE_IDS[args.competition])
    database = ResearchDatabase(args.database)
    match_rows, odds_rows = [], []
    for fixture in fixtures:
        totals = extract_totals_25(fixture, bookmaker_id=args.bookmaker_id)
        if totals is None:
            continue
        match_rows.append(fixture_to_match(fixture, competition=args.competition))
        odds_rows.extend(totals_to_odds(fixture, totals, bookmaker=f"sportmonks:{args.bookmaker_id}"))
    database.insert_future_matches(match_rows)
    database.insert_odds(odds_rows)
    enriched = 0
    for fixture in fixtures:
        try:
            enrichment = parse_fixture_information(fetch_fixture_enrichment(token, fixture["id"]))
        except (OSError, ValueError, KeyError):
            continue
        for kind, values in enrichment.items():
            for value in values:
                database.insert_information_snapshot(
                    f"sportmonks:{fixture['id']}",
                    kind,
                    value.source,  # type: ignore[attr-defined]
                    value.available_at,  # type: ignore[attr-defined]
                    json.dumps(getattr(value, "__dict__", None) or str(value), default=str, sort_keys=True),
                )
                enriched += 1
    database.close()
    print(
        f"Synced {len(match_rows)} fixtures, {len(odds_rows)} totals odds and {enriched} enrichment records"
    )
    return 0


def run_predict_sportmonks(args: argparse.Namespace) -> int:
    database = ResearchDatabase(args.database)
    rows = database.load_matches()
    history = [m for m in rows if m.home_goals is not None and m.away_goals is not None]
    scheduled = [m for m in rows if m.match_id.startswith("sportmonks:") and m.home_goals is None]
    if not scheduled:
        raise SystemExit("No scheduled Sportmonks fixtures found; run `parlay sync-sportmonks` first")
    if not history:
        raise SystemExit("No completed matches available for training")
    fixtures = []
    for match in scheduled:
        prices = {
            o.selection: o.odds
            for o in database.load_odds(match_id=match.match_id)
            if o.market == "totals_2.5"
        }
        fixtures.append(
            Fixture(
                match_date=match.date,
                home_team=match.home_team,
                away_team=match.away_team,
                odds={("totals", side, 2.5): price for side, price in prices.items()},
            )
        )
    database.close()
    config = PredictConfig(min_ev=args.min_ev, kelly_fraction=args.kelly_fraction, max_stake=args.max_stake)
    predictions = predict_fixtures(
        fixtures,
        history,
        model=args.model,
        estimator=args.estimator,
        half_life_days=args.half_life_days,
        l2_reg=args.l2_reg,
        config=config,
        corners=False,
    )
    print(fixture_summary(predictions))
    write_rows([p.to_row() for p in predictions], args.output)
    print(f"\nWrote {len(predictions)} predictions to {args.output}")
    return 0
