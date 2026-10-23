"""``parlay predict``: price upcoming fixtures from a CSV and list value bets."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from parlay.cli._common import add_database, add_model_options, load_history
from parlay.cli.render import fixture_detail, fixture_summary
from parlay.data.loaders import parse_date
from parlay.data.schemas import Match
from parlay.data.teams import canonical_team_name
from parlay.evaluation.serialization import write_rows
from parlay.models.corners import CornerStrengthModel, fit_corner_strength
from parlay.models.team_strength import DEFAULT_SOT_BLEND, TeamStrengthModel, fit_team_strength
from parlay.prediction.fixture import (
    Fixture,
    FixturePrediction,
    PredictConfig,
    parse_odds_columns,
    predict_fixture,
)
from parlay.prediction.markets import DE_VIG_METHODS
from parlay.prediction.odds import ODDS_FORMATS


def register(commands: argparse._SubParsersAction) -> None:
    predict = commands.add_parser(
        "predict",
        help="price upcoming fixtures (Date,HomeTeam,AwayTeam[,odds...]) and list value bets",
        description="Fixture CSV columns: Date,HomeTeam,AwayTeam (required) plus optional bookmaker odds, "
        "e.g. B365H/B365D/B365A, B365>2.5/B365<2.5 (any line), B365BTTS_Y/N, B365CRN>9.5/B365CRN<9.5 "
        "(corners), AHh with B365AHH/B365AHA (Asian handicap), B3651X/B365X2/B36512.",
    )
    predict.add_argument("fixtures", type=Path, help="fixtures CSV")
    add_database(predict)
    add_model_options(predict)
    staking = predict.add_argument_group("value & staking")
    staking.add_argument("--min-ev", type=float, default=0.02, help="minimum expected value (default: 0.02)")
    staking.add_argument(
        "--min-edge", type=float, default=0.0, help="minimum edge over de-vigged market (default: 0)"
    )
    staking.add_argument(
        "--kelly-fraction", type=float, default=0.25, help="Kelly multiplier (default: 0.25)"
    )
    staking.add_argument(
        "--max-stake", type=float, default=0.005, help="bankroll cap per bet (default: 0.005)"
    )
    staking.add_argument("--de-vig", choices=DE_VIG_METHODS, default="shin")
    odds = predict.add_argument_group("odds input")
    odds.add_argument("--odds-prefix", default="B365", help="bookmaker column prefix (default: B365)")
    odds.add_argument(
        "--odds-format", choices=ODDS_FORMATS, default="decimal", help="price format in the CSV"
    )
    output = predict.add_argument_group("output")
    output.add_argument("--goal-lines", nargs="+", type=float, default=list(PredictConfig().goal_lines))
    output.add_argument("--corner-lines", nargs="+", type=float, default=list(PredictConfig().corner_lines))
    output.add_argument("--no-corners", action="store_true", help="skip the corner model")
    output.add_argument("--summary", action="store_true", help="print only the summary table")
    output.add_argument("--output", type=Path, help="write a flat CSV with every market")
    predict.set_defaults(handler=run_predict, csv=[], csv_competition="EPL")


def read_fixtures(path: Path, *, prefix: str, odds_format: str) -> list[Fixture]:
    if not path.exists():
        raise SystemExit(f"Fixtures CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    missing = {"Date", "HomeTeam", "AwayTeam"} - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"Fixtures CSV is missing columns: {sorted(missing)}")
    fixtures = []
    for number, row in enumerate(reader, start=2):
        try:
            fixture_date = parse_date(row["Date"])
            odds = parse_odds_columns(row, prefix=prefix, odds_format=odds_format)
        except ValueError as exc:
            raise SystemExit(f"{path}:{number}: {exc}") from exc
        home_raw, away_raw = row["HomeTeam"].strip(), row["AwayTeam"].strip()
        fixtures.append(
            Fixture(
                match_date=fixture_date,
                home_team=canonical_team_name(home_raw),
                away_team=canonical_team_name(away_raw),
                home_raw=home_raw,
                away_raw=away_raw,
                odds=odds,
            )
        )
    if not fixtures:
        raise SystemExit("Fixtures CSV is empty")
    return fixtures


def predict_fixtures(
    fixtures: Iterable[Fixture],
    history: list[Match],
    *,
    model: str,
    estimator: str,
    half_life_days: float | None,
    l2_reg: float,
    config: PredictConfig,
    sot_blend: float = DEFAULT_SOT_BLEND,
    corners: bool = True,
) -> list[FixturePrediction]:
    """Fit once per fixture date on strictly earlier matches, then predict each fixture."""
    goal_models: dict[date, TeamStrengthModel | None] = {}
    corner_models: dict[date, CornerStrengthModel | None] = {}
    predictions = []
    for fixture in fixtures:
        day = fixture.match_date
        if day not in goal_models:
            training = [m for m in history if m.date < day]
            goal_models[day] = (
                fit_team_strength(
                    training,
                    model=model,
                    estimator=estimator,
                    as_of=day,
                    half_life_days=half_life_days,
                    l2_reg=l2_reg,
                    sot_blend=sot_blend,
                )
                if training
                else None
            )
            corner_models[day] = fit_corner_strength(training, as_of=day) if corners and training else None
        goal_model = goal_models[day]
        if goal_model is None:
            predictions.append(
                FixturePrediction(
                    fixture, status="no_history", message="no completed matches before this date"
                )
            )
            continue
        predictions.append(predict_fixture(fixture, goal_model, corner_models[day], config))
    return predictions


def run_predict(args: argparse.Namespace) -> int:
    fixtures = read_fixtures(args.fixtures, prefix=args.odds_prefix, odds_format=args.odds_format)
    history, _ = load_history(args)
    config = PredictConfig(
        goal_lines=tuple(args.goal_lines),
        corner_lines=tuple(args.corner_lines),
        min_ev=args.min_ev,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        de_vig_method=args.de_vig,
    )
    predictions = predict_fixtures(
        fixtures,
        history,
        model=args.model,
        estimator=args.estimator,
        half_life_days=args.half_life_days,
        l2_reg=args.l2_reg,
        sot_blend=args.sot_blend,
        config=config,
        corners=not args.no_corners,
    )
    print(
        f"\n{len(predictions)} fixtures · model {args.model} "
        f"(half-life {args.half_life_days}, l2 {args.l2_reg:g})\n"
    )
    print(fixture_summary(predictions))
    if not args.summary:
        for prediction in predictions:
            print()
            print(fixture_detail(prediction))
    if args.output:
        write_rows([p.to_row() for p in predictions], args.output)
        print(f"\nWrote {len(predictions)} rows to {args.output}")
    print(
        "\nResearch output only: compare fair odds with your own price; never stake more than you can lose."
    )
    return 0
