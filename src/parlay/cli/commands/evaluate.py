"""``parlay backtest | compare | tune | calibrate``: leakage-safe model evaluation."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from parlay.cli._common import add_history_source, add_model_options, half_life, iso_date, load_history
from parlay.cli.render import table
from parlay.evaluation.backtest import BacktestConfig, BacktestResult, run_backtest
from parlay.evaluation.calibration import temporally_safe_calibration
from parlay.evaluation.compare import compare_models
from parlay.evaluation.market import calibration_bins, evaluate_flat_stake
from parlay.evaluation.metrics import aggregate_scores, brier_score, log_loss
from parlay.evaluation.serialization import write_json, write_predictions
from parlay.evaluation.temporal import EVALUATION_MODES
from parlay.evaluation.tuning import DEFAULT_HALF_LIVES, DEFAULT_L2_GRID, grid_search, nested_tuning
from parlay.models.team_strength import GOAL_MODELS


def _add_walk_forward(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("walk-forward")
    group.add_argument(
        "--initial-train-days", type=int, default=730, help="first training window (default: 730)"
    )
    group.add_argument(
        "--test-days", type=int, default=7, help="days per test fold; refit cadence (default: 7)"
    )
    group.add_argument("--step-days", type=int, help="advance between folds (default: --test-days)")
    group.add_argument("--evaluation-mode", choices=EVALUATION_MODES, default="rolling_origin")
    group.add_argument("--forecast-lead-minutes", type=int, default=60)
    group.add_argument(
        "--competition",
        action="append",
        help="only score matches of this competition (repeatable; trains on all)",
    )
    group.add_argument("--from", dest="evaluation_start", type=iso_date, help="first evaluated match date")
    group.add_argument("--to", dest="evaluation_end", type=iso_date, help="last evaluated match date")
    group.add_argument(
        "--bookmaker", default="Bet365", help="pre-match prices for edges/EV (default: Bet365)"
    )


def _config(args: argparse.Namespace, **overrides: object) -> BacktestConfig:
    values = dict(
        model=getattr(args, "model", "dixon_coles"),
        estimator=args.estimator,
        half_life_days=args.half_life_days,
        l2_reg=args.l2_reg,
        sot_blend=args.sot_blend,
        initial_train_days=args.initial_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        evaluation_mode=args.evaluation_mode,
        forecast_lead_minutes=args.forecast_lead_minutes,
        competitions=tuple(args.competition) if args.competition else None,
        bookmaker=args.bookmaker,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
    )
    values.update(overrides)
    return BacktestConfig(**values)


def register(commands: argparse._SubParsersAction) -> None:
    backtest = commands.add_parser("backtest", help="walk-forward backtest of one goal model")
    add_history_source(backtest)
    add_model_options(backtest)
    _add_walk_forward(backtest)
    backtest.add_argument("--calibration-mode", choices=("none", "temperature"), default="none")
    backtest.add_argument("--min-edge", type=float, default=0.03, help="strategy: minimum probability edge")
    backtest.add_argument("--min-ev", type=float, default=0.02, help="strategy: minimum expected value")
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs/backtest"))
    backtest.set_defaults(handler=run_backtest_command)

    compare = commands.add_parser("compare", help="backtest several goal models on identical folds")
    add_history_source(compare)
    add_model_options(compare, model=False)
    _add_walk_forward(compare)
    compare.add_argument("--models", nargs="+", choices=GOAL_MODELS, default=list(GOAL_MODELS))
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/compare"))
    compare.set_defaults(handler=run_compare)

    tune = commands.add_parser("tune", help="grid-search half-life and L2 with an untouched holdout")
    add_history_source(tune)
    add_model_options(tune)
    _add_walk_forward(tune)
    tune.add_argument("--half-lives", nargs="+", type=half_life, default=list(DEFAULT_HALF_LIVES))
    tune.add_argument("--l2-grid", nargs="+", type=float, default=list(DEFAULT_L2_GRID))
    tune.add_argument(
        "--holdout-days", type=int, default=365, help="final period kept out of tuning (default: 365)"
    )
    tune.add_argument("--no-holdout", action="store_true", help="plain grid search on all data (optimistic)")
    tune.add_argument("--output-dir", type=Path, default=Path("outputs/tuning"))
    tune.set_defaults(handler=run_tune)

    calibrate = commands.add_parser("calibrate", help="fit temperature scaling on backtest predictions")
    calibrate.add_argument("predictions_csv", type=Path, help="predictions.csv written by `parlay backtest`")
    calibrate.add_argument("--train-end", help="calibration window starts after this forecast timestamp")
    calibrate.add_argument(
        "--calibrate-end", help="calibration window ends at this timestamp; later rows are test"
    )
    calibrate.set_defaults(handler=run_calibrate)


def _print_metrics(result: BacktestResult) -> None:
    m = result.metrics
    if not m.get("n"):
        print("No predictions were produced (check dates / competitions).")
        return
    print(
        f"Scored {m['n']:.0f} matches over {m['folds']:.0f} folds "
        f"({m['skipped_unknown_team']:.0f} skipped: unseen team)"
    )
    rows = [
        ["model 1X2", f"{m['log_loss']:.4f}", f"{m['brier_score']:.4f}", f"{m['rps']:.4f}", f"{m['n']:.0f}"]
    ]
    if "market_log_loss" in m:
        rows.append(
            [
                "  model (priced sample)",
                f"{m['model_log_loss_on_market_sample']:.4f}",
                "",
                "",
                f"{m['market_n']:.0f}",
            ]
        )
        rows.append(["  bookmaker de-vig", f"{m['market_log_loss']:.4f}", "", "", f"{m['market_n']:.0f}"])
    if "closing_log_loss" in m:
        rows.append(
            [
                "  model (closing sample)",
                f"{m['model_log_loss_on_closing_sample']:.4f}",
                "",
                "",
                f"{m['closing_n']:.0f}",
            ]
        )
        rows.append(
            ["  closing line de-vig", f"{m['closing_log_loss']:.4f}", "", "", f"{m['closing_n']:.0f}"]
        )
    rows.append(["over 2.5 (binary)", f"{m['over_2_5_log_loss']:.4f}", "", "", f"{m['n']:.0f}"])
    if "market_over_2_5_log_loss" in m:
        rows.append(
            [
                "  model (priced sample)",
                f"{m['model_over_2_5_log_loss_on_market_sample']:.4f}",
                "",
                "",
                f"{m['totals_market_n']:.0f}",
            ]
        )
        rows.append(
            [
                "  bookmaker de-vig",
                f"{m['market_over_2_5_log_loss']:.4f}",
                "",
                "",
                f"{m['totals_market_n']:.0f}",
            ]
        )
    rows.append(["btts (binary)", f"{m['btts_log_loss']:.4f}", "", "", f"{m['n']:.0f}"])
    print(table(["", "log loss", "brier", "rps", "n"], rows, "lrrrr"))


def run_backtest_command(args: argparse.Namespace) -> int:
    matches, odds = load_history(args)
    config = _config(
        args, calibration_mode=None if args.calibration_mode == "none" else args.calibration_mode
    )
    result = run_backtest(matches, odds, config)
    strategy = evaluate_flat_stake(result.records, min_edge=args.min_edge, min_ev=args.min_ev)
    write_predictions(result.records, args.output_dir / "predictions.csv")
    write_json(
        {
            "metrics": result.metrics,
            "strategy": strategy,
            "metadata": result.metadata,
            "fold_metrics": [asdict(row) for row in result.fold_metrics],
        },
        args.output_dir / "metrics.json",
    )
    write_json(calibration_bins(result.records), args.output_dir / "calibration.json")
    print(f"{config.version}")
    _print_metrics(result)
    if strategy["bets"]:
        clv = (
            f", mean CLV {strategy['mean_clv']:+.2%} (n={strategy['clv_n']:.0f})"
            if "mean_clv" in strategy
            else ""
        )
        print(
            f"Flat-stake strategy (edge >= {args.min_edge}, EV >= {args.min_ev}): "
            f"{strategy['bets']:.0f} bets, "
            f"yield {strategy['yield']:+.2%} ± {strategy['yield_se']:.2%}{clv}"
        )
    print(f"Artifacts written to {args.output_dir}/")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    matches, odds = load_history(args)
    _, summary = compare_models(matches, odds, _config(args), models=tuple(args.models))
    write_json(summary, args.output_dir / "model_comparison.json")
    rows = []
    for row in summary:
        stability: dict[str, float] = row.get("stability_log_loss") or {}  # type: ignore[assignment]
        rows.append(
            [
                row["model"],
                f"{row['n']:.0f}",
                f"{row['log_loss']:.4f}",
                f"{row['brier_score']:.4f}",
                f"{row['rps']:.4f}" if "rps" in row else "",
                f"{stability.get('std', 0.0):.4f}" if stability else "",
            ]
        )
    print(table(["model", "n", "log loss", "brier", "rps", "fold std"], rows))
    print(f"\nFull results: {args.output_dir / 'model_comparison.json'}")
    return 0


def run_tune(args: argparse.Namespace) -> int:
    matches, _ = load_history(args)
    config = _config(args)
    half_lives, l2_grid = tuple(args.half_lives), tuple(args.l2_grid)
    if args.no_holdout:
        rows = grid_search(matches, config, half_lives=half_lives, l2_grid=l2_grid)
        report: dict[str, object] = {"grid": rows}
    else:
        report = nested_tuning(
            matches, config, half_lives=half_lives, l2_grid=l2_grid, holdout_days=args.holdout_days
        )
        rows = report["inner_results"]  # type: ignore[assignment]
    write_json(report, args.output_dir / f"tune_{config.model}.json")
    print(
        table(
            ["half-life", "l2", "n", "log loss", "rps"],
            [
                [
                    str(r["half_life_days"]),
                    f"{r['l2_reg']:g}",
                    f"{r['n']:.0f}",
                    f"{r['log_loss']:.4f}",
                    f"{r['rps']:.4f}",
                ]
                for r in rows
            ],
        )
    )
    if not args.no_holdout:
        holdout: dict[str, float] = report["holdout_metrics"]  # type: ignore[assignment]
        print(f"\nSelected {report['hyperparameters']} on {report['tuned_on']}")
        scores = f"log loss {holdout['log_loss']:.4f}, rps {holdout['rps']:.4f}"
        print(f"Holdout ({report['evaluated_on']}): {scores}")
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    path: Path = args.predictions_csv
    if not path.exists():
        raise SystemExit(f"Predictions CSV not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        records = [
            SimpleNamespace(
                actual=row["actual"],
                forecast_timestamp=row.get("forecast_timestamp", ""),
                home_win=float(row["home_win"]),
                draw=float(row["draw"]),
                away_win=float(row["away_win"]),
            )
            for row in csv.DictReader(handle)
        ]
    if not records:
        raise SystemExit("No records found in CSV")
    if not all(r.forecast_timestamp for r in records):
        raise SystemExit("Predictions CSV must contain forecast_timestamp for leakage-safe calibration")
    if bool(args.train_end) != bool(args.calibrate_end):
        raise SystemExit("--train-end and --calibrate-end must be provided together")
    temperature, calibrated, calibrated_metrics = temporally_safe_calibration(
        records, train_end=args.train_end, calibrate_end=args.calibrate_end
    )
    test_rows = sorted(records, key=lambda r: r.forecast_timestamp)[-len(calibrated) :]
    original = aggregate_scores(
        {
            "log_loss": log_loss({"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual),
            "brier_score": brier_score(
                {"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual
            ),
        }
        for r in test_rows
    )
    print(
        f"Loaded {len(records)} predictions; calibration fitted on an earlier window, "
        f"scored on {len(test_rows)}."
    )
    print(f"Optimal temperature T = {temperature:.4f}")
    print(f"Test log loss  {original['log_loss']:.4f} -> {calibrated_metrics['log_loss']:.4f}")
    print(f"Test Brier     {original['brier_score']:.4f} -> {calibrated_metrics['brier_score']:.4f}")
    if math.isclose(temperature, 1.0, abs_tol=1e-3):
        print("The model is already well calibrated (T ≈ 1).")
    elif temperature > 1.0:
        print("T > 1: the model was over-confident; probabilities are softened toward uniform.")
    else:
        print("T < 1: the model was under-confident; probabilities are sharpened.")
    return 0
