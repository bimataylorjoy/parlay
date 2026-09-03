"""Command line entry points for local research experiments."""

import argparse
from dataclasses import asdict
from pathlib import Path

from parlay.data.database import ResearchDatabase
from parlay.data.ingestion import ingest_csv_files
from parlay.data.loaders import load_many
from parlay.data.sources import acquire_csv
from parlay.evaluation.backtest import run_backtest
from parlay.evaluation.compare import compare_models
from parlay.evaluation.market import calibration_bins, evaluate_flat_stake
from parlay.evaluation.metrics import aggregate_scores, log_loss, brier_score
from parlay.evaluation.serialization import write_json, write_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parlay")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="ingest football-data CSV files")
    ingest.add_argument("csv", nargs="+", type=Path)
    ingest.add_argument("--database", default="data/parlay.sqlite")
    ingest.add_argument("--manifest", default="data/snapshots/manifest.json")
    ingest.add_argument("--competition", default="EPL")
    ingest.add_argument("--season", default="unknown")
    ingest.add_argument("--feature-window", type=int, default=5)
    ingest.add_argument("--snapshot-dir", default="data/raw/snapshots")

    backtest = commands.add_parser("backtest", help="run expanding-window backtest")
    backtest.add_argument("csv", nargs="*", type=Path, help="CSV inputs when --database is not populated")
    backtest.add_argument("--model", choices=("poisson", "dixon_coles", "negative_binomial"), default="poisson")
    backtest.add_argument("--estimator", choices=("mle", "heuristic", "bayesian_hmc"), default="mle")
    backtest.add_argument("--competition", default="EPL")
    backtest.add_argument("--season", default="unknown")
    backtest.add_argument("--initial-train-days", type=int, default=730)
    backtest.add_argument("--test-days", type=int, default=30)
    backtest.add_argument("--step-days", type=int)
    backtest.add_argument("--half-life-days", type=float, default=730.0)
    backtest.add_argument("--sot-weight", type=float, default=0.0)
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs/backtest"))
    backtest.add_argument("--database", default="data/parlay.sqlite")
    backtest.add_argument("--bookmaker", default="Bet365")
    backtest.add_argument("--forecast-lead-minutes", type=int, default=60)
    backtest.add_argument("--min-edge", type=float, default=0.03)
    backtest.add_argument("--min-ev", type=float, default=0.02)

    compare = commands.add_parser("compare", help="compare all baseline models")
    compare.add_argument("csv", nargs="*", type=Path)
    compare.add_argument("--database", default="data/parlay.sqlite")
    compare.add_argument("--estimator", choices=("mle", "heuristic", "bayesian_hmc"), default="mle")
    compare.add_argument("--competition", default="EPL")
    compare.add_argument("--season", default="unknown")
    compare.add_argument("--initial-train-days", type=int, default=730)
    compare.add_argument("--test-days", type=int, default=30)
    compare.add_argument("--step-days", type=int)
    compare.add_argument("--half-life-days", type=float, default=730.0)
    compare.add_argument("--sot-weight", type=float, default=0.0)
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/compare"))
    compare.add_argument("--bookmaker", default="Bet365")

    tune = commands.add_parser("tune", help="hyperparameter grid search for time decay")
    tune.add_argument("csv", nargs="*", type=Path)
    tune.add_argument("--database", default="data/parlay.sqlite")
    tune.add_argument("--model", choices=("poisson", "dixon_coles", "negative_binomial"), default="poisson")
    tune.add_argument("--estimator", choices=("mle", "heuristic", "bayesian_hmc"), default="mle")
    tune.add_argument("--competition", default="EPL")
    tune.add_argument("--season", default="unknown")
    tune.add_argument("--initial-train-days", type=int, default=730)
    tune.add_argument("--test-days", type=int, default=30)
    tune.add_argument("--step-days", type=int)
    tune.add_argument("--output-dir", type=Path, default=Path("outputs/tuning"))

    predict = commands.add_parser("predict", help="generate predictions and betting tips for upcoming matches")
    predict.add_argument("fixtures", type=Path, help="CSV containing upcoming fixtures (Date,HomeTeam,AwayTeam,[B365H,B365D,B365A])")
    predict.add_argument("--database", default="data/parlay.sqlite")
    predict.add_argument("--model", choices=("poisson", "dixon_coles", "negative_binomial"), default="poisson")
    predict.add_argument("--estimator", choices=("mle", "heuristic", "bayesian_hmc"), default="mle")
    predict.add_argument("--half-life-days", type=float, default=365.0)
    predict.add_argument("--sot-weight", type=float, default=0.5)
    predict.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly multiplier (e.g. 0.25 for quarter Kelly)")
    predict.add_argument("--max-stake", type=float, default=0.005, help="Maximum bankroll fraction per selection")
    predict.add_argument("--min-ev", type=float, default=0.02, help="Minimum expected value required for a recommendation")
    predict.add_argument("--output", type=Path, help="Optional CSV output for auditable predictions")
    sync = commands.add_parser("sync-sportmonks", help="sync upcoming EPL fixtures and totals odds")
    sync.add_argument("--database", default="data/parlay.sqlite")
    sync.add_argument("--token", default=None, help="Sportmonks token; prefer SPORTMONKS_API_TOKEN")
    sync.add_argument("--start-date", required=True)
    sync.add_argument("--end-date", required=True)
    sync.add_argument("--bookmaker-id", type=int, default=2)
    live = commands.add_parser("predict-sportmonks", help="predict synced Sportmonks fixtures")
    live.add_argument("--database", default="data/parlay.sqlite")
    live.add_argument("--model", choices=("poisson", "dixon_coles", "negative_binomial"), default="dixon_coles")
    live.add_argument("--estimator", choices=("mle", "heuristic"), default="mle")
    live.add_argument("--half-life-days", type=float, default=365.0)
    live.add_argument("--min-ev", type=float, default=0.02)
    live.add_argument("--kelly-fraction", type=float, default=0.25)
    live.add_argument("--max-stake", type=float, default=0.005)
    live.add_argument("--output", type=Path, default=Path("outputs/sportmonks_predictions.csv"))

    calibrate = commands.add_parser("calibrate", help="find optimal temperature scaling on backtest predictions")
    calibrate.add_argument("predictions_csv", type=Path, help="CSV output from a previous backtest")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        snapshots = [acquire_csv(source, args.snapshot_dir)[0] for source in args.csv]
        database = ResearchDatabase(args.database)
        manifest = ingest_csv_files(
            snapshots, database, competition=args.competition, season=args.season,
            feature_window=args.feature_window, manifest_path=args.manifest,
        )
        print(f"Ingested {manifest['match_count']} matches and {manifest['odds_count']} odds snapshots")
        return 0

    if args.command == "calibrate":
        import csv
        import math
        from parlay.evaluation.calibration import find_optimal_temperature, apply_calibration
        from types import SimpleNamespace
        from parlay.evaluation.metrics import aggregate_scores
        
        path = args.predictions_csv
        if not path.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {path}")
            
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(SimpleNamespace(
                    actual=row["actual"],
                    home_win=float(row["home_win"]),
                    draw=float(row["draw"]),
                    away_win=float(row["away_win"]),
                ))
                
        if not records:
            print("No records found in CSV.")
            return 1
            
        original_metrics = aggregate_scores(
            [{"log_loss": log_loss({"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual),
              "brier_score": brier_score({"home_win": r.home_win, "draw": r.draw, "away_win": r.away_win}, r.actual)} 
             for r in records]
        )
        
        print(f"Loaded {len(records)} predictions.")
        print(f"Original Log Loss : {original_metrics['log_loss']:.4f}")
        print(f"Original Brier    : {original_metrics['brier_score']:.4f}")
        
        T_opt = find_optimal_temperature(records)
        print(f"\nOptimal Temperature T = {T_opt:.4f}")
        
        if math.isclose(T_opt, 1.0, abs_tol=1e-4):
            print("Model is already well-calibrated (T ~ 1.0).")
            return 0
            
        scaled_records = apply_calibration(records, T_opt)
        scaled_metrics = aggregate_scores(scaled_records)
        
        print(f"Calibrated Log Loss : {scaled_metrics['log_loss']:.4f} (diff: {scaled_metrics['log_loss'] - original_metrics['log_loss']:+.4f})")
        print(f"Calibrated Brier    : {scaled_metrics['brier_score']:.4f} (diff: {scaled_metrics['brier_score'] - original_metrics['brier_score']:+.4f})")
        
        if T_opt > 1.0:
            print("Interpretation: Model was OVERCONFIDENT. Probabilities have been softened towards 33%.")
        else:
            print("Interpretation: Model was UNDERCONFIDENT. Probabilities have been sharpened towards 0% or 100%.")
            
        return 0

    if args.command == "sync-sportmonks":
        import os
        from datetime import date
        from parlay.data.sportmonks import fetch_fixtures_with_odds, extract_totals_25, fixture_to_match, totals_to_odds
        token = args.token or os.environ.get("SPORTMONKS_API_TOKEN")
        if not token:
            raise SystemExit("Set SPORTMONKS_API_TOKEN or provide --token")
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
        if end < start:
            raise SystemExit("end date must not be before start date")
        dates = tuple(date.fromordinal(day) for day in range(start.toordinal(), end.toordinal() + 1))
        fixtures = fetch_fixtures_with_odds(token, dates)
        database = ResearchDatabase(args.database)
        match_rows, odds_rows = [], []
        for fixture in fixtures:
            totals = extract_totals_25(fixture, bookmaker_id=args.bookmaker_id)
            if totals is None:
                continue
            match_rows.append(fixture_to_match(fixture))
            odds_rows.extend(totals_to_odds(fixture, totals, bookmaker=f"sportmonks:{args.bookmaker_id}"))
        database.insert_future_matches(match_rows)
        database.insert_odds(odds_rows)
        print(f"Synced {len(match_rows)} fixtures and {len(odds_rows)} totals odds snapshots")
        return 0

    if args.command == "predict-sportmonks":
        from parlay.evaluation.market import SELECTIONS
        from parlay.models.poisson import outcome_probabilities, totals_probability
        from parlay.models.team_strength import fit_team_strength
        from parlay.prediction.markets import kelly_criterion
        database = ResearchDatabase(args.database)
        scheduled = [row for row in database.load_matches() if row.match_id.startswith("sportmonks:") and row.home_goals is None and row.away_goals is None]
        history = [row for row in database.load_matches() if row.home_goals is not None and row.away_goals is not None]
        if not scheduled:
            raise SystemExit("No scheduled Sportmonks fixtures found; run sync-sportmonks first")
        if not history:
            raise SystemExit("No completed matches available for training")
        output_rows = []
        for fixture in scheduled:
            training = [row for row in history if row.date < fixture.date]
            if not training or fixture.home_team not in {x for row in training for x in (row.home_team, row.away_team)} or fixture.away_team not in {x for row in training for x in (row.home_team, row.away_team)}:
                output_rows.append({"fixture_id": fixture.match_id, "home_team": fixture.home_team, "away_team": fixture.away_team, "status": "insufficient_history"})
                continue
            model = fit_team_strength(training, model=args.model, estimator=args.estimator, as_of=fixture.date, half_life_days=args.half_life_days)
            totals = totals_probability(model.score_matrix(fixture.home_team, fixture.away_team), 2.5)
            odds = {row.selection: row.odds for row in database.load_odds(match_id=fixture.match_id) if row.market == "totals_2.5"}
            if set(odds) != {"over", "under"}:
                output_rows.append({"fixture_id": fixture.match_id, "kickoff_at": fixture.kickoff_at.isoformat() if fixture.kickoff_at else "", "home_team": fixture.home_team, "away_team": fixture.away_team, "status": "missing_complete_totals_market"})
                continue
            candidates = []
            for selection, probability, key in (("over", totals["over"], "over"), ("under", totals["under"], "under")):
                price = odds.get(key)
                ev = probability * price - 1.0 if price is not None else None
                stake = kelly_criterion(probability, price, args.kelly_fraction, max_stake=args.max_stake) if price is not None and ev >= args.min_ev else 0.0
                if stake > 0: candidates.append((stake, selection, price, ev))
            best = max(candidates, default=(0.0, "", None, None))
            output_rows.append({"fixture_id": fixture.match_id, "kickoff_at": fixture.kickoff_at.isoformat() if fixture.kickoff_at else "", "home_team": fixture.home_team, "away_team": fixture.away_team, "status": "ok", "over_probability": totals["over"], "under_probability": totals["under"], "over_odds": odds.get("over"), "under_odds": odds.get("under"), "recommendation": best[1], "recommendation_odds": best[2], "recommendation_ev": best[3], "stake_fraction": best[0]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import csv
        fields = sorted({key for row in output_rows for key in row})
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(output_rows)
        for row in output_rows:
            print(row)
        print(f"Wrote {len(output_rows)} predictions to {args.output}")
        return 0

    database_path = Path(args.database)
    if database_path.exists():
        database = ResearchDatabase(str(database_path))
        matches = [match for match in database.load_matches() if match.home_goals is not None and match.away_goals is not None]
        odds = database.load_odds()
    else:
        if not args.csv:
            raise SystemExit("No database found; provide at least one CSV input")
        snapshots = [acquire_csv(source, "data/raw/snapshots")[0] for source in args.csv]
        matches, odds = load_many(snapshots, competition=args.competition, season=args.season)
    if args.command == "compare":
        _, summary = compare_models(
            matches, odds=odds, bookmaker=args.bookmaker,
            estimator=args.estimator,
            initial_train_days=args.initial_train_days, test_days=args.test_days,
            step_days=args.step_days, half_life_days=args.half_life_days,
            sot_weight=args.sot_weight,
        )
        write_json(summary, args.output_dir / "model_comparison.json")
        # Print compact table
        print(f"\n{'Model':<20} {'N':>5} {'LogLoss':>8} {'Brier':>8} {'Folds':>5} {'LL_std':>7} {'BR_std':>7}")
        print("-" * 62)
        for row in summary:
            ll_std = row.get("stability_log_loss", {}).get("std", 0.0)
            br_std = row.get("stability_brier", {}).get("std", 0.0)
            print(f"{row['model']:<20} {row['n']:>5.0f} {row['log_loss']:>8.4f} {row['brier_score']:>8.4f} {row['folds']:>5.0f} {ll_std:>7.4f} {br_std:>7.4f}")
        print(f"\nFull results: {args.output_dir / 'model_comparison.json'}")
        return 0

    if args.command == "tune":
        from parlay.evaluation.tuning import tune_half_life
        print(f"Running grid search for half_life_days on {args.model} ({args.estimator})...")
        results = tune_half_life(
            matches, model=args.model, estimator=args.estimator,
            initial_train_days=args.initial_train_days, test_days=args.test_days,
            step_days=args.step_days,
        )
        write_json(results, args.output_dir / f"tune_halflife_{args.model}.json")
        
        print(f"\n{'Half Life':<12} {'N':>5} {'LogLoss':>8} {'Brier':>8} {'Folds':>5}")
        print("-" * 45)
        for row in results:
            hl_str = str(row['half_life_days']) if row['half_life_days'] is not None else "None"
            print(f"{hl_str:<12} {row['n']:>5.0f} {row['log_loss']:>8.4f} {row['brier_score']:>8.4f} {row['folds']:>5.0f}")
        print(f"\nTuning results written to {args.output_dir / f'tune_halflife_{args.model}.json'}")
        return 0

    if args.command == "predict":
        import csv
        import os
        from datetime import date, datetime, timezone
        from parlay.data.normalization import canonical_team_name
        from parlay.models.team_strength import fit_team_strength
        from parlay.models.corners import fit_corner_strength, corner_totals_probabilities
        from parlay.models.poisson import outcome_probabilities, totals_probability
        from parlay.prediction.markets import correct_score_probabilities, grouped_score_markets, kelly_criterion, totals_settlement_probabilities

        fixtures = []
        with args.fixtures.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fixtures.append(row)

            if not any(row.get("B365>2.5") or row.get("B365<2.5") for row in fixtures):
                token = os.environ.get("SPORTMONKS_API_TOKEN")
                if token:
                    from parlay.data.sportmonks import extract_totals_25, fetch_fixtures_with_odds
                    from datetime import date as date_type
                    requested_dates = sorted({date_type.fromisoformat(row["Date"]) if "-" in row["Date"] else datetime.strptime(row["Date"], "%d/%m/%Y").date() for row in fixtures})
                    api_fixtures = fetch_fixtures_with_odds(token, tuple(requested_dates))
                    api_by_name = {f["name"].casefold(): f for f in api_fixtures}
                    for row in fixtures:
                        match = api_by_name.get(f"{row['HomeTeam']} vs {row['AwayTeam']}".casefold())
                        if match:
                            totals = extract_totals_25(match, bookmaker_id=2)
                            if totals:
                                row["B365>2.5"] = str(totals["over_odds"])
                                row["B365<2.5"] = str(totals["under_odds"])
                
        if not fixtures:
            raise SystemExit("Fixtures CSV is empty")
        required_fixture_columns = {"Date", "HomeTeam", "AwayTeam"}
        missing = required_fixture_columns - set(fixtures[0])
        if missing:
            raise SystemExit(f"Fixtures CSV is missing columns: {sorted(missing)}")

        output_rows = []
        print(f"\nPredictions for {len(fixtures)} upcoming fixtures:")
        print(f"{'Date':<10} | {'Home':<15} vs {'Away':<15} | {'1':>5} {'X':>5} {'2':>5} | {'O2.5':>5} {'U2.5':>5} | {'Recommendation':<42}")
        print("-" * 122)
        
        for row in fixtures:
            home_raw = row["HomeTeam"].strip()
            away_raw = row["AwayTeam"].strip()
            home = canonical_team_name(home_raw)
            away = canonical_team_name(away_raw)
            date_str = row.get("Date", "Unknown")
            fixture_date = None
            for date_format in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
                try:
                    fixture_date = datetime.strptime(date_str.strip(), date_format).date()
                    break
                except ValueError:
                    continue
            if fixture_date is None:
                raise SystemExit(f"Invalid fixture date {date_str!r}; expected DD/MM/YYYY, DD/MM/YY, or YYYY-MM-DD")

            available_matches = [match for match in matches if match.date < fixture_date]
            if not available_matches:
                output_rows.append({"date": date_str, "home_team": home, "away_team": away, "home_raw": home_raw, "away_raw": away_raw, "status": "no_pre_match_history"})
                print(f"{date_str:<10} | {home_raw:<15} vs {away_raw:<15} | {'N/A':>5} {'N/A':>5} {'N/A':>5} | {'N/A':>5} {'N/A':>5} | No pre-match history")
                continue
            fitted = fit_team_strength(
                available_matches, model=args.model, estimator=args.estimator,
                as_of=fixture_date, half_life_days=args.half_life_days,
                sot_weight=args.sot_weight,
            )
            
            if home not in fitted.teams or away not in fitted.teams:
                output_rows.append({"date": date_str, "home_team": home, "away_team": away, "home_raw": home_raw, "away_raw": away_raw, "status": "unknown_team"})
                print(f"{date_str:<10} | {home_raw:<15} vs {away_raw:<15} | {'N/A':>5} {'N/A':>5} {'N/A':>5} | {'N/A':>5} {'N/A':>5} | New team, no history")
                continue
                
            matrix = fitted.score_matrix(home, away)
            probs = outcome_probabilities(matrix)
            totals = totals_probability(matrix, 2.5)
            cs = correct_score_probabilities(matrix, max_home=4, max_away=4)
            grouped = grouped_score_markets(matrix)

            # Fit corner model for current context
            corner_model = fit_corner_strength(available_matches, half_life_days=args.half_life_days or 365.0)
            exp_h_corn, exp_a_corn = corner_model.expected_corners(home, away)
            corn_matrix = corner_model.corner_matrix(home, away)
            corn_105 = corner_totals_probabilities(corn_matrix, 10.5)

            p_1 = probs["home_win"]
            p_x = probs["draw"]
            p_2 = probs["away_win"]
            p_o = totals["over"]
            p_u = totals["under"]

            # Sort top 3 most likely correct scorelines
            top_scores = sorted([(k, v) for k, v in cs.items() if k != "other"], key=lambda x: x[1], reverse=True)[:3]
            top_scores_str = ", ".join(f"{k}({v*100:.1f}%)" for k, v in top_scores)
            
            bet_str = "No value found"
            best_kelly = 0.0
            best_selection = ""
            best_odds = None
            best_ev = None
            
            # Evaluate 1X2 market
            if "B365H" in row and "B365D" in row and "B365A" in row:
                try:
                    edges = [
                        ("Home", p_1, float(row["B365H"])),
                        ("Draw", p_x, float(row["B365D"])),
                        ("Away", p_2, float(row["B365A"]))
                    ]
                    for sel, p, o in edges:
                        ev = p * o - 1.0
                        k = kelly_criterion(p, o, fraction=args.kelly_fraction, max_stake=args.max_stake) if ev >= args.min_ev else 0.0
                        if k > best_kelly:
                            best_kelly = k
                            best_selection, best_odds, best_ev = sel, o, ev
                            bet_str = f"Bet {sel} @ {o:.2f}, EV {ev*100:+.1f}% (Stake: {k*100:.1f}%)"
                except ValueError:
                    pass
            
            # Evaluate Over/Under 2.5 market
            if "B365>2.5" in row and "B365<2.5" in row:
                try:
                    edges_ou = [
                        ("O2.5", p_o, float(row["B365>2.5"])),
                        ("U2.5", p_u, float(row["B365<2.5"]))
                    ]
                    for sel, p, o in edges_ou:
                        ev = p * o - 1.0
                        k = kelly_criterion(p, o, fraction=args.kelly_fraction, max_stake=args.max_stake) if ev >= args.min_ev else 0.0
                        if k > best_kelly:
                            best_kelly = k
                            best_selection, best_odds, best_ev = sel, o, ev
                            bet_str = f"Bet {sel} @ {o:.2f}, EV {ev*100:+.1f}% (Stake: {k*100:.1f}%)"
                except ValueError:
                    pass
            
            # Compute multi-line totals for detailed dashboard & output
            from parlay.models.team_strength import fit_team_strength as _fts  # noqa: ensure import available
            exp_hg, exp_ag = fitted.expected_goals(home, away)
            multi_lines = [1.5, 2.5, 2.75, 3.5]
            multi_totals = {line: totals_settlement_probabilities(matrix, line) for line in multi_lines}
            corn_lines = [9.5, 10.5, 11.5]
            corn_probs_map = {cl: corner_totals_probabilities(corn_matrix, cl) for cl in corn_lines}
            corner_1x2 = __import__('parlay.models.corners', fromlist=['corner_match_betting']).corner_match_betting(corn_matrix)
            cs_top5 = sorted([(k,v) for k,v in cs.items() if k!='other'], key=lambda x: x[1], reverse=True)[:5]

            # Evaluate additional markets if odds provided
            # BTTS
            if "B365BTTS_Y" in row and "B365BTTS_N" in row:
                try:
                    for sel,p,o_col in [("BTTS_Y", grouped["btts_yes"], "B365BTTS_Y"), ("BTTS_N", grouped["btts_no"], "B365BTTS_N")]:
                        o=float(row[o_col]); ev=p*o-1; k=kelly_criterion(p,o,fraction=args.kelly_fraction,max_stake=args.max_stake) if ev>=args.min_ev else 0.0
                        if k>best_kelly: best_kelly=k; best_selection, best_odds, best_ev = sel,o,ev; bet_str=f"Bet {sel} @ {o:.2f}, EV {ev*100:+.1f}% (Stake: {k*100:.1f}%)"
                except: pass
            # Corners O/U
            for cl in corn_lines:
                col_o=f"B365C>{cl}"; col_u=f"B365C<{cl}"
                if col_o in row and col_u in row and row[col_o] and row[col_u]:
                    try:
                        p_o_c=corn_probs_map[cl]["over"]; p_u_c=corn_probs_map[cl]["under"]
                        for sel,p,o_col in [(f"Corners O{cl}",p_o_c,col_o),(f"Corners U{cl}",p_u_c,col_u)]:
                            o=float(row[o_col]); ev=p*o-1; k=kelly_criterion(p,o,fraction=args.kelly_fraction,max_stake=args.max_stake) if ev>=args.min_ev else 0.0
                            if k>best_kelly: best_kelly=k; best_selection, best_odds, best_ev = sel,o,ev; bet_str=f"Bet {sel} @ {o:.2f}, EV {ev*100:+.1f}% (Stake: {k*100:.1f}%)"
                    except: pass

            # Anomaly diagnostics (§17) — treat extreme disagreement as anomaly, not auto edge
            from parlay.evaluation.anomaly import diagnose
            # n_historical per team
            n_home_hist = sum(1 for m in available_matches if m.home_team==home or m.away_team==home)
            n_away_hist = sum(1 for m in available_matches if m.home_team==away or m.away_team==away)
            n_hist = min(n_home_hist, n_away_hist)
            is_promoted = n_hist < 10
            # Use market home prob if available
            mkt_home_p = None
            try:
                if "B365H" in row and row["B365H"]:
                    # raw implied without de-vig as proxy; for true use implied_probabilities
                    mkt_home_p = 1.0/float(row["B365H"])
            except: pass
            diag = diagnose(p_1, mkt_home_p, n_historical=n_hist, is_promoted=is_promoted)
            anomaly_note = f" | {diag.decision} {','.join(diag.anomaly_flags) if diag.anomaly_flags else 'ok'}" if diag.decision != "PASS" or diag.anomaly_flags else ""
            print(f"{date_str:<10} | {home_raw:<15} vs {away_raw:<15} | {p_1*100:>4.1f}% {p_x*100:>4.1f}% {p_2*100:>4.1f}% | {p_o*100:>4.1f}% {p_u*100:>4.1f}% | {bet_str}{anomaly_note}")
            # Build comprehensive output row with fair odds for manual comparison
            def _fair(p): return round(1/p,2) if p>1e-9 else 999
            output_rows.append({
                "date": date_str, "home_team": home, "away_team": away, "home_raw": home_raw, "away_raw": away_raw, "status": "ok",
                # 1X2
                "home_win_prob": round(p_1,4), "draw_prob": round(p_x,4), "away_win_prob": round(p_2,4),
                "fair_home": _fair(p_1), "fair_draw": _fair(p_x), "fair_away": _fair(p_2),
                "double_chance_1x": round(grouped["double_chance_1x"],4), "double_chance_x2": round(grouped["double_chance_x2"],4), "double_chance_12": round(grouped["double_chance_12"],4),
                "fair_1x": _fair(grouped["double_chance_1x"]), "fair_x2": _fair(grouped["double_chance_x2"]), "fair_12": _fair(grouped["double_chance_12"]),
                # Totals goals
                **{f"over_{str(line).replace('.','_')}_prob": round(multi_totals[line]["over"],4) for line in multi_lines},
                **{f"under_{str(line).replace('.','_')}_prob": round(multi_totals[line]["under"],4) for line in multi_lines},
                **{f"fair_over_{str(line).replace('.','_')}": _fair(multi_totals[line]["over"]) for line in multi_lines},
                **{f"fair_under_{str(line).replace('.','_')}": _fair(multi_totals[line]["under"]) for line in multi_lines},
                # BTTS & Win to Nil
                "btts_yes_prob": round(grouped["btts_yes"],4), "btts_no_prob": round(grouped["btts_no"],4), "fair_btts_yes": _fair(grouped["btts_yes"]), "fair_btts_no": _fair(grouped["btts_no"]),
                "home_win_to_nil_prob": round(grouped["home_win_to_nil"],4), "away_win_to_nil_prob": round(grouped["away_win_to_nil"],4),
                "fair_home_win_to_nil": _fair(grouped["home_win_to_nil"]), "fair_away_win_to_nil": _fair(grouped["away_win_to_nil"]),
                "score_draw_prob": round(grouped["score_draw"],4), "scoreless_draw_prob": round(grouped["scoreless_draw"],4),
                # Correct Score
                **{f"cs_{k.replace('-','_')}_prob": round(v,4) for k,v in cs_top5},
                **{f"fair_cs_{k.replace('-','_')}": _fair(v) for k,v in cs_top5},
                "cs_other_prob": round(cs.get("other",0),4),
                "top_scorelines": top_scores_str,
                # Goals expectation
                "exp_home_goals": round(exp_hg,2), "exp_away_goals": round(exp_ag,2), "exp_total_goals": round(exp_hg+exp_ag,2),
                # Corners
                "exp_home_corners": round(exp_h_corn,1), "exp_away_corners": round(exp_a_corn,1), "exp_total_corners": round(exp_h_corn+exp_a_corn,1),
                "corners_home_most_prob": round(corner_1x2["home_most"],4), "corners_tie_prob": round(corner_1x2["tie"],4), "corners_away_most_prob": round(corner_1x2["away_most"],4),
                **{f"corners_over_{str(cl).replace('.','_')}_prob": round(corn_probs_map[cl]["over"],4) for cl in corn_lines},
                **{f"corners_under_{str(cl).replace('.','_')}_prob": round(corn_probs_map[cl]["under"],4) for cl in corn_lines},
                **{f"fair_corners_over_{str(cl).replace('.','_')}": _fair(corn_probs_map[cl]["over"]) for cl in corn_lines},
                # Market odds passthrough for audit
                "B365H": row.get("B365H"), "B365D": row.get("B365D"), "B365A": row.get("B365A"),
                "B365>2.5": row.get("B365>2.5"), "B365<2.5": row.get("B365<2.5"),
                "recommendation": best_selection, "recommendation_odds": best_odds, "recommendation_ev": round(best_ev,4) if best_ev is not None else None, "stake_fraction": round(best_kelly,4),
                "anomaly_decision": diag.decision, "anomaly_flags": ",".join(diag.anomaly_flags), "n_historical": n_hist, "is_promoted": is_promoted,
                "forecast_policy": "kickoff_minus_60m", "model_update_frequency": "per_fold"
            })

            # Detailed per-match dashboard for manual use
            print(f"  └─ xG: {home} {exp_hg:.2f} - {exp_ag:.2f} {away}  |  BTTS Yes {grouped['btts_yes']*100:.1f}% (fair {1/grouped['btts_yes']:.2f})  |  Home Win to Nil {grouped['home_win_to_nil']*100:.1f}%  |  Away Win to Nil {grouped['away_win_to_nil']*100:.1f}%")
            for line in multi_lines:
                p = multi_totals[line]
                fair_over = 1/p['over'] if p['over']>1e-6 else 999
                fair_under = 1/p['under'] if p['under']>1e-6 else 999
                push_txt = f" push {p['push']*100:.1f}%" if p['push']>0.002 else ""
                # Mark market odds if available for 2.5
                market_note = ""
                if line == 2.5 and "B365>2.5" in row:
                    try:
                        m_over = float(row["B365>2.5"]); m_under = float(row["B365<2.5"])
                        ev_o = p['over']*m_over-1; ev_u = p['under']*m_under-1
                        market_note = f"  [Mkt O@{m_over:.2f} EV {ev_o*100:+.1f}% | U@{m_under:.2f} EV {ev_u*100:+.1f}%]"
                    except: pass
                print(f"     O/U {line:<4} → Over {p['over']*100:5.1f}% (fair {fair_over:5.2f}) | Under {p['under']*100:5.1f}% (fair {fair_under:5.2f}){push_txt}{market_note}")
            cs_line = "  |  ".join(f"{k} {v*100:.1f}% (fair {1/v:.2f})" for k,v in cs_top5)
            print(f"     Correct Score Top 5: {cs_line}  |  other {cs.get('other',0)*100:.1f}%")
            print(f"     xCorners: {home} {exp_h_corn:.1f} - {exp_a_corn:.1f} {away} (total {exp_h_corn+exp_a_corn:.1f}) | Corners 1X2: Home {corner_1x2['home_most']*100:.1f}% / Tie {corner_1x2['tie']*100:.1f}% / Away {corner_1x2['away_most']*100:.1f}%")
            for cl in corn_lines:
                p = corn_probs_map[cl]
                print(f"       Corners O/U {cl:<4} → Over {p['over']*100:5.1f}% (fair {1/p['over']:.2f}) | Under {p['under']*100:5.1f}%")
            print()

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for item in output_rows for key in item}))
                writer.writeheader()
                writer.writerows(output_rows)
            
        return 0

    from parlay.evaluation.backtest import run_backtest_full
    full_result = run_backtest_full(
        matches, model=args.model, estimator=args.estimator,
        initial_train_days=args.initial_train_days,
        test_days=args.test_days, step_days=args.step_days,
        half_life_days=args.half_life_days, sot_weight=args.sot_weight, odds=odds, bookmaker=args.bookmaker,
        forecast_lead_minutes=args.forecast_lead_minutes,
        strategy_min_edge=args.min_edge, strategy_min_ev=args.min_ev,
    )
    records, metrics = full_result.records, full_result.metrics
    strategy = evaluate_flat_stake(records, min_edge=args.min_edge, min_ev=args.min_ev)
    output = args.output_dir
    write_predictions(records, output / "predictions.csv")
    write_json({
        "metrics": metrics,
        "strategy": strategy,
        "fold_metrics": [asdict(row) for row in full_result.fold_metrics],
    }, output / "metrics.json")
    write_json(calibration_bins(records), output / "calibration.json")
    print(f"Backtested {len(records)} matches; log_loss={metrics['log_loss']:.4f}; brier={metrics['brier_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
