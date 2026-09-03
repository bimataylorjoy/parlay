"""Expanding-window backtest runner for the local model baseline."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json

from parlay.data.schemas import Match
from parlay.evaluation.metrics import aggregate_scores, brier_score, log_loss, outcome_label
from parlay.evaluation.temporal import TemporalFold, expanding_window
from parlay.models.team_strength import fit_team_strength
from parlay.models.poisson import outcome_probabilities
from parlay.data.schemas import OddsSnapshot
from parlay.prediction.markets import implied_probabilities, latest_odds_as_of, model_edge


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    match_id: str
    forecast_timestamp: str
    model_version: str
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    actual: str
    home_goals: int
    away_goals: int
    log_loss: float
    brier_score: float
    market_home: float | None = None
    market_draw: float | None = None
    market_away: float | None = None
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    edge_home: float | None = None
    edge_draw: float | None = None
    edge_away: float | None = None
    ev_home: float | None = None
    ev_draw: float | None = None
    ev_away: float | None = None
    pin_home: float | None = None
    pin_draw: float | None = None
    pin_away: float | None = None


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Per-fold scoring breakdown."""
    fold_index: int
    train_end: str
    test_start: str
    test_end: str
    n: int
    log_loss: float
    brier_score: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Full backtest output with per-fold visibility."""
    records: list[PredictionRecord]
    metrics: dict[str, float]
    fold_metrics: list[FoldMetrics]
    metadata: dict[str, object] | None = None


def _forecast_market_odds(
    odds: list[OddsSnapshot], match: Match, forecast_timestamp, bookmaker: str | None,
) -> tuple[dict[str, float] | None, dict[str, float]]:
    market = latest_odds_as_of(
        [row for row in odds if row.match_id == match.match_id],
        forecast_timestamp,
        bookmaker=bookmaker,
    )
    if not market:
        return None, {}
    try:
        return implied_probabilities(market, bookmaker=bookmaker), {
            row.selection: row.odds for row in market
        }
    except ValueError:
        return None, {}


def run_backtest(
    matches: list[Match],
    *,
    model: str = "poisson",
    estimator: str = "mle",
    model_version: str | None = None,
    initial_train_days: int = 730,
    test_days: int = 30,
    step_days: int | None = None,
    half_life_days: float | None = 730.0,
    max_goals: int = 10,
    sot_weight: float = 0.0,
    odds: list[OddsSnapshot] | None = None,
    bookmaker: str | None = None,
    forecast_lead_minutes: int = 60,
    strategy_min_edge: float = 0.03,
    strategy_min_ev: float = 0.02,
    evaluation_mode: str = "rolling_origin",
    calibration_mode: str | None = None,
) -> tuple[list[PredictionRecord], dict[str, float]]:
    result = run_backtest_full(
        matches, model=model, estimator=estimator, model_version=model_version,
        initial_train_days=initial_train_days, test_days=test_days,
        step_days=step_days, half_life_days=half_life_days,
        max_goals=max_goals, sot_weight=sot_weight, odds=odds, bookmaker=bookmaker,
        forecast_lead_minutes=forecast_lead_minutes,
        strategy_min_edge=strategy_min_edge, strategy_min_ev=strategy_min_ev,
        evaluation_mode=evaluation_mode, calibration_mode=calibration_mode,
    )
    return result.records, result.metrics


def run_backtest_full(
    matches: list[Match],
    *,
    model: str = "poisson",
    estimator: str = "mle",
    model_version: str | None = None,
    initial_train_days: int = 730,
    test_days: int = 30,
    step_days: int | None = None,
    half_life_days: float | None = 730.0,
    max_goals: int = 10,
    sot_weight: float = 0.0,
    odds: list[OddsSnapshot] | None = None,
    bookmaker: str | None = None,
    forecast_lead_minutes: int = 60,
    strategy_min_edge: float = 0.03,
    strategy_min_ev: float = 0.02,
    evaluation_mode: str = "rolling_origin",
    calibration_mode: str | None = None,
    calibration_records: list[PredictionRecord] | None = None,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
) -> BacktestResult:
    if forecast_lead_minutes < 0:
        raise ValueError("forecast_lead_minutes must be non-negative")
    if evaluation_mode not in ("rolling_origin", "fixed_origin"):
        raise ValueError("evaluation_mode must be rolling_origin or fixed_origin")
    folds = expanding_window(
        matches,
        initial_train_days=initial_train_days,
        test_days=test_days,
        step_days=step_days,
        mode=evaluation_mode,
        forecast_lead_minutes=forecast_lead_minutes,
        start_date=datetime.fromisoformat(evaluation_start).date() if evaluation_start else None,
        end_date=datetime.fromisoformat(evaluation_end).date() if evaluation_end else None,
    )
    records: list[PredictionRecord] = []
    fold_metrics_list: list[FoldMetrics] = []
    version = model_version or f"{model}_{estimator}_v1"
    for fold_index, fold in enumerate(folds):
        prior_records = list(records)
        fold_records: list[PredictionRecord] = []
        fitted = fit_team_strength(
            list(fold.train), model=model, estimator=estimator, as_of=fold.train_end,
            half_life_days=half_life_days, max_goals=max_goals, sot_weight=sot_weight,
        )
        for match in fold.test:
            if match.home_team not in fitted.teams or match.away_team not in fitted.teams:
                continue
            probabilities = outcome_probabilities(
                fitted.score_matrix(match.home_team, match.away_team, match.neutral)
            )
            actual = outcome_label(match.home_goals, match.away_goals)
            forecast_timestamp = (
                match.kickoff_at - timedelta(minutes=forecast_lead_minutes)
                if match.kickoff_at is not None
                else datetime.combine(match.date, datetime.min.time(), timezone.utc)
            )
            market_probabilities, odds_by_selection = _forecast_market_odds(
                odds or [], match, forecast_timestamp, bookmaker,
            )
            
            pin_market = [row for row in (odds or []) if row.match_id == match.match_id and row.bookmaker == "Pinnacle" and row.is_closing]
            # fallback to non-closing pinnacle if closing not found
            if not pin_market:
                pin_market = latest_odds_as_of([row for row in (odds or []) if row.match_id == match.match_id], forecast_timestamp, bookmaker="Pinnacle")
                
            pin_probabilities = None
            if pin_market:
                try:
                    pin_probabilities = implied_probabilities(pin_market, bookmaker="Pinnacle")
                except ValueError:
                    pass
                    
            model_by_selection = {
                "home": probabilities["home_win"], "draw": probabilities["draw"], "away": probabilities["away_win"],
            }
            record = PredictionRecord(
                match_id=match.match_id,
                forecast_timestamp=forecast_timestamp.isoformat(),
                model_version=version,
                home_team=match.home_team,
                away_team=match.away_team,
                home_win=probabilities["home_win"],
                draw=probabilities["draw"],
                away_win=probabilities["away_win"],
                actual=actual,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                log_loss=log_loss(probabilities, actual),
                brier_score=brier_score(probabilities, actual),
                market_home=market_probabilities.get("home_win") if market_probabilities else None,
                market_draw=market_probabilities.get("draw") if market_probabilities else None,
                market_away=market_probabilities.get("away_win") if market_probabilities else None,
                odds_home=odds_by_selection.get("home"), odds_draw=odds_by_selection.get("draw"),
                odds_away=odds_by_selection.get("away"),
                edge_home=(probabilities["home_win"] - market_probabilities["home_win"]) if market_probabilities else None,
                edge_draw=(probabilities["draw"] - market_probabilities["draw"]) if market_probabilities else None,
                edge_away=(probabilities["away_win"] - market_probabilities["away_win"]) if market_probabilities else None,
                ev_home=model_edge(model_by_selection["home"], odds_by_selection["home"]) if "home" in odds_by_selection else None,
                ev_draw=model_edge(model_by_selection["draw"], odds_by_selection["draw"]) if "draw" in odds_by_selection else None,
                ev_away=model_edge(model_by_selection["away"], odds_by_selection["away"]) if "away" in odds_by_selection else None,
                pin_home=pin_probabilities.get("home_win") if pin_probabilities else None,
                pin_draw=pin_probabilities.get("draw") if pin_probabilities else None,
                pin_away=pin_probabilities.get("away_win") if pin_probabilities else None,
            )
            fold_records.append(record)
        records.extend(fold_records)
        if fold_records:
            available_calibration = calibration_records if calibration_records is not None else prior_records
            if calibration_mode == "temperature" and available_calibration:
                from parlay.evaluation.calibration import find_optimal_temperature, apply_calibration
                eligible = [r for r in available_calibration if r.forecast_timestamp < fold_records[0].forecast_timestamp]
                if eligible:
                    temperature = find_optimal_temperature(eligible)
                    calibrated = apply_calibration(fold_records, temperature)
                    fold_records = [PredictionRecord(**row) for row in calibrated]
                    records[-len(fold_records):] = fold_records
            fold_agg = aggregate_scores(asdict(r) for r in fold_records)
            fold_metrics_list.append(FoldMetrics(
                fold_index=fold_index,
                train_end=fold.train_end.isoformat(),
                test_start=fold.test_start.isoformat(),
                test_end=fold.test_end.isoformat(),
                n=len(fold_records),
                log_loss=fold_agg["log_loss"],
                brier_score=fold_agg["brier_score"],
            ))
    metrics = aggregate_scores(asdict(record) for record in records)
    metrics["folds"] = float(len(folds))
    metadata = {
        "evaluation_mode": evaluation_mode,
        "forecast_timestamp_policy": f"kickoff_minus_{forecast_lead_minutes}m",
        "model_update_frequency": "per_fold" if evaluation_mode == "rolling_origin" else "fixed_origin",
        "calibration_mode": calibration_mode,
        "calibration_window": "prior_predictions_only" if calibration_mode == "temperature" else None,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
    }
    # Regime analysis (§18) — sample size + log loss/brier per regime
    try:
        from parlay.evaluation.regime import analyze_by_regime, promoted_regime
        regime_out = analyze_by_regime(records, lambda r: promoted_regime(r), min_n=30)
        metadata["regime_analysis"] = regime_out
    except Exception:
        pass
    return BacktestResult(records=records, metrics=metrics, fold_metrics=fold_metrics_list, metadata=metadata)
