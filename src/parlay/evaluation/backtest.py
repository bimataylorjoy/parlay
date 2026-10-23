"""Walk-forward backtest of goal models against realized results and market prices.

For every fold the goal model is refitted on information available before
the fold, then each test match is forecast at ``kickoff - forecast_lead``.
Market prices are taken from the latest *pre-forecast* snapshot of the
chosen bookmaker; closing prices (e.g. Pinnacle) are kept only as an
evaluation benchmark and for closing-line value, never as model input.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from parlay.data.schemas import Match, OddsSnapshot
from parlay.evaluation.metrics import (
    aggregate_scores,
    binary_log_loss,
    brier_score,
    log_loss,
    outcome_label,
    ranked_probability_score,
)
from parlay.evaluation.regime import analyze_by_regime, competition_regime, team_history_regime
from parlay.evaluation.temporal import EVALUATION_MODES, expanding_window
from parlay.models.team_strength import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_L2_REG,
    DEFAULT_SOT_BLEND,
    fit_team_strength,
)
from parlay.prediction.markets import (
    de_vig,
    expected_value,
    implied_probabilities,
    latest_odds_as_of,
    outcome_probabilities,
    totals_probabilities,
)

SELECTIONS = ("home", "draw", "away")
OUTCOME_BY_SELECTION = {"home": "home_win", "draw": "draw", "away": "away_win"}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    model: str = "dixon_coles"
    estimator: str = "mle"
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS
    l2_reg: float = DEFAULT_L2_REG
    sot_blend: float = DEFAULT_SOT_BLEND
    max_goals: int | None = None
    initial_train_days: int = 730
    test_days: int = 7
    step_days: int | None = None
    evaluation_mode: str = "rolling_origin"
    forecast_lead_minutes: int = 60
    competitions: tuple[str, ...] | None = None
    bookmaker: str | None = "Bet365"
    closing_bookmaker: str = "Pinnacle"
    calibration_mode: str | None = None
    evaluation_start: date | None = None
    evaluation_end: date | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        if self.forecast_lead_minutes < 0:
            raise ValueError("forecast_lead_minutes must be non-negative")
        if self.evaluation_mode not in EVALUATION_MODES:
            raise ValueError(f"evaluation_mode must be one of {EVALUATION_MODES}")
        if self.calibration_mode not in (None, "temperature"):
            raise ValueError("calibration_mode must be None or 'temperature'")

    @property
    def version(self) -> str:
        return (
            self.model_version
            or f"{self.model}_{self.estimator}_hl{self.half_life_days}_l2{self.l2_reg}_sot{self.sot_blend}"
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    match_id: str
    competition: str
    match_date: str
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
    rps: float
    expected_home_goals: float
    expected_away_goals: float
    over_2_5: float
    btts_yes: float
    min_team_matches: int
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
    closing_home: float | None = None
    closing_draw: float | None = None
    closing_away: float | None = None
    market_over_2_5: float | None = None
    odds_over_2_5: float | None = None
    odds_under_2_5: float | None = None

    def probabilities(self) -> dict[str, float]:
        return {"home_win": self.home_win, "draw": self.draw, "away_win": self.away_win}

    def market_probabilities(self) -> dict[str, float] | None:
        if self.market_home is None or self.market_draw is None or self.market_away is None:
            return None
        return {"home_win": self.market_home, "draw": self.market_draw, "away_win": self.market_away}

    def closing_probabilities(self) -> dict[str, float] | None:
        if self.closing_home is None or self.closing_draw is None or self.closing_away is None:
            return None
        return {"home_win": self.closing_home, "draw": self.closing_draw, "away_win": self.closing_away}


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_index: int
    train_end: str
    test_start: str
    test_end: str
    n: int
    log_loss: float
    brier_score: float
    rps: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    records: list[PredictionRecord]
    metrics: dict[str, float]
    fold_metrics: list[FoldMetrics]
    metadata: dict[str, object] = field(default_factory=dict)


def forecast_timestamp(match: Match, lead_minutes: int) -> datetime:
    """``kickoff - lead``; midnight UTC of the match day when kickoff is unknown."""
    if match.kickoff_at is None:
        return datetime.combine(match.date, time(0, 0), tzinfo=timezone.utc)
    kickoff = match.kickoff_at if match.kickoff_at.tzinfo else match.kickoff_at.replace(tzinfo=timezone.utc)
    return kickoff - timedelta(minutes=lead_minutes)


def _index_odds(odds: Iterable[OddsSnapshot]) -> dict[str, list[OddsSnapshot]]:
    index: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for row in odds:
        index[row.match_id].append(row)
    return index


def _pre_match_1x2(rows: list[OddsSnapshot], as_of: datetime, bookmaker: str | None):
    market = latest_odds_as_of(rows, as_of, bookmaker=bookmaker)
    if not market:
        return None, {}
    try:
        return implied_probabilities(market, bookmaker=bookmaker), {row.selection: row.odds for row in market}
    except ValueError:
        return None, {}


def _closing_1x2(rows: list[OddsSnapshot], bookmaker: str) -> dict[str, float] | None:
    closing = [row for row in rows if row.is_closing and row.market == "1x2" and row.bookmaker == bookmaker]
    if not closing:
        return None
    latest = max(row.captured_at for row in closing)
    try:
        return implied_probabilities(
            [row for row in closing if row.captured_at == latest], bookmaker=bookmaker
        )
    except ValueError:
        return None


def _pre_match_totals(rows: list[OddsSnapshot], as_of: datetime, bookmaker: str | None):
    market = latest_odds_as_of(rows, as_of, bookmaker=bookmaker, market="totals_2.5")
    prices = {row.selection: row.odds for row in market}
    if set(prices) != {"over", "under"}:
        return None, None, None
    fair = de_vig({"over": 1.0 / prices["over"], "under": 1.0 / prices["under"]}, method="shin")
    return fair["over"], prices["over"], prices["under"]


def _outcome_value(source: dict[str, float] | None, selection: str) -> float | None:
    return source[OUTCOME_BY_SELECTION[selection]] if source else None


def _predict_fold(fold, fitted, config: BacktestConfig, odds_index) -> list[PredictionRecord]:
    records = []
    for match in fold.test:
        if config.competitions and match.competition not in config.competitions:
            continue
        if match.home_team not in fitted or match.away_team not in fitted:
            continue
        matrix = fitted.score_matrix(match.home_team, match.away_team, match.neutral)
        probabilities = outcome_probabilities(matrix)
        exp_home, exp_away = fitted.expected_goals(match.home_team, match.away_team, match.neutral)
        actual = outcome_label(match.home_goals, match.away_goals)
        forecast = forecast_timestamp(match, config.forecast_lead_minutes)
        odds_rows = odds_index.get(match.match_id, [])
        market, prices = _pre_match_1x2(odds_rows, forecast, config.bookmaker)
        closing = _closing_1x2(odds_rows, config.closing_bookmaker)
        market_over, odds_over, odds_under = _pre_match_totals(odds_rows, forecast, config.bookmaker)
        totals = totals_probabilities(matrix, 2.5)
        btts = float(matrix[1:, 1:].sum())
        model_by_selection = {sel: probabilities[OUTCOME_BY_SELECTION[sel]] for sel in SELECTIONS}

        edges = {
            sel: model_by_selection[sel] - market[OUTCOME_BY_SELECTION[sel]] if market else None
            for sel in SELECTIONS
        }
        evs = {
            sel: expected_value(model_by_selection[sel], prices[sel]) if sel in prices else None
            for sel in SELECTIONS
        }

        records.append(
            PredictionRecord(
                match_id=match.match_id,
                competition=match.competition,
                match_date=match.date.isoformat(),
                forecast_timestamp=forecast.isoformat(),
                model_version=config.version,
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
                rps=ranked_probability_score(probabilities, actual),
                expected_home_goals=exp_home,
                expected_away_goals=exp_away,
                over_2_5=totals["over"],
                btts_yes=btts,
                min_team_matches=min(
                    fitted.historical_matches.get(match.home_team, 0),
                    fitted.historical_matches.get(match.away_team, 0),
                ),
                market_home=_outcome_value(market, "home"),
                market_draw=_outcome_value(market, "draw"),
                market_away=_outcome_value(market, "away"),
                odds_home=prices.get("home"),
                odds_draw=prices.get("draw"),
                odds_away=prices.get("away"),
                edge_home=edges["home"],
                edge_draw=edges["draw"],
                edge_away=edges["away"],
                ev_home=evs["home"],
                ev_draw=evs["draw"],
                ev_away=evs["away"],
                closing_home=_outcome_value(closing, "home"),
                closing_draw=_outcome_value(closing, "draw"),
                closing_away=_outcome_value(closing, "away"),
                market_over_2_5=market_over,
                odds_over_2_5=odds_over,
                odds_under_2_5=odds_under,
            )
        )
    return records


def _calibrate(fold_records, history, config: BacktestConfig):
    if config.calibration_mode != "temperature" or not history:
        return fold_records
    from parlay.evaluation.calibration import apply_calibration, find_optimal_temperature

    cutoff = min(record.forecast_timestamp for record in fold_records)
    eligible = [record for record in history if record.forecast_timestamp < cutoff]
    if not eligible:
        return fold_records
    temperature = find_optimal_temperature(eligible)
    return [PredictionRecord(**row) for row in apply_calibration(fold_records, temperature)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_records(records: list[PredictionRecord]) -> dict[str, float]:
    """1X2 scores, market benchmarks on identical samples, and secondary markets."""
    if not records:
        return {"n": 0.0}
    metrics = aggregate_scores(asdict(record) for record in records)

    def over(r: PredictionRecord) -> bool:
        return r.home_goals + r.away_goals > 2.5

    priced = [(r, r.market_probabilities()) for r in records]
    priced = [(r, p) for r, p in priced if p is not None]
    if priced:
        metrics["market_n"] = float(len(priced))
        metrics["model_log_loss_on_market_sample"] = _mean([r.log_loss for r, _ in priced])
        metrics["market_log_loss"] = _mean([log_loss(p, r.actual) for r, p in priced])  # type: ignore[arg-type]
    closed = [(r, r.closing_probabilities()) for r in records]
    closed = [(r, p) for r, p in closed if p is not None]
    if closed:
        metrics["closing_n"] = float(len(closed))
        metrics["model_log_loss_on_closing_sample"] = _mean([r.log_loss for r, _ in closed])
        metrics["closing_log_loss"] = _mean([log_loss(p, r.actual) for r, p in closed])  # type: ignore[arg-type]

    metrics["over_2_5_log_loss"] = _mean([binary_log_loss(r.over_2_5, over(r)) for r in records])
    metrics["btts_log_loss"] = _mean(
        [binary_log_loss(r.btts_yes, r.home_goals > 0 and r.away_goals > 0) for r in records]
    )
    totals = [(r, r.market_over_2_5) for r in records if r.market_over_2_5 is not None]
    if totals:
        metrics["totals_market_n"] = float(len(totals))
        metrics["model_over_2_5_log_loss_on_market_sample"] = _mean(
            [binary_log_loss(r.over_2_5, over(r)) for r, _ in totals]
        )
        metrics["market_over_2_5_log_loss"] = _mean([binary_log_loss(p, over(r)) for r, p in totals])
    return metrics


def run_backtest(
    matches: list[Match],
    odds: list[OddsSnapshot] | None = None,
    config: BacktestConfig | None = None,
    *,
    calibration_records: list[PredictionRecord] | None = None,
) -> BacktestResult:
    """Walk-forward backtest; see :class:`BacktestConfig` for every option."""
    config = config or BacktestConfig()
    completed = [m for m in matches if m.home_goals is not None and m.away_goals is not None]
    folds = expanding_window(
        completed,
        initial_train_days=config.initial_train_days,
        test_days=config.test_days,
        step_days=config.step_days,
        mode=config.evaluation_mode,
        forecast_lead_minutes=config.forecast_lead_minutes,
        start_date=config.evaluation_start,
        end_date=config.evaluation_end,
    )
    odds_index = _index_odds(odds or [])
    records: list[PredictionRecord] = []
    fold_metrics: list[FoldMetrics] = []
    skipped = 0
    for fold_index, fold in enumerate(folds):
        fitted = fit_team_strength(
            list(fold.train),
            model=config.model,
            estimator=config.estimator,
            as_of=fold.train_end,
            half_life_days=config.half_life_days,
            l2_reg=config.l2_reg,
            sot_blend=config.sot_blend,
            max_goals=config.max_goals,
            use_cache=False,
        )
        fold_records = _predict_fold(fold, fitted, config, odds_index)
        eligible = sum(
            1 for m in fold.test if not config.competitions or m.competition in config.competitions
        )
        skipped += eligible - len(fold_records)
        if not fold_records:
            continue
        history = calibration_records if calibration_records is not None else records
        fold_records = _calibrate(fold_records, history, config)
        records.extend(fold_records)
        scores = aggregate_scores(asdict(r) for r in fold_records)
        fold_metrics.append(
            FoldMetrics(
                fold_index=fold_index,
                train_end=fold.train_end.isoformat(),
                test_start=fold.test_start.isoformat(),
                test_end=fold.test_end.isoformat(),
                n=len(fold_records),
                log_loss=scores["log_loss"],
                brier_score=scores["brier_score"],
                rps=scores["rps"],
            )
        )

    metrics = summarize_records(records)
    metrics["folds"] = float(len(fold_metrics))
    metrics["skipped_unknown_team"] = float(skipped)
    metadata: dict[str, object] = {
        "config": {
            key: (str(value) if isinstance(value, date) else value) for key, value in asdict(config).items()
        },
        "model_version": config.version,
        "forecast_timestamp_policy": f"kickoff_minus_{config.forecast_lead_minutes}m",
        "model_update_frequency": "per_fold"
        if config.evaluation_mode == "rolling_origin"
        else "fixed_origin",
        "calibration_window": "prior_predictions_only" if config.calibration_mode else None,
        "regimes": {
            "competition": analyze_by_regime(records, competition_regime),
            "team_history": analyze_by_regime(records, team_history_regime),
        },
    }
    return BacktestResult(records=records, metrics=metrics, fold_metrics=fold_metrics, metadata=metadata)
