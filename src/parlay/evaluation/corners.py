"""Temporal evaluation for the independent corners model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from parlay.data.schemas import Match
from parlay.evaluation.market import evaluate_binary_market
from parlay.evaluation.temporal import expanding_window
from parlay.models.corners import corner_totals_probabilities, fit_corner_strength


@dataclass(frozen=True, slots=True)
class CornerPredictionRecord:
    match_id: str
    forecast_timestamp: str
    home_team: str
    away_team: str
    line: float
    over_probability: float
    push_probability: float
    under_probability: float
    actual_market: str
    home_corners: int
    away_corners: int
    fit_converged: bool
    fit_warning: str | None

    @property
    def actual(self) -> str:
        return self.actual_market


@dataclass(frozen=True, slots=True)
class CornersBacktestResult:
    records: list[CornerPredictionRecord]
    metrics: dict[str, float]
    metadata: dict[str, object]


def _settle(actual_total: int, line: float) -> str:
    if line % 0.5 == 0:
        if actual_total > line:
            return "over"
        if actual_total < line:
            return "under"
        return "push"
    # Quarter lines are represented as two half/integer lines. A match is
    # recorded as push only when both component bets push, otherwise use the
    # directional result of the split stake.
    lower = int(line * 2) / 2
    upper = lower + 0.5
    outcomes = (_settle(actual_total, lower), _settle(actual_total, upper))
    if outcomes[0] == outcomes[1]:
        return outcomes[0]
    return "split"


def run_corners_backtest(
    matches: list[Match], *, line: float = 10.5, initial_train_days: int = 730,
    test_days: int = 30, step_days: int | None = None,
    half_life_days: float = 365.0, dispersion: float = 15.0,
    max_corners: int | None = 20, evaluation_mode: str = "rolling_origin",
    forecast_lead_minutes: int = 60,
) -> CornersBacktestResult:
    """Fit and score corners independently on chronological folds."""
    if line * 4 != int(line * 4):
        raise ValueError("line must be a multiple of 0.25")
    folds = expanding_window(
        matches, initial_train_days=initial_train_days, test_days=test_days,
        step_days=step_days, mode=evaluation_mode,
        forecast_lead_minutes=forecast_lead_minutes,
    )
    records: list[CornerPredictionRecord] = []
    for fold in folds:
        fitted = fit_corner_strength(
            list(fold.train), half_life_days=half_life_days,
            dispersion=dispersion, max_corners=max_corners,
        )
        for match in fold.test:
            if match.home_corners is None or match.away_corners is None:
                continue
            matrix = fitted.corner_matrix(match.home_team, match.away_team)
            probabilities = corner_totals_probabilities(matrix, line)
            kickoff = match.kickoff_at
            if kickoff is None:
                forecast = datetime.combine(match.date, datetime.min.time(), timezone.utc)
            else:
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
                forecast = kickoff - timedelta(minutes=forecast_lead_minutes)
            records.append(CornerPredictionRecord(
                match_id=match.match_id,
                forecast_timestamp=forecast.isoformat(),
                home_team=match.home_team,
                away_team=match.away_team,
                line=line,
                over_probability=probabilities["over"] + 0.5 * probabilities["push"],
                push_probability=probabilities["push"],
                under_probability=probabilities["under"] + 0.5 * probabilities["push"],
                actual_market=_settle(match.home_corners + match.away_corners, line),
                home_corners=match.home_corners,
                away_corners=match.away_corners,
                fit_converged=fitted.fit_converged,
                fit_warning=fitted.fit_warning,
            ))
    scored = [record for record in records if record.actual_market != "split"]
    metrics = evaluate_binary_market(
        scored, probability_attr="over_probability", actual_attr="actual_market"
    ) if scored else {"n": 0.0, "pushes": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    metrics["records"] = float(len(records))
    metrics["split_settlements"] = float(len(records) - len(scored))
    return CornersBacktestResult(
        records=records,
        metrics=metrics,
        metadata={
            "market": "corners_total",
            "line": line,
            "evaluation_mode": evaluation_mode,
            "forecast_timestamp_policy": f"kickoff_minus_{forecast_lead_minutes}m",
            "model_update_frequency": "per_fold" if evaluation_mode == "rolling_origin" else "fixed_origin",
            "dispersion": dispersion,
            "max_corners": max_corners,
        },
    )
