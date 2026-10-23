"""Walk-forward evaluation of the corner model, independent of the goal model."""

from __future__ import annotations

from dataclasses import dataclass

from parlay.data.schemas import Match
from parlay.evaluation.backtest import forecast_timestamp
from parlay.evaluation.market import evaluate_binary_market
from parlay.evaluation.temporal import expanding_window
from parlay.models.corners import (
    DEFAULT_CORNER_HALF_LIFE_DAYS,
    DEFAULT_CORNER_L2_REG,
    DEFAULT_MAX_CORNERS,
    fit_corner_strength,
)
from parlay.prediction.markets import totals_probabilities


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
    expected_home_corners: float
    expected_away_corners: float
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


def settle_total(actual_total: int, line: float) -> str:
    """``over`` / ``under`` / ``push`` — or ``split`` when a quarter line's halves settle differently."""
    if (line * 2) % 1 == 0:
        if actual_total > line:
            return "over"
        if actual_total < line:
            return "under"
        return "push"
    lower = int(line * 2) / 2
    outcomes = (settle_total(actual_total, lower), settle_total(actual_total, lower + 0.5))
    return outcomes[0] if outcomes[0] == outcomes[1] else "split"


def run_corners_backtest(
    matches: list[Match],
    *,
    line: float = 10.5,
    initial_train_days: int = 730,
    test_days: int = 7,
    step_days: int | None = None,
    half_life_days: float | None = DEFAULT_CORNER_HALF_LIFE_DAYS,
    l2_reg: float = DEFAULT_CORNER_L2_REG,
    dispersion: float | None = None,
    max_corners: int = DEFAULT_MAX_CORNERS,
    evaluation_mode: str = "rolling_origin",
    forecast_lead_minutes: int = 60,
) -> CornersBacktestResult:
    """Fit and score the corners total market on chronological folds."""
    if line * 4 != int(line * 4):
        raise ValueError("line must be a multiple of 0.25")
    completed = [m for m in matches if m.home_goals is not None and m.away_goals is not None]
    folds = expanding_window(
        completed,
        initial_train_days=initial_train_days,
        test_days=test_days,
        step_days=step_days,
        mode=evaluation_mode,
        forecast_lead_minutes=forecast_lead_minutes,
    )
    records: list[CornerPredictionRecord] = []
    for fold in folds:
        fitted = fit_corner_strength(
            list(fold.train),
            as_of=fold.train_end,
            half_life_days=half_life_days,
            l2_reg=l2_reg,
            dispersion=dispersion,
            max_corners=max_corners,
        )
        for match in fold.test:
            if match.home_corners is None or match.away_corners is None:
                continue
            probabilities = totals_probabilities(fitted.corner_matrix(match.home_team, match.away_team), line)
            exp_home, exp_away = fitted.expected_corners(match.home_team, match.away_team)
            records.append(
                CornerPredictionRecord(
                    match_id=match.match_id,
                    forecast_timestamp=forecast_timestamp(match, forecast_lead_minutes).isoformat(),
                    home_team=match.home_team,
                    away_team=match.away_team,
                    line=line,
                    over_probability=probabilities["over"] + 0.5 * probabilities["push"],
                    push_probability=probabilities["push"],
                    under_probability=probabilities["under"] + 0.5 * probabilities["push"],
                    actual_market=settle_total(match.home_corners + match.away_corners, line),
                    home_corners=match.home_corners,
                    away_corners=match.away_corners,
                    expected_home_corners=exp_home,
                    expected_away_corners=exp_away,
                    fit_converged=fitted.fit_converged,
                    fit_warning=fitted.fit_warning,
                )
            )
    scored = [record for record in records if record.actual_market != "split"]
    metrics = (
        evaluate_binary_market(scored, probability_attr="over_probability", actual_attr="actual_market")
        if scored
        else {"n": 0.0, "pushes": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    )
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
            "dispersion": dispersion if dispersion is not None else "estimated",
            "max_corners": max_corners,
        },
    )
