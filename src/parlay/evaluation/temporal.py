"""Expanding-window splits with explicit anti-leakage semantics.

Supports fixed-origin (frozen train) vs rolling-origin (expanding train) (§2).
"""

from dataclasses import dataclass
from datetime import date, timedelta, datetime, timezone, time
from typing import Iterable, Literal

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches
from parlay.data.information import InformationSet, forecast_timestamp_for_match


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train: tuple[Match, ...]
    test: tuple[Match, ...]
    train_end: date
    test_start: date
    test_end: date
    evaluation_mode: str = "rolling_origin"
    forecast_policy: str = "kickoff_minus_60m"
    model_update_frequency: str = "per_fold"


def expanding_window(
    matches: Iterable[Match],
    *,
    initial_train_days: int,
    test_days: int,
    step_days: int | None = None,
    mode: Literal["rolling_origin", "fixed_origin"] = "rolling_origin",
    forecast_lead_minutes: int = 60,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TemporalFold]:
    """Walk-forward splits with InformationSet semantics (§1,§2).

    rolling_origin: train expands each fold (production-like)
    fixed_origin: train frozen at initial_train_end
    Forecast timestamp = kickoff - lead (or midnight if no kickoff), and
    a match is in train only if its result_known_at <= forecast of first test.
    For simplicity we use date-based gating with InformationSet fallback.
    """
    rows = sorted(validate_matches(matches), key=lambda row: (row.date, row.match_id))
    if initial_train_days <= 0 or test_days <= 0:
        raise ValueError("initial_train_days and test_days must be positive")
    step_days = step_days or test_days
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if mode not in ("rolling_origin", "fixed_origin"):
        raise ValueError("mode must be rolling_origin or fixed_origin")
    first_date = rows[0].date
    last_date = rows[-1].date
    folds: list[TemporalFold] = []
    train_end_fixed = first_date + timedelta(days=initial_train_days - 1)
    train_end = train_end_fixed
    fixed_forecast = datetime.combine(
        train_end_fixed + timedelta(days=1), time(0, 0), tzinfo=timezone.utc
    ) - timedelta(minutes=forecast_lead_minutes)
    while train_end < last_date:
        test_start = train_end + timedelta(days=1)
        test_end = min(test_start + timedelta(days=test_days - 1), last_date)
        # Determine train set based on mode
        if mode == "fixed_origin":
            effective_train_end = train_end_fixed
        else:
            effective_train_end = train_end
        # Use InformationSet for eligibility: forecast is test_start 00:00 - lead
        # Conservative: a train match is knowable if known_at <= forecast of test_start
        forecast_for_train = fixed_forecast if mode == "fixed_origin" else (
            datetime.combine(test_start, time(0, 0), tzinfo=timezone.utc)
            - timedelta(minutes=forecast_lead_minutes)
        )
        train_info = InformationSet(as_of=forecast_for_train)
        train = tuple(m for m in rows if train_info.is_result_knowable(m) and m.date <= effective_train_end)
        test = tuple(
            row for row in rows
            if test_start <= row.date <= test_end
            and (start_date is None or row.date >= start_date)
            and (end_date is None or row.date <= end_date)
        )
        if train and test:
            folds.append(TemporalFold(
                train, test, effective_train_end, test_start, test_end,
                evaluation_mode=mode,
                forecast_policy=f"kickoff_minus_{forecast_lead_minutes}m",
                model_update_frequency="per_fold" if mode == "rolling_origin" else "fixed_origin",
            ))
        train_end += timedelta(days=step_days)
    return folds
