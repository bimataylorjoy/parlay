"""Walk-forward splits with explicit anti-leakage semantics.

``rolling_origin`` refits on an expanding training window at every fold
(production-like); ``fixed_origin`` freezes the training window at the
initial cutoff. Training eligibility uses :class:`InformationSet`, so a match
is used only if its result was knowable before the first forecast of the fold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from parlay.data.information import InformationSet
from parlay.data.schemas import Match
from parlay.data.validation import validate_matches

EvaluationMode = Literal["rolling_origin", "fixed_origin"]
EVALUATION_MODES: tuple[str, ...] = ("rolling_origin", "fixed_origin")


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train: tuple[Match, ...]
    test: tuple[Match, ...]
    train_end: date
    test_start: date
    test_end: date
    evaluation_mode: str = "rolling_origin"
    forecast_policy: str = "kickoff_minus_60m"


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), tzinfo=timezone.utc)


def expanding_window(
    matches: Iterable[Match],
    *,
    initial_train_days: int,
    test_days: int,
    step_days: int | None = None,
    mode: str = "rolling_origin",
    forecast_lead_minutes: int = 60,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TemporalFold]:
    """Chronological train/test folds; test windows are contiguous and non-overlapping by default."""
    rows = sorted(validate_matches(matches), key=lambda row: (row.date, row.match_id))
    if initial_train_days <= 0 or test_days <= 0:
        raise ValueError("initial_train_days and test_days must be positive")
    step_days = step_days or test_days
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if mode not in EVALUATION_MODES:
        raise ValueError(f"mode must be one of {EVALUATION_MODES}")
    if not rows:
        return []

    first_date, last_date = rows[0].date, rows[-1].date
    initial_end = first_date + timedelta(days=initial_train_days - 1)
    lead = timedelta(minutes=forecast_lead_minutes)
    policy = f"kickoff_minus_{forecast_lead_minutes}m"
    folds: list[TemporalFold] = []
    train_end = initial_end
    while train_end < last_date:
        test_start = train_end + timedelta(days=1)
        test_end = min(test_start + timedelta(days=test_days - 1), last_date)
        effective_end = initial_end if mode == "fixed_origin" else train_end
        info = InformationSet(as_of=_midnight(effective_end + timedelta(days=1)) - lead)
        train = tuple(m for m in rows if m.date <= effective_end and info.is_result_knowable(m))
        test = tuple(
            row
            for row in rows
            if test_start <= row.date <= test_end
            and (start_date is None or row.date >= start_date)
            and (end_date is None or row.date <= end_date)
        )
        if train and test:
            folds.append(
                TemporalFold(
                    train,
                    test,
                    effective_end,
                    test_start,
                    test_end,
                    evaluation_mode=mode,
                    forecast_policy=policy,
                )
            )
        train_end += timedelta(days=step_days)
    return folds
