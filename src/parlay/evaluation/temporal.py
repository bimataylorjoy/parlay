"""Expanding-window splits with explicit anti-leakage semantics."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train: tuple[Match, ...]
    test: tuple[Match, ...]
    train_end: date
    test_start: date
    test_end: date


def expanding_window(
    matches: Iterable[Match],
    *,
    initial_train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[TemporalFold]:
    rows = sorted(validate_matches(matches), key=lambda row: (row.date, row.match_id))
    if initial_train_days <= 0 or test_days <= 0:
        raise ValueError("initial_train_days and test_days must be positive")
    step_days = step_days or test_days
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    first_date = rows[0].date
    last_date = rows[-1].date
    folds: list[TemporalFold] = []
    train_end = first_date + timedelta(days=initial_train_days - 1)
    while train_end < last_date:
        test_start = train_end + timedelta(days=1)
        test_end = min(test_start + timedelta(days=test_days - 1), last_date)
        train = tuple(row for row in rows if row.date <= train_end)
        test = tuple(row for row in rows if test_start <= row.date <= test_end)
        if train and test:
            folds.append(TemporalFold(train, test, train_end, test_start, test_end))
        train_end += timedelta(days=step_days)
    return folds
