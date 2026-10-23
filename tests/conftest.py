"""Shared fixtures: match factories and a synthetic league with known strengths."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from parlay.data.schemas import Match
from parlay.models.team_strength import clear_fit_cache


@pytest.fixture(autouse=True)
def _fresh_fit_cache():
    clear_fit_cache()
    yield
    clear_fit_cache()


def make_match(
    match_id: str = "m1",
    day: date = date(2024, 1, 1),
    home: str = "A",
    away: str = "B",
    home_goals: int | None = 1,
    away_goals: int | None = 0,
    **extra,
) -> Match:
    values = dict(
        match_id=match_id,
        date=day,
        competition="EPL",
        season="2023-24",
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
    )
    values.update(extra)
    return Match(**values)


def simple_series(n: int = 12, start: date = date(2020, 1, 1)) -> list[Match]:
    """``n`` daily A-vs-B matches with varied scores."""
    return [make_match(str(i), start + timedelta(days=i), "A", "B", i % 3, (i + 1) % 2) for i in range(n)]


TRUE_ATTACK = {"Lions": 0.45, "Hawks": 0.2, "Bears": 0.0, "Wolves": -0.15, "Foxes": -0.5}
TRUE_DEFENSE = {"Lions": 0.35, "Hawks": 0.1, "Bears": 0.0, "Wolves": -0.1, "Foxes": -0.35}
TRUE_INTERCEPT = math.log(1.3)
TRUE_HOME_ADVANTAGE = 0.25


def synthetic_league(
    rounds: int = 12,
    *,
    seed: int = 7,
    start: date = date(2021, 8, 1),
    corners: bool = False,
    with_kickoff: bool = True,
) -> list[Match]:
    """Double round-robins of Poisson goals from ``TRUE_*`` parameters, one match day per week."""
    rng = np.random.default_rng(seed)
    teams = list(TRUE_ATTACK)
    matches = []
    day = start
    counter = 0
    for _ in range(rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam_h = math.exp(
                    TRUE_INTERCEPT + TRUE_HOME_ADVANTAGE + TRUE_ATTACK[home] - TRUE_DEFENSE[away]
                )
                lam_a = math.exp(TRUE_INTERCEPT + TRUE_ATTACK[away] - TRUE_DEFENSE[home])
                extra = {}
                if corners:
                    extra["home_corners"] = int(rng.negative_binomial(8, 8 / (8 + 5.5 * lam_h / 1.6)))
                    extra["away_corners"] = int(rng.negative_binomial(8, 8 / (8 + 4.5 * lam_a / 1.3)))
                if with_kickoff:
                    extra["kickoff_at"] = datetime.combine(
                        day, datetime.min.time(), timezone.utc
                    ) + timedelta(hours=15)
                matches.append(
                    make_match(
                        f"s{counter}",
                        day,
                        home,
                        away,
                        int(rng.poisson(lam_h)),
                        int(rng.poisson(lam_a)),
                        **extra,
                    )
                )
                counter += 1
                day += timedelta(days=1)
    return matches


@pytest.fixture
def league() -> list[Match]:
    return synthetic_league()
