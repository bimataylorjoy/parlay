"""Pre-match prediction of one fixture: every market, priced quotes and value bets.

This module is the engine behind ``parlay predict``. It is pure (no I/O):
callers supply fitted goal/corner models and the bookmaker prices, and get
back a :class:`FixturePrediction` that can be rendered or flattened to CSV.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from parlay.evaluation.anomaly import DecisionDiagnostics, diagnose
from parlay.models.corners import CornerStrengthModel
from parlay.models.team_strength import TeamStrengthModel
from parlay.prediction.markets import (
    asian_handicap,
    de_vig,
    expected_value,
    grouped_score_markets,
    kelly_criterion,
    match_winner_counts,
    most_likely_scores,
    outcome_probabilities,
    totals_probabilities,
)
from parlay.prediction.odds import to_decimal

OddsKey = tuple[str, str, float | None]  # (market, selection, line)


@dataclass(frozen=True, slots=True)
class PredictConfig:
    goal_lines: tuple[float, ...] = (1.5, 2.25, 2.5, 2.75, 3.5)
    corner_lines: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5)
    handicap_lines: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5)
    min_ev: float = 0.02
    min_edge: float = 0.0
    kelly_fraction: float = 0.25
    max_stake: float = 0.005
    de_vig_method: str = "shin"
    min_team_matches: int = 10


@dataclass(frozen=True, slots=True)
class Fixture:
    match_date: date
    home_team: str
    away_team: str
    home_raw: str = ""
    away_raw: str = ""
    odds: Mapping[OddsKey, float] = field(default_factory=dict)
    neutral: bool = False

    @property
    def label(self) -> str:
        return f"{self.home_raw or self.home_team} vs {self.away_raw or self.away_team}"


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """Model settlement probabilities for one selection, optionally priced."""

    market: str
    selection: str
    line: float | None
    win: float
    push: float
    loss: float
    odds: float | None = None
    market_probability: float | None = None
    kelly_fraction: float = 0.25
    max_stake: float = 0.005

    @property
    def name(self) -> str:
        line = (
            ""
            if self.line is None
            else f" {self.line:+g}"
            if self.market == "asian_handicap"
            else f" {self.line:g}"
        )
        return f"{self.market}:{self.selection}{line}"

    @property
    def fair_odds(self) -> float:
        """Break-even decimal odds (EV = 0), accounting for push: ``1 + loss / win``."""
        return 1.0 + self.loss / self.win if self.win > 0 else float("inf")

    @property
    def decisive_probability(self) -> float:
        """Win probability conditional on the bet not being void."""
        decisive = self.win + self.loss
        return self.win / decisive if decisive > 0 else 0.0

    @property
    def expected_value(self) -> float | None:
        return expected_value(self.win, self.odds, self.loss) if self.odds else None

    @property
    def edge(self) -> float | None:
        """Model minus de-vigged market probability (both conditional on no push)."""
        if self.market_probability is None:
            return None
        return self.decisive_probability - self.market_probability

    @property
    def stake(self) -> float:
        if not self.odds:
            return 0.0
        return kelly_criterion(
            self.win,
            self.odds,
            self.kelly_fraction,
            loss_probability=self.loss,
            max_stake=self.max_stake,
        )


@dataclass(frozen=True, slots=True)
class FixturePrediction:
    fixture: Fixture
    status: str
    message: str = ""
    expected_goals: tuple[float, float] | None = None
    outcome: dict[str, float] = field(default_factory=dict)
    grouped: dict[str, float] = field(default_factory=dict)
    totals: dict[float, dict[str, float]] = field(default_factory=dict)
    handicaps: dict[float, dict[str, float]] = field(default_factory=dict)
    top_scores: list[tuple[str, float]] = field(default_factory=list)
    expected_corners: tuple[float, float] | None = None
    corner_totals: dict[float, dict[str, float]] = field(default_factory=dict)
    corner_winner: dict[str, float] = field(default_factory=dict)
    quotes: list[MarketQuote] = field(default_factory=list)
    value_bets: list[MarketQuote] = field(default_factory=list)
    diagnostics: DecisionDiagnostics | None = None
    model_info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, Any]:
        """Flat, CSV-friendly representation."""
        fx = self.fixture
        row: dict[str, Any] = {
            "date": fx.match_date.isoformat(),
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "home_raw": fx.home_raw,
            "away_raw": fx.away_raw,
            "status": self.status,
            "message": self.message,
        }
        if not self.ok or self.expected_goals is None:
            return row
        r4 = lambda value: round(float(value), 4)  # noqa: E731
        fair = lambda p: round(1.0 / p, 2) if p > 1e-9 else None  # noqa: E731
        exp_h, exp_a = self.expected_goals
        row.update(
            exp_home_goals=round(exp_h, 3),
            exp_away_goals=round(exp_a, 3),
            exp_total_goals=round(exp_h + exp_a, 3),
        )
        for key, label in (("home_win", "home"), ("draw", "draw"), ("away_win", "away")):
            row[f"{label}_prob"] = r4(self.outcome[key])
            row[f"fair_{label}"] = fair(self.outcome[key])
        for key, value in self.grouped.items():
            row[f"{key}_prob"] = r4(value)
        for line, settlement in self.totals.items():
            tag = f"{line:g}".replace(".", "_")
            row[f"over_{tag}_prob"] = r4(settlement["over"])
            row[f"under_{tag}_prob"] = r4(settlement["under"])
            if settlement["push"] > 0:
                row[f"push_{tag}_prob"] = r4(settlement["push"])
        for line, settlement in self.handicaps.items():
            tag = f"{line:+g}".replace(".", "_").replace("+", "p").replace("-", "m")
            row[f"ah_home_{tag}_win_prob"] = r4(settlement["win"])
            row[f"ah_home_{tag}_push_prob"] = r4(settlement["push"])
        row["top_scores"] = ", ".join(f"{score} ({p:.1%})" for score, p in self.top_scores)
        if self.expected_corners is not None:
            ch, ca = self.expected_corners
            row.update(
                exp_home_corners=round(ch, 2),
                exp_away_corners=round(ca, 2),
                exp_total_corners=round(ch + ca, 2),
            )
            for line, settlement in self.corner_totals.items():
                tag = f"{line:g}".replace(".", "_")
                row[f"corners_over_{tag}_prob"] = r4(settlement["over"])
                row[f"corners_under_{tag}_prob"] = r4(settlement["under"])
            for key, value in self.corner_winner.items():
                row[f"corners_{key}_prob"] = r4(value)
        best = self.value_bets[0] if self.value_bets else None
        row.update(
            recommendation=best.name if best else "",
            recommendation_odds=best.odds if best else None,
            recommendation_ev=r4(best.expected_value) if best and best.expected_value is not None else None,
            recommendation_edge=r4(best.edge) if best and best.edge is not None else None,
            stake_fraction=r4(best.stake) if best else 0.0,
            value_bets="; ".join(
                f"{q.name} @{q.odds or 0:.2f} EV {q.expected_value or 0:+.1%}" for q in self.value_bets
            ),
        )
        if self.diagnostics is not None:
            row.update(
                anomaly_decision=self.diagnostics.decision,
                anomaly_flags=",".join(self.diagnostics.anomaly_flags),
                market_disagreement=r4(self.diagnostics.disagreement)
                if self.diagnostics.disagreement is not None
                else None,
            )
        row.update(self.model_info)
        return row


# --------------------------------------------------------------------------- #
# Parsing bookmaker columns
# --------------------------------------------------------------------------- #

_NUMBER = r"(\d+(?:\.\d+)?)"
LEGACY_CORNER_MIN_LINE = 5.0


def _side(symbol: str) -> str:
    return "over" if symbol == ">" else "under"


def parse_odds_columns(
    row: Mapping[str, str], *, prefix: str = "B365", odds_format: str = "decimal"
) -> dict[OddsKey, float]:
    """Read football-data style odds columns into ``{(market, selection, line): decimal_odds}``.

    Recognized (with ``prefix`` = ``B365``): ``B365H/D/A``; ``B365>2.5`` /
    ``B365<2.5`` (any line); ``B365BTTS_Y/N``; ``B365CRN>9.5`` / ``B365CRN<9.5``
    (corners; legacy ``B365C>9.5`` accepted for lines >= 5); ``B365AHH`` / ``B365AHA``
    with the home handicap in ``AHh``;
    ``B3651X`` / ``B365X2`` / ``B36512`` (double chance). Empty cells are skipped.
    """
    p = re.escape(prefix)
    patterns: list[tuple[re.Pattern[str], Any]] = [
        (
            re.compile(rf"^{p}([HDA])$"),
            lambda m: ("1x2", {"H": "home", "D": "draw", "A": "away"}[m[1]], None),
        ),
        (re.compile(rf"^{p}CRN([<>]){_NUMBER}$"), lambda m: ("corners_totals", _side(m[1]), float(m[2]))),
        # Legacy corners columns (``B365C>9.5``). football-data uses ``B365C>2.5`` for
        # *closing* goal totals, so only lines that can only be corner lines are accepted.
        (
            re.compile(rf"^{p}C([<>]){_NUMBER}$"),
            lambda m: (
                ("corners_totals", _side(m[1]), float(m[2]))
                if float(m[2]) >= LEGACY_CORNER_MIN_LINE
                else None
            ),
        ),
        (re.compile(rf"^{p}([<>]){_NUMBER}$"), lambda m: ("totals", _side(m[1]), float(m[2]))),
        (re.compile(rf"^{p}BTTS_([YN])$"), lambda m: ("btts", "yes" if m[1] == "Y" else "no", None)),
        (re.compile(rf"^{p}(1X|X2|12)$"), lambda m: ("double_chance", m[1].lower(), None)),
    ]
    odds: dict[OddsKey, float] = {}
    for column, raw in row.items():
        if column is None or raw is None or not str(raw).strip():
            continue
        column = column.strip()
        for pattern, build in patterns:
            match = pattern.match(column)
            if match:
                key = build(match)
                if key is not None:
                    odds[key] = to_decimal(str(raw).strip(), odds_format)
                break
    handicap = (row.get("AHh") or "").strip()
    if handicap:
        line = float(handicap)
        for side, column in (("home", f"{prefix}AHH"), ("away", f"{prefix}AHA")):
            value = (row.get(column) or "").strip()
            if value:
                odds[("asian_handicap", side, line)] = to_decimal(value, odds_format)
    return odds


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def _settlements(matrix: np.ndarray, goal_lines, handicap_lines) -> tuple[dict, dict]:
    totals = {line: totals_probabilities(matrix, line) for line in goal_lines}
    handicaps = {line: asian_handicap(matrix, line) for line in handicap_lines}
    return totals, handicaps


def _build_quotes(
    fixture: Fixture,
    matrix: np.ndarray,
    corner_matrix: np.ndarray | None,
    outcome: dict[str, float],
    grouped: dict[str, float],
    config: PredictConfig,
) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []

    def add(market: str, selection: str, line: float | None, win: float, push: float, loss: float) -> None:
        quotes.append(
            MarketQuote(
                market,
                selection,
                line,
                win,
                push,
                loss,
                odds=fixture.odds.get((market, selection, line)),
                kelly_fraction=config.kelly_fraction,
                max_stake=config.max_stake,
            )
        )

    for selection, key in (("home", "home_win"), ("draw", "draw"), ("away", "away_win")):
        add("1x2", selection, None, outcome[key], 0.0, 1.0 - outcome[key])
    for selection, key in (
        ("1x", "double_chance_1x"),
        ("x2", "double_chance_x2"),
        ("12", "double_chance_12"),
    ):
        add("double_chance", selection, None, grouped[key], 0.0, 1.0 - grouped[key])
    add("btts", "yes", None, grouped["btts_yes"], 0.0, grouped["btts_no"])
    add("btts", "no", None, grouped["btts_no"], 0.0, grouped["btts_yes"])

    priced_lines = {key[2] for key in fixture.odds if key[0] == "totals"}
    for line in sorted(set(config.goal_lines) | priced_lines):  # type: ignore[type-var]
        s = totals_probabilities(matrix, line)  # type: ignore[arg-type]
        add("totals", "over", line, s["over"], s["push"], s["under"])
        add("totals", "under", line, s["under"], s["push"], s["over"])
    priced_handicaps = {key[2] for key in fixture.odds if key[0] == "asian_handicap"}
    for line in sorted(priced_handicaps):  # type: ignore[type-var]
        s = asian_handicap(matrix, line)  # type: ignore[arg-type]
        add("asian_handicap", "home", line, s["win"], s["push"], s["loss"])
        add("asian_handicap", "away", line, s["loss"], s["push"], s["win"])
    if corner_matrix is not None:
        priced_corner_lines = {key[2] for key in fixture.odds if key[0] == "corners_totals"}
        for line in sorted(set(config.corner_lines) | priced_corner_lines):  # type: ignore[type-var]
            s = totals_probabilities(corner_matrix, line)  # type: ignore[arg-type]
            add("corners_totals", "over", line, s["over"], s["push"], s["under"])
            add("corners_totals", "under", line, s["under"], s["push"], s["over"])

    # Attach de-vigged market probabilities where the full market is priced.
    groups: dict[tuple[str, float | None], list[int]] = {}
    for i, quote in enumerate(quotes):
        if quote.odds:
            groups.setdefault((quote.market, quote.line), []).append(i)
    complete_size = {"1x2": 3, "double_chance": 0}
    for (market, _line), indices in groups.items():
        if len(indices) != complete_size.get(market, 2):
            continue
        fair = de_vig({str(i): 1.0 / quotes[i].odds for i in indices}, method=config.de_vig_method)  # type: ignore[operator]
        for i in indices:
            q = quotes[i]
            quotes[i] = MarketQuote(
                q.market,
                q.selection,
                q.line,
                q.win,
                q.push,
                q.loss,
                q.odds,
                fair[str(i)],
                q.kelly_fraction,
                q.max_stake,
            )
    return quotes


def predict_fixture(
    fixture: Fixture,
    goal_model: TeamStrengthModel,
    corner_model: CornerStrengthModel | None = None,
    config: PredictConfig | None = None,
) -> FixturePrediction:
    """Predict every supported market for one fixture and select value bets."""
    config = config or PredictConfig()
    missing = [team for team in (fixture.home_team, fixture.away_team) if team not in goal_model]
    if missing:
        return FixturePrediction(
            fixture, status="unknown_team", message=f"no pre-match history for {', '.join(missing)}"
        )

    matrix = goal_model.score_matrix(fixture.home_team, fixture.away_team, fixture.neutral)
    expected = goal_model.expected_goals(fixture.home_team, fixture.away_team, fixture.neutral)
    outcome = outcome_probabilities(matrix)
    grouped = grouped_score_markets(matrix)
    totals, handicaps = _settlements(matrix, config.goal_lines, config.handicap_lines)

    corner_matrix = expected_corners = None
    corner_totals: dict[float, dict[str, float]] = {}
    corner_winner: dict[str, float] = {}
    if corner_model is not None and corner_model.fit_converged:
        corner_matrix = corner_model.corner_matrix(fixture.home_team, fixture.away_team)
        expected_corners = corner_model.expected_corners(fixture.home_team, fixture.away_team)
        corner_totals = {line: totals_probabilities(corner_matrix, line) for line in config.corner_lines}
        corner_winner = match_winner_counts(corner_matrix)

    quotes = _build_quotes(fixture, matrix, corner_matrix, outcome, grouped, config)
    n_hist = min(
        goal_model.historical_matches.get(fixture.home_team, 0),
        goal_model.historical_matches.get(fixture.away_team, 0),
    )
    market_home = next(
        (q.market_probability for q in quotes if q.market == "1x2" and q.selection == "home"), None
    )
    has_1x2 = all(("1x2", sel, None) in fixture.odds for sel in ("home", "draw", "away"))
    diagnostics = diagnose(
        outcome["home_win"],
        market_home,
        n_historical=n_hist,
        is_promoted=n_hist < config.min_team_matches,
        lambda_total=expected[0] + expected[1],
        data_missing=not has_1x2,
    )
    value_bets: list[MarketQuote] = []
    if diagnostics.decision != "REJECT":
        value_bets = sorted(
            (
                q
                for q in quotes
                if q.expected_value is not None
                and q.expected_value >= config.min_ev
                and (q.edge is None or q.edge >= config.min_edge)
                and q.stake > 0
            ),
            key=lambda q: q.expected_value or 0.0,
            reverse=True,
        )
    return FixturePrediction(
        fixture=fixture,
        status="ok",
        expected_goals=expected,
        outcome=outcome,
        grouped=grouped,
        totals=totals,
        handicaps=handicaps,
        top_scores=most_likely_scores(matrix, 5),
        expected_corners=expected_corners,
        corner_totals=corner_totals,
        corner_winner=corner_winner,
        quotes=quotes,
        value_bets=value_bets,
        diagnostics=diagnostics,
        model_info={
            "model": goal_model.model,
            "rho": round(goal_model.rho, 4),
            "home_advantage": round(goal_model.home_advantage, 4),
            "n_historical": n_hist,
            "fit_converged": goal_model.fit_converged,
        },
    )
