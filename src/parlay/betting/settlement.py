"""Settle bets on 1X2, goal totals and Asian handicaps (integer, half and quarter lines)."""

from __future__ import annotations

from dataclasses import dataclass

from parlay.data.odds_columns import parse_market

SELECTIONS = {"1x2": ("home", "draw", "away"), "totals": ("over", "under"), "ah": ("home", "away")}


def selections_for(market: str) -> tuple[str, ...]:
    kind, _ = parse_market(market)
    if kind not in SELECTIONS:
        raise ValueError(f"unsupported market: {market}")
    return SELECTIONS[kind]


@dataclass(frozen=True, slots=True)
class Settlement:
    """Shares of the stake that win, are returned, and lose (sum to 1)."""

    win: float
    push: float
    loss: float

    @property
    def label(self) -> str:
        if self.win == 1:
            return "win"
        if self.loss == 1:
            return "loss"
        if self.push == 1:
            return "push"
        return "half_win" if self.win > 0 else "half_loss"

    def profit(self, stake: float, odds: float) -> float:
        return stake * (self.win * (odds - 1.0) - self.loss)


def _line_settlement(margin: float) -> Settlement:
    """Settle ``margin`` = goals in favour of the bet minus the line (quarter lines split)."""
    if (margin * 2) % 1 != 0:  # quarter line: half the stake on each adjacent line
        lower, upper = _line_settlement(margin - 0.25), _line_settlement(margin + 0.25)
        return Settlement(
            0.5 * (lower.win + upper.win), 0.5 * (lower.push + upper.push), 0.5 * (lower.loss + upper.loss)
        )
    if margin > 0:
        return Settlement(1.0, 0.0, 0.0)
    if margin < 0:
        return Settlement(0.0, 0.0, 1.0)
    return Settlement(0.0, 1.0, 0.0)


def settle(market: str, selection: str, home_goals: int, away_goals: int) -> Settlement:
    kind, line = parse_market(market)
    if selection not in selections_for(market):
        raise ValueError(f"unknown selection {selection!r} for {market}")
    if kind == "1x2":
        outcome = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
        return Settlement(1.0, 0.0, 0.0) if outcome == selection else Settlement(0.0, 0.0, 1.0)
    assert line is not None
    if kind == "totals":
        margin = home_goals + away_goals - line
        return _line_settlement(margin if selection == "over" else -margin)
    margin = home_goals - away_goals + line  # home handicap
    return _line_settlement(margin if selection == "home" else -margin)
