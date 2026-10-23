"""Uncertainty-first prediction API for a single match."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from parlay.evaluation.anomaly import diagnose
from parlay.models.team_strength import TeamStrengthModel
from parlay.prediction.markets import (
    correct_score_probabilities,
    de_vig,
    grouped_score_markets,
    outcome_probabilities,
    totals_probabilities,
)


@dataclass(frozen=True, slots=True)
class ExpectedRates:
    lambda_home_mean: float
    lambda_home_ci: tuple[float, float] | None
    lambda_away_mean: float
    lambda_away_ci: tuple[float, float] | None
    exp_total_goals_mean: float


@dataclass(frozen=True, slots=True)
class Uncertainty:
    posterior_std_home: float | None = None
    posterior_std_away: float | None = None
    market_disagreement: float | None = None
    anomaly_flags: tuple[str, ...] = ()
    decision: str = "PASS"


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """1X2 probabilities plus expected rates, uncertainty and derived markets.

    ``result["home_win"]`` is shorthand for ``result.probabilities["home_win"]``.
    """

    probabilities: dict[str, float]
    expected_rates: ExpectedRates
    uncertainty: Uncertainty
    model_metadata: dict[str, Any] = field(default_factory=dict)
    score_matrix: np.ndarray | None = field(default=None, repr=False)
    grouped_markets: dict[str, float] = field(default_factory=dict)
    totals: dict[float, dict[str, float]] = field(default_factory=dict)
    correct_scores: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.probabilities[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "expected_rates": {
                "lambda_home_mean": self.expected_rates.lambda_home_mean,
                "lambda_home_ci": self.expected_rates.lambda_home_ci,
                "lambda_away_mean": self.expected_rates.lambda_away_mean,
                "lambda_away_ci": self.expected_rates.lambda_away_ci,
            },
            "uncertainty": {
                "posterior_std_home": self.uncertainty.posterior_std_home,
                "posterior_std_away": self.uncertainty.posterior_std_away,
                "market_disagreement": self.uncertainty.market_disagreement,
                "anomaly_flags": list(self.uncertainty.anomaly_flags),
                "decision": self.uncertainty.decision,
            },
            "model_metadata": self.model_metadata,
        }


def predict_with_uncertainty(
    model: TeamStrengthModel,
    home_team: str,
    away_team: str,
    *,
    idata: Any = None,
    market_odds: dict[str, float] | None = None,
    n_historical: int | None = None,
    is_promoted: bool = False,
    totals_lines: tuple[float, ...] = (1.5, 2.5, 2.75, 3.5),
) -> PredictionResult:
    """Predict one match from a fitted model.

    With a Bayesian posterior (``idata`` or ``model.posterior``) the 1X2
    probabilities are the posterior predictive mean and rate intervals are
    reported; otherwise only event randomness is represented.
    ``market_odds`` (decimal, keys ``home``/``draw``/``away``) feeds the
    anomaly layer after Shin de-vigging.
    """
    exp_h, exp_a = model.expected_goals(home_team, away_team)
    matrix = model.score_matrix(home_team, away_team)
    probabilities = outcome_probabilities(matrix)
    if n_historical is None:
        n_historical = min(
            model.historical_matches.get(home_team, 0), model.historical_matches.get(away_team, 0)
        )

    idata = idata if idata is not None else model.posterior
    std_h = std_a = None
    ci_h = ci_a = None
    posterior_error = None
    if idata is not None:
        try:
            from parlay.prediction.bayesian_predictive import posterior_predictive_1x2

            predictive = posterior_predictive_1x2(
                idata, home_team, away_team, model.teams, model=model.model, rho=model.rho
            )
            probabilities = predictive.mean_probability
            ci_h, ci_a = predictive.lambda_home_ci, predictive.lambda_away_ci
            # 90% interval of a normal spans 3.29 standard deviations
            std_h = (ci_h[1] - ci_h[0]) / 3.29
            std_a = (ci_a[1] - ci_a[0]) / 3.29
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            posterior_error = f"posterior predictive unavailable: {exc}"

    market_home = None
    if market_odds and all(float(market_odds.get(key, 0)) > 1.0 for key in ("home", "draw", "away")):
        implied = {key: 1.0 / float(market_odds[key]) for key in ("home", "draw", "away")}
        market_home = de_vig(implied, method="shin")["home"]
    diagnostics = diagnose(
        probabilities["home_win"],
        market_home,
        posterior_std=std_h,
        n_historical=n_historical,
        is_promoted=is_promoted,
        lambda_total=exp_h + exp_a,
    )
    return PredictionResult(
        probabilities=probabilities,
        expected_rates=ExpectedRates(exp_h, ci_h, exp_a, ci_a, exp_h + exp_a),
        uncertainty=Uncertainty(
            posterior_std_home=std_h,
            posterior_std_away=std_a,
            market_disagreement=diagnostics.disagreement,
            anomaly_flags=diagnostics.anomaly_flags,
            decision=diagnostics.decision,
        ),
        model_metadata={
            "model": model.model,
            "rho": model.rho,
            "home_advantage": model.home_advantage,
            "fit_converged": model.fit_converged,
            "fit_warning": model.fit_warning,
            "posterior_predictive_error": posterior_error,
            "uncertainty_source": "posterior_predictive"
            if idata is not None and posterior_error is None
            else "none",
        },
        score_matrix=matrix,
        grouped_markets=grouped_score_markets(matrix),
        totals={line: totals_probabilities(matrix, line) for line in totals_lines},
        correct_scores=correct_score_probabilities(matrix, max_home=4, max_away=4),
    )
