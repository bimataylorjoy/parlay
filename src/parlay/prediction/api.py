"""Uncertainty-first prediction API (§15)."""

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


@dataclass(frozen=True, slots=True)
class ExpectedRates:
    lambda_home_mean: float
    lambda_home_ci: tuple[float, float]
    lambda_away_mean: float
    lambda_away_ci: tuple[float, float]
    exp_total_goals_mean: float
    exp_total_goals_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Uncertainty:
    # Parameter uncertainty vs event randomness
    posterior_std_home: float | None = None
    posterior_std_away: float | None = None
    market_disagreement: float | None = None
    calibration_error: float | None = None
    anomaly_flags: tuple[str, ...] = ()
    decision: str = "PASS"  # RESEARCH_SIGNAL/WATCH/PASS/REJECT


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Rich prediction output (§15).

    Simple access: `result.probabilities["home_win"]`
    Full: `result.expected_rates.lambda_home_mean`, `result.uncertainty`, `result.model_metadata`
    """

    probabilities: dict[str, float]  # 1X2 mean
    expected_rates: ExpectedRates
    uncertainty: Uncertainty
    model_metadata: dict[str, Any] = field(default_factory=dict)
    calibration_metadata: dict[str, Any] | None = None
    # Convenience: keep score matrix mean for derived markets if needed
    score_matrix_mean: Any = None  # np.ndarray
    btts_probabilities: dict[str, float] | None = None
    totals_probabilities: dict[str, dict[str, float]] | None = None  # line -> {over, push, under}
    corners_probabilities: dict[str, Any] | None = None
    correct_score_probabilities: dict[str, float] | None = None

    def __getitem__(self, key: str) -> float:
        # Allow dict-like access to probabilities for backward compat
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
                "anomaly_flags": self.uncertainty.anomaly_flags,
                "decision": self.uncertainty.decision,
            },
            "model_metadata": self.model_metadata,
        }


def predict_with_uncertainty(
    model,
    home_team: str,
    away_team: str,
    *,
    idata=None,  # Bayesian posterior if available
    market_odds: dict[str, float] | None = None,
    n_historical: int | None = None,
    is_promoted: bool = False,
) -> PredictionResult:
    """Build PredictionResult from a fitted TeamStrengthModel and optional posterior.

    If idata is provided, uses posterior predictive (§5) for uncertainty; otherwise
    uses point estimate with event randomness only.
    """
    from parlay.models.poisson import outcome_probabilities
    from parlay.prediction.markets import grouped_score_markets, correct_score_probabilities
    from parlay.evaluation.anomaly import diagnose
    import math as _math

    # Point estimate
    exp_h, exp_a = model.expected_goals(home_team, away_team)
    M = model.score_matrix(home_team, away_team)
    probs = outcome_probabilities(M)

    # Posterior uncertainty if available
    posterior_std_h = None
    posterior_std_a = None
    ci_h = (exp_h * 0.85, exp_h * 1.15)
    ci_a = (exp_a * 0.85, exp_a * 1.15)
    if idata is not None:
        try:
            from parlay.prediction.bayesian_predictive import posterior_predictive_1x2
            pp = posterior_predictive_1x2(idata, home_team, away_team, model.teams, model=model.model, rho=model.rho)
            probs = pp.mean_probability
            ci_h = pp.lambda_home_ci
            ci_a = pp.lambda_away_ci
            posterior_std_h = float((ci_h[1] - ci_h[0]) / 3.29)  # approx std from 90% CI
            posterior_std_a = float((ci_a[1] - ci_a[0]) / 3.29)
        except Exception:
            pass

    # Anomaly layer
    market_p = None
    if market_odds and "home" in market_odds and market_odds["home"] > 1:
        # Convert odds to implied then de-vig not available here, use raw implied as proxy
        market_p = 1.0 / market_odds["home"]
    diag = diagnose(
        model_probability=probs["home_win"],
        market_probability=market_p,
        posterior_std=posterior_std_h,
        n_historical=n_historical,
        is_promoted=is_promoted,
    )

    # Derived markets for completeness
    totals = {}
    for line in [1.5, 2.5, 2.75, 3.5]:
        from parlay.prediction.markets import totals_settlement_probabilities
        try:
            totals[str(line)] = totals_settlement_probabilities(M, line)
        except Exception:
            pass

    return PredictionResult(
        probabilities=probs,
        expected_rates=ExpectedRates(
            lambda_home_mean=exp_h, lambda_home_ci=ci_h,
            lambda_away_mean=exp_a, lambda_away_ci=ci_a,
            exp_total_goals_mean=exp_h + exp_a,
        ),
        uncertainty=Uncertainty(
            posterior_std_home=posterior_std_h,
            posterior_std_away=posterior_std_a,
            market_disagreement=diag.disagreement,
            anomaly_flags=diag.anomaly_flags,
            decision=diag.decision,
        ),
        model_metadata={"model": model.model, "rho": model.rho, "home_advantage": model.home_advantage},
        score_matrix_mean=M,
        btts_probabilities=grouped_score_markets(M),
        totals_probabilities=totals,
        correct_score_probabilities=correct_score_probabilities(M, max_home=4, max_away=4),
    )
