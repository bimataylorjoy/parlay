"""Treat large model-market disagreement as an anomaly to investigate, not an automatic edge.

Liquid closing markets are hard to beat; a model that disagrees strongly with
them is more often missing information (injuries, lineups, promoted-team
uncertainty) than right. ``diagnose`` turns disagreement plus data-quality
signals into a research decision: ``RESEARCH_SIGNAL``, ``WATCH``, ``PASS``
or ``REJECT``.
"""

from __future__ import annotations

from dataclasses import dataclass

EXTREME_DISAGREEMENT = 0.25
SIGNAL_DISAGREEMENT = 0.15
WATCH_DISAGREEMENT = 0.08


@dataclass(frozen=True, slots=True)
class DecisionDiagnostics:
    model_probability: float
    market_probability: float | None
    disagreement: float | None
    uncertainty: str  # low / medium / high
    calibration_regime: str
    anomaly_flags: tuple[str, ...]
    decision: str  # RESEARCH_SIGNAL / WATCH / PASS / REJECT


def diagnose(
    model_probability: float,
    market_probability: float | None,
    *,
    posterior_std: float | None = None,
    calibration_error: float | None = None,
    n_historical: int | None = None,
    is_promoted: bool = False,
    lambda_total: float | None = None,
    data_missing: bool = False,
) -> DecisionDiagnostics:
    flags: list[str] = []
    disagreement = abs(model_probability - market_probability) if market_probability is not None else None
    if disagreement is not None and disagreement > EXTREME_DISAGREEMENT:
        flags.append("extreme_model_market_disagreement")
    if posterior_std is not None and posterior_std > 0.15:
        flags.append("wide_posterior_uncertainty")
    if calibration_error is not None and calibration_error > 0.1:
        flags.append("outside_historical_calibration_regime")
    if n_historical is not None and n_historical < 10:
        flags.append("insufficient_historical_sample")
    if is_promoted:
        flags.append("promoted_new_team_uncertainty")
    if lambda_total is not None and (lambda_total < 0.5 or lambda_total > 5.0):
        flags.append("extreme_lambda")
    if data_missing:
        flags.append("data_missingness")

    # Uncertainty level
    if "wide_posterior_uncertainty" in flags or "insufficient_historical_sample" in flags:
        uncertainty = "high"
    elif flags:
        uncertainty = "medium"
    else:
        uncertainty = "low"

    # Decision
    if "extreme_model_market_disagreement" in flags and uncertainty == "high":
        decision = "REJECT"
    elif disagreement is not None and disagreement > SIGNAL_DISAGREEMENT and uncertainty != "high":
        decision = "RESEARCH_SIGNAL"
    elif disagreement is not None and disagreement > WATCH_DISAGREEMENT:
        decision = "WATCH"
    else:
        decision = "PASS"

    return DecisionDiagnostics(
        model_probability=model_probability,
        market_probability=market_probability,
        disagreement=disagreement,
        uncertainty=uncertainty,
        calibration_regime="poor" if calibration_error and calibration_error > 0.1 else "ok",
        anomaly_flags=tuple(flags),
        decision=decision,
    )
