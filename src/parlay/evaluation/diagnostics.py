"""Statistical diagnostics for fitted models and their residuals."""

from __future__ import annotations

from typing import Any

import numpy as np

from parlay.models.mle import RateFit


def frequentist_diagnostics(fit: RateFit) -> dict[str, Any]:
    """Optimizer status of an MLE fit."""
    return {
        "converged": fit.converged,
        "nll": fit.negative_log_likelihood,
        "iterations": fit.iterations,
        "warning": None if fit.converged else f"optimizer_failed: {fit.message}",
    }


def bayesian_diagnostics(idata: Any) -> dict[str, Any]:
    """R-hat, effective sample size and divergences of a PyMC posterior (needs ``arviz``)."""
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover - optional dependency
        return {"error": f"arviz unavailable: {exc}"}
    rhat_max = float(az.rhat(idata).to_array().values.max())
    ess_min = int(az.ess(idata).to_array().values.min())
    divergences = int(idata.sample_stats["diverging"].values.sum()) if "sample_stats" in idata else 0
    return {
        "r_hat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
        "ok": rhat_max < 1.01 and divergences == 0,
    }


def distribution_diagnostics(observed: list[float], expected: list[float]) -> dict[str, float]:
    """Compare observed counts with fitted means; ``dispersion_index`` > 1 means over-dispersion.

    ``dispersion_index`` is the Pearson statistic ``mean((y - mu)^2 / mu)``,
    which is 1 in expectation under a correctly specified Poisson model.
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.maximum(np.asarray(expected, dtype=float), 1e-9)
    return {
        "mean_observed": float(obs.mean()),
        "mean_expected": float(exp.mean()),
        "var_observed": float(obs.var()),
        "dispersion_index": float(np.mean((obs - exp) ** 2 / exp)),
    }
