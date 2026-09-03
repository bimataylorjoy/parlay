"""Statistical diagnostics (§19)."""

from typing import Any


def frequentist_diagnostics(mle_result: tuple) -> dict[str, Any]:
    # mle_result = (att, def, mu, gamma, rho, converged, nll)
    _, _, _, _, _, converged, nll = mle_result
    return {"converged": bool(converged), "nll": float(nll), "warning": None if converged else "optimizer_failed"}


def bayesian_diagnostics(idata) -> dict[str, Any]:
    try:
        import arviz as az
        rhat = az.rhat(idata)
        ess = az.ess(idata)
        # max rhat
        rhat_max = float(rhat.to_array().values.max())
        ess_min = int(ess.to_array().values.min())
        divs = 0
        if "sample_stats" in idata:
            divs = int((idata.sample_stats["diverging"].values).sum())
        return {"r_hat_max": rhat_max, "ess_min": ess_min, "divergences": divs, "ok": rhat_max < 1.01 and divs == 0}
    except Exception as e:
        return {"error": str(e)}


def distribution_diagnostics(observed: list[float], expected: list[float]) -> dict[str, float]:
    import numpy as np
    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    resid = obs - exp
    return {
        "mean_observed": float(np.mean(obs)),
        "mean_expected": float(np.mean(exp)),
        "var_observed": float(np.var(obs)),
        "var_expected": float(np.var(exp)),
        "overdispersion": float(np.var(obs) / (np.mean(exp) + 1e-9)),
    }
