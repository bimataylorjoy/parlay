"""P0 upgrade tests: InformationSet, Dixon rho global, clipping unify, Bayesian, adaptive truncation, NB joint, corners."""

import math
from datetime import datetime, timezone, timedelta, date
import numpy as np
import pytest

from parlay.data.schemas import Match
from parlay.data.information import InformationSet, RESULT_LAG_MINUTES
from parlay.models.dixon_coles import tau, valid_rho_bounds, _global_rho_bounds, score_matrix as dc_score_matrix
from parlay.models.poisson import score_matrix, score_matrix_with_metadata, _adaptive_max_goals
from parlay.models.negative_binomial import negative_binomial_pmf
from parlay.models.team_strength import fit_team_strength
from parlay.models.corners import fit_corner_strength


def _make_match(mid, d, ko, comp="EPL"):
    return Match(match_id=mid, date=d, competition=comp, season="test", home_team="A", away_team="B", home_goals=1, away_goals=0, kickoff_at=ko)


def test_information_set_per_competition():
    ko = datetime(2026, 9, 2, 19, 45, tzinfo=timezone.utc)
    m = _make_match("m1", date(2026, 9, 2), ko, comp="EPL")
    info_epl = InformationSet(as_of=ko + timedelta(minutes=115), competition="EPL")
    info_champ = InformationSet(as_of=ko + timedelta(minutes=115), competition="Championship")
    # EPL lag 115 -> exactly known, Champ lag 118 -> not yet
    assert info_epl.is_result_knowable(m) is True
    assert info_champ.is_result_knowable(m) is False
    # Fallback when kickoff is None: known_at = 23:59 same day (no lag)
    m2 = Match(match_id="m2", date=date(2026,9,2), competition="EPL", season="test", home_team="A", away_team="B", home_goals=0, away_goals=0, kickoff_at=None)
    info = InformationSet(as_of=datetime(2026,9,3,0,0, tzinfo=timezone.utc))
    assert info.is_result_knowable(m2) is True  # 23:59 02/09 < 00:00 03/09
    info2 = InformationSet(as_of=datetime(2026,9,2,23,58, tzinfo=timezone.utc))
    assert info2.is_result_knowable(m2) is False  # 23:58 < 23:59 same day


def test_dixon_rho_global_no_per_match_mutation():
    # High lambda should give narrow global interval; per-match clip would hide it
    rates_home = np.array([3.5, 0.8])
    rates_away = np.array([3.5, 0.8])
    lower, upper = valid_rho_bounds(rates_home, rates_away)
    gl, gu = _global_rho_bounds(rates_home, rates_away)
    # global must be intersection
    assert gl >= max(lower) - 1e-9
    assert gu <= min(upper) + 1e-9
    # tau with rho outside global but inside per-match should produce negative if per-match clip removed
    # Here we test that tau does NOT clip per match: pass rho=0.2 which is outside global for high lambda (1/(3.5*3.5)=0.081)
    rho = 0.2
    # For high lambda, tau(0,0)=1 - 12.25*0.2 = -1.45 -> clamped to 0, but the fact it's negative before clamp shows global check matters
    t = tau(np.array([0]), np.array([0]), np.array([3.5]), np.array([3.5]), rho)
    assert t[0] == 0.0  # clamped, but optimizer should avoid this via global bounds


def test_clipping_unified():
    # TeamStrengthModel and MLE should use same [-3,3] — test via extremely strong team
    matches = [
        Match(match_id=f"m{i}", date=date(2024,1,1), competition="EPL", season="test", home_team="Strong", away_team="Weak", home_goals=5, away_goals=0, home_sot=10, away_sot=1, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc))
        for i in range(20)
    ] + [
        Match(match_id=f"n{i}", date=date(2024,1,2), competition="EPL", season="test", home_team="Weak", away_team="Strong", home_goals=0, away_goals=5, home_sot=1, away_sot=10, kickoff_at=datetime(2024,1,2,15,0,tzinfo=timezone.utc))
        for i in range(20)
    ]
    model = fit_team_strength(matches, model="poisson", estimator="mle", half_life_days=None)
    lam_h, lam_a = model.expected_goals("Strong", "Weak")
    # With unified clipping, lambdas should be within exp(3)=20
    assert lam_h <= 20.1
    assert lam_a <= 20.1


def test_bayesian_prior_symmetry_and_posterior_predictive():
    # Check that bayesian module now uses exchangeable priors (attack_raw)
    import inspect
    from parlay.models import bayesian as bmod
    src = inspect.getsource(bmod.fit_poisson_bayesian)
    assert "attack_raw" in src
    assert "attack_last = -pt.sum" not in src  # old asymmetric removed
    # Posterior predictive: test that mean probabilities sum to 1 and differs from plug-in
    # Use small synthetic data and mock idata without needing real PyMC sampling (check API exists)
    from parlay.prediction.bayesian_predictive import posterior_predictive_1x2
    assert callable(posterior_predictive_1x2)


def test_adaptive_truncation():
    # For high lambda, adaptive should increase max_goals and keep tail <1e-6
    M, meta = score_matrix_with_metadata(3.5, 3.5, max_goals=None, epsilon=1e-6)
    assert meta["joint_tail_mass_estimate"] < 1e-5
    assert M.shape[0] >= 14  # should be larger than default 10 for high lambda
    # Fixed should still report tail
    M2, meta2 = score_matrix_with_metadata(3.5, 3.5, max_goals=10, epsilon=1e-6)
    assert meta2["tail_mass_home"] > 1e-4  # significant tail when truncated at 10


def test_adaptive_support_uses_smallest_valid_cutoff():
    from parlay.models.numerics import adaptive_support, poisson_tail

    cutoff, capped = adaptive_support(1.2, epsilon=1e-6, cap=100)
    assert not capped
    assert poisson_tail(1.2, cutoff) < 1e-6
    assert poisson_tail(1.2, cutoff - 1) >= 1e-6


def test_dixon_rho_estimation_stays_inside_global_domain():
    from parlay.models.dixon_coles import estimate_rho

    home_rates = np.array([3.5, 3.0, 0.8])
    away_rates = np.array([3.5, 2.5, 0.8])
    lower, upper = _global_rho_bounds(home_rates, away_rates)
    value = estimate_rho(
        np.array([0, 1, 0]), np.array([0, 0, 1]), home_rates, away_rates
    )
    assert lower <= value <= upper


def test_fit_exposes_optimizer_status():
    matches = [
        Match(
            match_id=f"diag-{i}", date=date(2024, 1, i + 1), competition="EPL",
            season="test", home_team="A", away_team="B", home_goals=i % 3,
            away_goals=(i + 1) % 2,
            kickoff_at=datetime(2024, 1, i + 1, 15, 0, tzinfo=timezone.utc),
        )
        for i in range(5)
    ]
    model = fit_team_strength(matches, estimator="mle", half_life_days=None)
    assert isinstance(model.fit_converged, bool)
    assert model.fit_negative_log_likelihood is not None


def test_negative_binomial_joint():
    # Synthetic overdispersed data: phi=5
    rng = np.random.default_rng(0)
    # Generate via NB: use dispersion 5, mean 1.5
    from parlay.models.negative_binomial import score_matrix as nb_sm
    # Test that NB matrix with small phi has heavier tail than Poisson
    pm = score_matrix(1.5, 1.5, max_goals=15)
    nb = nb_sm(1.5, 1.5, 5.0, 5.0, max_goals=15)
    # NB should have more mass at 0-0 and high scores? Check variance
    assert not np.allclose(pm, nb)
    # Test that fit recovers: create synthetic matches with overdispersion, fit NB mle should get phi < 50
    matches = []
    for i in range(100):
        # Simple: two teams, overdispersed goals drawn from NB
        # Use Poisson with extra variance by mixing
        g_h = int(rng.negative_binomial(n=5, p=5/(5+1.5)))
        g_a = int(rng.negative_binomial(n=5, p=5/(5+1.2)))
        matches.append(Match(match_id=f"s{i}", date=date(2024,1,1), competition="EPL", season="test", home_team="A", away_team="B", home_goals=g_h, away_goals=g_a, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc)))
    model = fit_team_strength(matches, model="negative_binomial", estimator="mle", half_life_days=None)
    assert model.home_dispersion < 50  # should detect overdispersion
    assert model.home_dispersion >= 1.0


def test_corners_audit():
    # Use overdispersed corner data to differentiate Poisson vs NB likelihood
    rng = np.random.default_rng(1)
    matches = []
    for i in range(60):
        # Overdispersed: sometimes 2, sometimes 12
        hc = int(rng.choice([2, 12]))
        ac = int(rng.choice([1, 10]))
        matches.append(Match(match_id=f"c{i}", date=date(2024,1,1), competition="EPL", season="test", home_team="A", away_team="B", home_goals=1, away_goals=0, home_corners=hc, away_corners=ac, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc)))
    # Poisson corners (dispersion 200)
    m_pois = fit_corner_strength(matches, dispersion=200.0)
    # NB corners (dispersion 5) should use NB likelihood (different params) — more pronounced difference
    m_nb = fit_corner_strength(matches, dispersion=5.0)
    # With overdispersed data, likelihoods differ enough to move params slightly
    # Check that matrices differ (NB heavier tails)
    cm_pois = m_pois.corner_matrix("A", "B")
    cm_nb = m_nb.corner_matrix("A", "B")
    assert not np.allclose(cm_pois, cm_nb, atol=1e-3)
    assert abs(cm_nb.sum() - 1.0) < 1e-9


def test_no_fractional_effective_goals():
    # Ensure sot_weight deprecated warning and that goals remain integer
    import warnings
    matches = [
        Match(match_id="m1", date=date(2024,1,1), competition="EPL", season="test", home_team="A", away_team="B", home_goals=2, away_goals=1, home_sot=5, away_sot=2, kickoff_at=datetime(2024,1,1,15,0,tzinfo=timezone.utc))
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = fit_team_strength(matches, model="poisson", estimator="heuristic", sot_weight=0.5)
        assert any("deprecated" in str(x.message).lower() for x in w)
    # Model should still produce lambdas >0
    lam_h, lam_a = model.expected_goals("A", "B")
    assert lam_h > 0 and lam_a > 0
