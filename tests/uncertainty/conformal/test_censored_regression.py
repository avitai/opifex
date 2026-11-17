"""Tests for clean-room right-censored and doubly-robust conformal regression.

Verifies the mathematical contracts in
``docs/clean_room_specs/censored_conformal_prediction.md``:

* Candès, Lei & Ren (2023, JRSS-B 85(1):24-45, arXiv:2103.09763) — cutoff filtration
  ``C_i >= c_0``, IPCW weights ``1 / G(c_0 | X_i)``, and monotonicity lifting to
  ``P(T >= L_hat(X)) >= 1 - alpha``.
* Tibshirani, Foygel Barber, Candès & Ramdas (2019, NeurIPS, Eq. 5) — weighted
  conformal quantile with test-point atom ``w_{n+1}(x)`` at ``+inf``.
* Sesia & Svetnik (2025, ICML, arXiv:2412.09729) — general right-censoring IPCW and
  doubly-robust intersection ``max(q_hat_{1-alpha}, 0)``.
"""

from __future__ import annotations

import dataclasses as dc
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from substrax.testing import TraceCounter

from opifex.uncertainty.conformal import (
    CensoredConformalState,
    CensoredSplitConformalRegressor,
    effective_sample_size,
    ipcw_conformal_quantile,
    minimum_calibration_size,
    signed_lower_residual_score,
)


class TestCensoredScoreAndQuantilePrimitives:
    """Unit invariants for signed lower scores, Kish ESS, and weighted quantiles."""

    def test_signed_lower_residual_score_truncates_targets_and_sanitizes_inf(self) -> None:
        preds = jnp.array([6.0, 8.0, 7.5])
        targets = jnp.array([5.0, 9.0, jnp.inf])
        scores = signed_lower_residual_score(predictions=preds, targets=targets, cutoff=7.0)
        # min(targets, 7.0) = [5.0, 7.0, 7.0] -> preds - min_t = [1.0, 1.0, 0.5]
        assert jnp.allclose(scores, jnp.array([1.0, 1.0, 0.5]), atol=1e-6)

        cqr_scores = signed_lower_residual_score(
            predictions=preds,
            targets=targets,
            cutoff=7.0,
            truncate_predictions=True,
        )
        # min(preds, 7.0) = [6.0, 7.0, 7.0] -> [1.0, 0.0, 0.0]
        assert jnp.allclose(cqr_scores, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)

        scaled = signed_lower_residual_score(
            predictions=preds,
            targets=targets,
            cutoff=7.0,
            scales=jnp.array([2.0, 0.5, 1.0]),
        )
        assert jnp.allclose(scaled, jnp.array([0.5, 2.0, 0.5]), atol=1e-6)

    def test_effective_sample_size_satisfies_kish_identities(self) -> None:
        equal_active = jnp.array([2.5, 2.5, 0.0, 2.5, 0.0, 2.5])
        assert float(effective_sample_size(equal_active)) == pytest.approx(4.0, rel=1e-6)
        assert float(effective_sample_size(7.3 * equal_active)) == pytest.approx(4.0, rel=1e-6)
        assert float(effective_sample_size(jnp.zeros(5))) == 0.0
        dominant = jnp.array([1e6, 1.0, 1.0])
        assert 1.0 <= float(effective_sample_size(dominant)) < 1.01

    def test_ipcw_quantile_matches_exact_unweighted_conformal_under_constant_survival(
        self,
    ) -> None:
        """Constant survival ``g_0`` cancels identically when ``w_{n+1} = 1 / g_0``."""
        active_scores = jnp.array([-0.8, -0.2, 0.1, 0.4, 0.7, 1.1, 1.5, 1.9, 2.3, 3.0])
        # Embed 10 active scores among 5 excluded calibration rows (weight 0, arbitrary scores)
        scores = jnp.concatenate([active_scores, jnp.array([-99.0, -50.0, 0.0, 42.0, 100.0])])
        g0 = 0.25
        weights = jnp.concatenate([jnp.full(10, 1.0 / g0), jnp.zeros(5)])

        m = int(active_scores.shape[0])
        for alpha in (0.1, 0.2, 0.35):
            q_ipcw = float(
                ipcw_conformal_quantile(
                    scores=scores,
                    calibration_weights=weights,
                    test_weight=1.0 / g0,
                    alpha=alpha,
                )
            )
            # Exact order statistic inf{s_(k) : k / (m + 1) >= 1 - alpha} = s_(ceil((m+1)(1-alpha)))
            rank_1based = math.ceil((m + 1) * (1.0 - alpha))
            q_exact = float(active_scores[rank_1based - 1])
            assert q_ipcw == pytest.approx(q_exact, abs=1e-6)

        # With only m=10 active points and alpha=0.05 (1/(10+1) > 0.05), quantile MUST be +inf
        q_unattainable = ipcw_conformal_quantile(
            scores=scores,
            calibration_weights=weights,
            test_weight=1.0 / g0,
            alpha=0.05,
        )
        assert bool(jnp.isposinf(q_unattainable))

    def test_ipcw_quantile_is_invariant_to_tie_permutations(self) -> None:
        scores = jnp.array([0.5, 0.5, -0.2, 1.0, 0.5, 1.0])
        weights = jnp.array([1.2, 0.8, 1.5, 1.0, 2.0, 0.5])
        ref = float(
            ipcw_conformal_quantile(
                scores=scores, calibration_weights=weights, test_weight=1.0, alpha=0.25
            )
        )
        rng = np.random.default_rng(123)
        for _ in range(10):
            perm = rng.permutation(scores.shape[0])
            val = float(
                ipcw_conformal_quantile(
                    scores=scores[perm],
                    calibration_weights=weights[perm],
                    test_weight=1.0,
                    alpha=0.25,
                )
            )
            assert val == pytest.approx(ref, abs=0.0)


class TestCensoredSplitConformalRegressorContracts:
    """State, refusal, warning, and JIT/vmap contracts."""

    def test_state_is_frozen_pytree_and_predict_before_fit_raises(self) -> None:
        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=8.0)
        with pytest.raises(RuntimeError, match=r"fit"):
            reg.predict(predictions=jnp.zeros(4))

        state = reg.fit(
            predictions=jnp.linspace(5.0, 7.0, 20),
            observed_targets=jnp.linspace(5.2, 7.2, 20),
            censoring_times=jnp.full(20, 8.5),
        )
        assert isinstance(state, CensoredConformalState)
        with pytest.raises(dc.FrozenInstanceError):
            state.cutoff = 9.0  # type: ignore[misc]

        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) == 2
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(rebuilt.calibration_scores, state.calibration_scores)

    def test_underfilled_effective_sample_size_warns_and_returns_unbounded_interval(self) -> None:
        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=8.0)
        # Only 4 points satisfy C_i >= 8.0 (< minimum_calibration_size(0.1) == 9)
        censoring_times = jnp.array([8.5] * 4 + [6.0] * 16)
        with pytest.warns(UserWarning, match=r"effective sample size"):
            state = reg.fit(
                predictions=jnp.full(20, 7.0),
                observed_targets=jnp.full(20, 6.0),
                censoring_times=censoring_times,
            )
        assert state.underfilled is True
        assert state.effective_n < minimum_calibration_size(alpha=0.1)
        interval = reg.with_state(state).predict(predictions=jnp.array([6.5, 7.5]))
        assert bool(jnp.all(jnp.isneginf(interval.lower)))
        assert bool(jnp.all(jnp.isposinf(interval.upper)))

    def test_predict_is_jit_and_vmap_single_compile_compatible(self) -> None:
        rng = np.random.default_rng(7)
        n_cal = 64
        preds_cal = jnp.asarray(rng.uniform(5.0, 8.0, size=n_cal), dtype=jnp.float32)
        c_cal = jnp.asarray(rng.uniform(7.0, 10.0, size=n_cal), dtype=jnp.float32)
        t_latent = preds_cal + jnp.asarray(rng.normal(0.0, 0.5, size=n_cal), dtype=jnp.float32)
        t_obs = jnp.minimum(t_latent, c_cal)
        g_cal = jnp.full(n_cal, 0.75, dtype=jnp.float32)

        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=7.5)
        state = reg.fit(
            predictions=preds_cal,
            observed_targets=t_obs,
            censoring_times=c_cal,
            censoring_survival=g_cal,
        )
        fitted = reg.with_state(state)

        counter = TraceCounter()
        jitted = jax.jit(
            counter.wrap(lambda p, g: fitted.predict(predictions=p, censoring_survival=g))
        )
        p_test = jnp.linspace(5.5, 8.5, 16, dtype=jnp.float32)
        g_test = jnp.linspace(0.5, 0.9, 16, dtype=jnp.float32)
        with counter.expect(new_traces=1):
            iv1 = jitted(p_test, g_test)
        with counter.expect(new_traces=0):
            iv2 = jitted(p_test + 0.2, g_test)

        eager = fitted.predict(predictions=p_test, censoring_survival=g_test)
        assert jnp.allclose(iv1.lower, eager.lower, atol=1e-6)
        assert bool(jnp.all(jnp.isposinf(iv1.upper)))
        assert bool(jnp.all(iv1.lower <= 7.5))
        assert bool(jnp.all(iv2.lower <= 7.5))


class TestRightCensoredMonteCarloCoverage:
    """500-draw Monte Carlo latent-target coverage studies on Weibull and log-normal data."""

    def test_weibull_covariate_dependent_censoring_achieves_nominal_latent_coverage(self) -> None:
        """Candès, Lei & Ren (2023) Theorem 1: 500 draws under covariate-dependent censoring."""
        alpha = 0.1
        cutoff = 2.0
        n_cal = 180
        n_test = 200
        num_draws = 500

        total_covered = 0
        total_points = 0

        for draw in range(num_draws):
            rng = np.random.default_rng(10_000 + draw)
            x_cal = rng.uniform(-1.0, 1.0, size=n_cal)
            x_test = rng.uniform(-1.0, 1.0, size=n_test)

            # Latent Weibull outcome T | X ~ Weibull(shape=2.0, scale=1.8 + 0.6 * X)
            scale_cal = 1.8 + 0.6 * x_cal
            scale_test = 1.8 + 0.6 * x_test
            t_cal = scale_cal * rng.weibull(2.0, size=n_cal)
            t_test = scale_test * rng.weibull(2.0, size=n_test)

            # Covariate-dependent exponential censoring C | X ~ Exp(rate = 0.20 + 0.15*(X + 1))
            rate_cal = 0.20 + 0.15 * (x_cal + 1.0)
            rate_test = 0.20 + 0.15 * (x_test + 1.0)
            c_cal = rng.exponential(scale=1.0 / rate_cal, size=n_cal)
            g_cal = np.exp(-rate_cal * cutoff)
            g_test = np.exp(-rate_test * cutoff)

            t_obs_cal = np.minimum(t_cal, c_cal)
            # Imperfect base predictor (deliberately misspecified scale)
            pred_cal = 1.5 + 0.4 * x_cal
            pred_test = 1.5 + 0.4 * x_test

            reg = CensoredSplitConformalRegressor(
                alpha=alpha, cutoff=cutoff, score_type="signed_residual"
            )
            state = reg.fit(
                predictions=jnp.asarray(pred_cal, dtype=jnp.float32),
                observed_targets=jnp.asarray(t_obs_cal, dtype=jnp.float32),
                censoring_times=jnp.asarray(c_cal, dtype=jnp.float32),
                censoring_survival=jnp.asarray(g_cal, dtype=jnp.float32),
            )
            interval = reg.with_state(state).predict(
                predictions=jnp.asarray(pred_test, dtype=jnp.float32),
                censoring_survival=jnp.asarray(g_test, dtype=jnp.float32),
            )
            lower_np = np.asarray(interval.lower)
            total_covered += int(np.sum(t_test >= lower_np))
            total_points += n_test

        empirical_coverage = total_covered / total_points
        binomial_se = math.sqrt(alpha * (1.0 - alpha) / num_draws)
        assert empirical_coverage >= (1.0 - alpha) - 2.5 * binomial_se, empirical_coverage

    def test_lognormal_general_right_censoring_and_doubly_robust_quantile_coverage(
        self,
    ) -> None:
        """Sesia & Svetnik (2025): 500 draws with unobserved C_i on events + DR robustness."""
        alpha = 0.1
        cutoff = 7.5
        n_cal = 200
        n_test = 200
        num_draws = 500

        km_covered = 0
        dr_misspecified_g_covered = 0
        total_points = 0
        z_alpha = float(stats.norm.ppf(alpha))

        for draw in range(num_draws):
            rng = np.random.default_rng(50_000 + draw)
            x_cal = rng.uniform(0.0, 1.0, size=n_cal)
            x_test = rng.uniform(0.0, 1.0, size=n_test)

            # Log-potency / log-survival: T | X ~ LogNormal(mu = 1.8 + 0.3 * X, sigma = 0.25)
            # Mapped to potency-like scale via log(T) ~ Normal(6.5 + 1.0 * X, 0.5^2)
            mu_cal = 6.5 + 1.0 * x_cal
            mu_test = 6.5 + 1.0 * x_test
            t_cal = mu_cal + 0.5 * rng.standard_normal(size=n_cal)
            t_test = mu_test + 0.5 * rng.standard_normal(size=n_test)

            # Random right-censoring threshold C ~ Uniform(6.2, 9.2), unobserved when T <= C
            c_cal = rng.uniform(6.2, 9.2, size=n_cal)
            t_obs = np.minimum(t_cal, c_cal)
            events = (t_cal <= c_cal).astype(np.bool_)

            # 1) Regime II Kaplan-Meier IPCW with imperfect point predictor
            reg_km = CensoredSplitConformalRegressor(alpha=alpha, cutoff=cutoff)
            state_km = reg_km.fit(
                predictions=jnp.asarray(mu_cal + 0.3, dtype=jnp.float32),
                observed_targets=jnp.asarray(t_obs, dtype=jnp.float32),
                event_indicators=jnp.asarray(events),
            )
            iv_km = reg_km.with_state(state_km).predict(
                predictions=jnp.asarray(mu_test + 0.3, dtype=jnp.float32)
            )
            km_covered += int(np.sum(t_test >= np.asarray(iv_km.lower)))

            # 2) Doubly-robust intersection under severely misspecified censoring weights
            #    when conditional alpha-quantile predictor q_alpha(X) is consistent
            q_alpha_cal = mu_cal + 0.5 * z_alpha
            q_alpha_test = mu_test + 0.5 * z_alpha
            bad_g_cal = np.where(x_cal > 0.5, 0.95, 0.20)
            bad_g_test = np.where(x_test > 0.5, 0.95, 0.20)
            reg_dr = CensoredSplitConformalRegressor(
                alpha=alpha,
                cutoff=cutoff,
                score_type="cqr_lower",
                doubly_robust=True,
            )
            state_dr = reg_dr.fit(
                predictions=jnp.asarray(q_alpha_cal, dtype=jnp.float32),
                observed_targets=jnp.asarray(t_obs, dtype=jnp.float32),
                event_indicators=jnp.asarray(events),
                censoring_survival=jnp.asarray(bad_g_cal, dtype=jnp.float32),
            )
            iv_dr = reg_dr.with_state(state_dr).predict(
                predictions=jnp.asarray(q_alpha_test, dtype=jnp.float32),
                censoring_survival=jnp.asarray(bad_g_test, dtype=jnp.float32),
            )
            dr_misspecified_g_covered += int(np.sum(t_test >= np.asarray(iv_dr.lower)))
            total_points += n_test

        binomial_se = math.sqrt(alpha * (1.0 - alpha) / num_draws)
        assert (km_covered / total_points) >= (1.0 - alpha) - 2.5 * binomial_se
        assert (dr_misspecified_g_covered / total_points) >= (1.0 - alpha) - 2.5 * binomial_se
