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

    def test_regime2_km_default_test_survival_uses_km_at_cutoff_not_unity(self) -> None:
        """Regime II KM calibration sets ``default_test_survival`` to ``G_hat(c_0)``, not ``1.0``."""
        rng = np.random.default_rng(42)
        n_cal = 160
        cutoff = 7.5
        t_latent = rng.uniform(6.0, 10.0, size=n_cal)
        c_latent = rng.uniform(5.5, 9.0, size=n_cal)
        t_obs = np.minimum(t_latent, c_latent)
        events = t_latent <= c_latent
        preds_cal = jnp.full(n_cal, 8.0, dtype=jnp.float32)

        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=cutoff)
        state = reg.fit(
            predictions=preds_cal,
            observed_targets=jnp.asarray(t_obs, dtype=jnp.float32),
            event_indicators=jnp.asarray(events),
        )
        assert state.requires_test_censoring_survival is False
        assert 0.0 < state.default_test_survival < 0.85

        fitted = reg.with_state(state)
        preds_test = jnp.array([7.8, 8.2], dtype=jnp.float32)
        iv_default = fitted.predict(predictions=preds_test)
        iv_explicit_km = fitted.predict(
            predictions=preds_test,
            censoring_survival=jnp.full_like(preds_test, state.default_test_survival),
        )
        iv_unity_surv = fitted.predict(
            predictions=preds_test,
            censoring_survival=jnp.ones_like(preds_test),
        )
        assert jnp.allclose(iv_default.lower, iv_explicit_km.lower, atol=1e-6)
        assert bool(jnp.all(iv_default.lower <= iv_unity_surv.lower))

    def test_predict_requires_censoring_survival_when_fit_supplied_censoring_survival(
        self,
    ) -> None:
        """When ``fit`` uses covariate-dependent ``censoring_survival``, ``predict`` requires it."""
        preds_cal = jnp.linspace(5.0, 8.0, 30, dtype=jnp.float32)
        obs_cal = jnp.linspace(5.2, 8.2, 30, dtype=jnp.float32)
        g_cal = jnp.linspace(0.5, 0.9, 30, dtype=jnp.float32)
        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=7.5)

        state_r1 = reg.fit(
            predictions=preds_cal,
            observed_targets=obs_cal,
            censoring_times=jnp.full(30, 8.0, dtype=jnp.float32),
            censoring_survival=g_cal,
        )
        assert state_r1.requires_test_censoring_survival is True
        with pytest.raises(ValueError, match=r"censoring_survival"):
            reg.with_state(state_r1).predict(predictions=jnp.array([6.5, 7.0]))

        state_r2 = reg.fit(
            predictions=preds_cal,
            observed_targets=obs_cal,
            event_indicators=jnp.ones(30, dtype=jnp.bool_),
            censoring_survival=g_cal,
        )
        assert state_r2.requires_test_censoring_survival is True
        with pytest.raises(ValueError, match=r"censoring_survival"):
            reg.with_state(state_r2).predict(predictions=jnp.array([6.5, 7.0]))

        valid_iv = reg.with_state(state_r2).predict(
            predictions=jnp.array([6.5, 7.0], dtype=jnp.float32),
            censoring_survival=jnp.array([0.7, 0.8], dtype=jnp.float32),
        )
        assert valid_iv.lower.shape == (2,)

    def test_censored_conformal_preserves_float64_dtype(self) -> None:
        """Float64 inputs preserve float64 precision when ``jax_enable_x64`` is active."""
        with jax.enable_x64(True):
            preds_cal = jnp.linspace(5.0, 8.0, 24, dtype=jnp.float64)
            obs_cal = preds_cal - jnp.asarray(0.25, dtype=jnp.float64)
            reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=7.5)
            state = reg.fit(
                predictions=preds_cal,
                observed_targets=obs_cal,
                censoring_times=jnp.full(24, 8.5, dtype=jnp.float64),
            )
            assert state.calibration_scores.dtype == jnp.float64
            assert state.calibration_weights.dtype == jnp.float64
            iv = reg.with_state(state).predict(predictions=jnp.array([6.5, 7.2], dtype=jnp.float64))
            assert iv.lower.dtype == jnp.float64
            assert iv.upper.dtype == jnp.float64

    def test_ipcw_quantile_and_censored_validation_ulps_and_nan(self) -> None:
        """IPCW quantile validates weights, avoids float32 ULP +inf shortfalls, and propagates NaN."""
        with pytest.raises(ValueError, match="non-negative"):
            ipcw_conformal_quantile(
                scores=jnp.array([0.1, 1.0]),
                calibration_weights=jnp.array([-1.0, 1.0]),
                test_weight=1.0,
                alpha=0.1,
            )
        with pytest.raises(ValueError, match="non-negative"):
            ipcw_conformal_quantile(
                scores=jnp.array([0.1, 1.0]),
                calibration_weights=jnp.array([1.0, 1.0]),
                test_weight=-0.5,
                alpha=0.1,
            )
        with pytest.raises(ValueError, match="scalar"):
            ipcw_conformal_quantile(
                scores=jnp.array([0.1, 1.0]),
                calibration_weights=jnp.array([1.0, 1.0]),
                test_weight=jnp.array([1.0, 1.0]),
                alpha=0.1,
            )

        # Float32 prefix/tail complement avoids cumsum/sum ULP shortfall at test_weight=0.0
        scores_41 = jnp.arange(1.0, 42.0, dtype=jnp.float32)
        q_zero_test = ipcw_conformal_quantile(
            scores=scores_41,
            calibration_weights=jnp.ones(41, dtype=jnp.float32),
            test_weight=jnp.float32(0.0),
            alpha=1e-7,
        )
        assert float(q_zero_test) == pytest.approx(41.0, abs=1e-5)

        # IEEE-754 float(1/3) = 0.3333333333333333 < 1/3, so minimum_calibration_size(1/3) == 3:
        # n == 2 returns +inf, whereas n == 3 with equal w = 0.3 in float32 returns scores[2] = 3.0.
        assert minimum_calibration_size(alpha=1.0 / 3.0) == 3
        q_under_1_3 = ipcw_conformal_quantile(
            scores=jnp.array([1.0, 2.0], dtype=jnp.float32),
            calibration_weights=jnp.array([0.3, 0.3], dtype=jnp.float32),
            test_weight=jnp.float32(0.3),
            alpha=1.0 / 3.0,
        )
        assert bool(jnp.isposinf(q_under_1_3))
        q_boundary = ipcw_conformal_quantile(
            scores=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
            calibration_weights=jnp.array([0.3, 0.3, 0.3], dtype=jnp.float32),
            test_weight=jnp.float32(0.3),
            alpha=1.0 / 3.0,
        )
        assert float(q_boundary) == pytest.approx(3.0, abs=1e-6)

        # NaN propagation in scores, calibration_weights, and test_weight
        assert bool(
            jnp.isnan(
                ipcw_conformal_quantile(
                    scores=jnp.array([jnp.nan, 1.0]),
                    calibration_weights=jnp.ones(2),
                    test_weight=1.0,
                    alpha=0.2,
                )
            )
        )
        assert bool(
            jnp.isnan(
                ipcw_conformal_quantile(
                    scores=jnp.array([0.5, 1.0]),
                    calibration_weights=jnp.array([jnp.nan, 1.0]),
                    test_weight=1.0,
                    alpha=0.2,
                )
            )
        )
        assert bool(
            jnp.isnan(
                ipcw_conformal_quantile(
                    scores=jnp.array([0.5, 1.0]),
                    calibration_weights=jnp.ones(2),
                    test_weight=jnp.nan,
                    alpha=0.2,
                )
            )
        )

        # CensoredSplitConformalRegressor shape and probability-bound validation
        reg = CensoredSplitConformalRegressor(alpha=0.1, cutoff=7.5)
        preds_cal = jnp.linspace(5.0, 7.0, 12, dtype=jnp.float32)
        obs_cal = jnp.linspace(5.1, 7.1, 12, dtype=jnp.float32)
        c_cal = jnp.full(12, 8.0, dtype=jnp.float32)
        for bad_surv in (
            jnp.ones((1, 12), dtype=jnp.float32),
            jnp.full(12, -0.2, dtype=jnp.float32),
            jnp.full(12, 1.5, dtype=jnp.float32),
        ):
            with pytest.raises(ValueError, match="censoring_survival"):
                reg.fit(
                    predictions=preds_cal,
                    observed_targets=obs_cal,
                    censoring_times=c_cal,
                    censoring_survival=bad_surv,
                )
        fitted = reg.with_state(
            reg.fit(predictions=preds_cal, observed_targets=obs_cal, censoring_times=c_cal)
        )
        with pytest.raises(ValueError, match="1-D"):
            fitted.predict(predictions=jnp.zeros((2, 1), dtype=jnp.float32))


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

        # 3.5 * SE has two-sided normal false-failure probability 2 * Phi(-3.5) = 4.65e-4.
        binomial_se = math.sqrt(alpha * (1.0 - alpha) / num_draws)
        assert (km_covered / total_points) >= (1.0 - alpha) - 3.5 * binomial_se
        assert (dr_misspecified_g_covered / total_points) >= (1.0 - alpha) - 3.5 * binomial_se

    def test_tied_event_and_censoring_times_km_survival_and_coverage(self) -> None:
        """Reverse-KM retains the full risk set ``observed_times >= u_k`` at ties
        (matching ``survival::survfit(Surv(time, 1 - status))``) and returns exact ``0.0``."""
        from calibrax.metrics.functional import reverse_kaplan_meier

        def _km_cens(obs: np.ndarray, cens_ev: np.ndarray, eval_q: np.ndarray) -> np.ndarray:
            with jax.enable_x64(True):
                return np.asarray(
                    reverse_kaplan_meier(
                        jnp.asarray(obs, dtype=jnp.float64),
                        jnp.asarray(~cens_ev),
                        eval_times=jnp.asarray(eval_q, dtype=jnp.float64),
                        left_continuous=True,
                    )
                )

        # Single-tie fixture: 3 observations at t=2.0 (1 censored, 2 uncensored events).
        # Full risk set at u=2.0 is 3, drops=1 -> G(2.0) = 1.0, G(2.5) = 2/3.
        obs_tie = np.array([2.0, 2.0, 2.0])
        cens_tie = np.array([True, False, False])
        g_tie = _km_cens(obs_tie, cens_tie, np.array([2.0, 2.5]))
        assert float(g_tie[0]) == pytest.approx(1.0, abs=1e-12)
        assert float(g_tie[1]) == pytest.approx(2.0 / 3.0, abs=1e-12)

        # Multi-tie closed-form parity with survival::survfit(Surv(time, 1 - status)):
        # u_1 = 2.0: at_risk = 6, drops = 1 -> factor 5/6
        # u_2 = 4.0: at_risk = 3, drops = 2 -> factor 1/3 -> cumulative 5/18
        obs_multi = np.array([1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 5.0])
        cens_multi = np.array([False, True, False, False, True, True, False])
        eval_t = np.array([0.5, 2.0, 3.0, 4.0, 4.5, 6.0])
        g_multi = _km_cens(obs_multi, cens_multi, eval_t)
        expected_multi = np.array([1.0, 1.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 18.0, 5.0 / 18.0])
        np.testing.assert_allclose(g_multi, expected_multi, rtol=1e-12, atol=1e-12)

        # Terminal censoring of all remaining at-risk observations yields exact 0.0 (not tiny).
        obs_zero = np.array([1.0, 3.0, 3.0])
        cens_zero = np.array([False, True, True])
        g_zero = _km_cens(obs_zero, cens_zero, np.array([3.0, 3.5]))
        assert float(g_zero[0]) == 1.0
        assert float(g_zero[1]) == 0.0

        # Discrete censoring C in {5.0 (p=0.8), 10.0 (p=0.2)} under tie convention 1{T < C}
        # where censorings precede tied failures: true P(C >= 6.0) = 0.20.
        rng_km = np.random.default_rng(2026)
        n_km = 4000
        t_km = rng_km.choice([5.0, 7.0], size=n_km, p=[0.75, 0.25])
        c_km = rng_km.choice([5.0, 10.0], size=n_km, p=[0.80, 0.20])
        obs_km = np.minimum(t_km, c_km)
        cens_ev = c_km <= t_km
        est_g = float(_km_cens(obs_km, cens_ev, np.array([6.0]))[0])
        # Since min(T, C) >= 5.0 for all n_km=4000 draws and C == 5.0 with p = 0.80,
        # est_g = 1 - #{C == 5} / n_km is Binomial(n_km, 0.20) / n_km with SE = sqrt(0.16 / n_km);
        # 3.5 * SE has two-sided normal false-failure probability 2 * Phi(-3.5) = 4.65e-4.
        se_km = math.sqrt(0.20 * 0.80 / n_km)
        assert abs(est_g - 0.20) <= 3.5 * se_km

        with pytest.raises(TypeError, match="static_argnames"):
            jax.jit(
                lambda a: CensoredSplitConformalRegressor(alpha=a, cutoff=7.0).fit(  # type: ignore[arg-type]
                    predictions=jnp.ones(12, dtype=jnp.float32),
                    observed_targets=jnp.ones(12, dtype=jnp.float32),
                    event_indicators=jnp.ones(12, dtype=jnp.bool_),
                )
            )(jnp.float32(0.1))

    def test_exponential_general_right_censoring_km_coverage(self) -> None:
        """250-draw exponential censoring coverage under reverse-KM IPCW."""
        alpha = 0.1
        cutoff = 7.0
        n_cal = 240
        n_test = 200
        num_draws = 250
        covered_total = 0
        points_total = 0

        for draw in range(num_draws):
            rng = np.random.default_rng(90_000 + draw)
            t_cal = rng.exponential(scale=8.0, size=n_cal)
            c_cal = rng.exponential(scale=10.0, size=n_cal)
            t_test = rng.exponential(scale=8.0, size=n_test)

            obs_cal = np.minimum(t_cal, c_cal)
            events_cal = t_cal <= c_cal
            reg = CensoredSplitConformalRegressor(alpha=alpha, cutoff=cutoff)
            state = reg.fit(
                predictions=jnp.full(n_cal, 5.0, dtype=jnp.float32),
                observed_targets=jnp.asarray(obs_cal, dtype=jnp.float32),
                event_indicators=jnp.asarray(events_cal),
            )
            iv = reg.with_state(state).predict(predictions=jnp.full(n_test, 5.0, dtype=jnp.float32))
            covered_total += int(np.sum(t_test >= np.asarray(iv.lower)))
            points_total += n_test

        # 3.5 * SE has two-sided normal false-failure probability 2 * Phi(-3.5) = 4.65e-4.
        se = math.sqrt(alpha * (1.0 - alpha) / num_draws)
        assert (covered_total / points_total) >= (1.0 - alpha) - 3.5 * se
