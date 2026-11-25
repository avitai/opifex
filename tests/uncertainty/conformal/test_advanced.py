"""Cross-conformal / Jackknife+ / weighted conformal contracts.

References:

* Barber, Candes, Ramdas, Tibshirani 2021, "Predictive Inference with the
  Jackknife+" (arXiv:1905.02928). J+ guarantees coverage at least
  ``1 - 2*alpha`` with no exchangeability slack; CV+ is the K-fold variant.
* Tibshirani, Foygel Barber, Candes, Ramdas 2019, "Conformal Prediction
  Under Covariate Shift" (arXiv:1904.06019). Weighted conformal under
  known likelihood ratios.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter


def _import_advanced():
    from opifex.uncertainty.conformal import advanced

    return advanced


# ---------------------------------------------------------------------------
# CV+ aggregation
# ---------------------------------------------------------------------------


def test_cv_plus_intervals_deterministic_on_fixed_fold_predictions() -> None:
    """For fixed fold predictions and targets, CV+ output must be deterministic."""
    advanced = _import_advanced()
    # 3 folds, 4 calib points per fold, 5 test points each.
    fold_val_preds = jnp.asarray(
        [
            [0.0, 0.1, -0.1, 0.05],
            [0.2, -0.2, 0.0, 0.1],
            [0.05, -0.05, 0.15, -0.1],
        ]
    )
    fold_val_targets = jnp.asarray(
        [
            [0.1, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.1, 0.1],
            [0.15, 0.0, 0.2, 0.0],
        ]
    )
    fold_test_preds = jnp.asarray(
        [
            [0.0, 0.1, 0.2, -0.1, 0.05],
            [0.05, 0.1, 0.15, -0.05, 0.0],
            [-0.05, 0.15, 0.2, -0.1, 0.1],
        ]
    )
    interval_a = advanced.cv_plus_intervals(
        fold_val_predictions=fold_val_preds,
        fold_val_targets=fold_val_targets,
        fold_test_predictions=fold_test_preds,
        alpha=0.1,
    )
    interval_b = advanced.cv_plus_intervals(
        fold_val_predictions=fold_val_preds,
        fold_val_targets=fold_val_targets,
        fold_test_predictions=fold_test_preds,
        alpha=0.1,
    )
    assert bool(jnp.allclose(interval_a.lower, interval_b.lower))
    assert bool(jnp.allclose(interval_a.upper, interval_b.upper))

    # Exact decimal order-statistic boundary: n=9, alpha=0.3 -> lo_rank=3, hi_rank=7 (not 2, 8)
    loo_targets = jnp.arange(1.0, 10.0, dtype=jnp.float32)
    loo_preds = jnp.zeros(9, dtype=jnp.float32)
    loo_test = jnp.zeros((9, 1), dtype=jnp.float32)
    jp_03 = advanced.jackknife_plus_intervals(
        loo_val_predictions=loo_preds,
        loo_val_targets=loo_targets,
        loo_test_predictions=loo_test,
        alpha=0.3,
    )
    # Candidate lowers on [-9..-1]: 3rd smallest is -7.0; candidate uppers on [1..9]: 7th is +7.0
    assert float(jp_03.lower[0]) == pytest.approx(-7.0)
    assert float(jp_03.upper[0]) == pytest.approx(7.0)


def test_cv_plus_records_fold_count_and_alpha_metadata() -> None:
    advanced = _import_advanced()
    fold_val_preds = jnp.zeros((3, 8))
    fold_val_targets = jnp.zeros((3, 8))
    fold_test_preds = jnp.zeros((3, 4))
    interval = advanced.cv_plus_intervals(
        fold_val_predictions=fold_val_preds,
        fold_val_targets=fold_val_targets,
        fold_test_predictions=fold_test_preds,
        alpha=0.1,
    )
    md = dict(interval.metadata)
    assert md["method"] == "cv_plus"
    assert int(md["num_folds"]) == 3
    assert md["alpha"] == pytest.approx(0.1)
    assert int(md["calibration_size_per_fold"]) == 8


def test_cv_plus_rejects_mismatched_fold_shapes() -> None:
    advanced = _import_advanced()
    fold_val_preds = jnp.zeros((3, 8))
    fold_val_targets = jnp.zeros((4, 8))  # wrong: 4 folds vs 3
    fold_test_preds = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match=r"(?i)fold"):
        advanced.cv_plus_intervals(
            fold_val_predictions=fold_val_preds,
            fold_val_targets=fold_val_targets,
            fold_test_predictions=fold_test_preds,
            alpha=0.1,
        )


def test_cv_plus_hits_target_coverage_on_exchangeable_data() -> None:
    """CV+ guarantee: empirical coverage >= 1 - 2*alpha on exchangeable data."""
    advanced = _import_advanced()
    rng = np.random.default_rng(0)
    n_per_fold = 256
    n_folds = 5
    n_test = 1024
    noise = 0.5
    fold_val_preds = jnp.asarray(rng.normal(size=(n_folds, n_per_fold)))
    fold_val_targets = fold_val_preds + noise * jnp.asarray(rng.normal(size=(n_folds, n_per_fold)))
    test_truth = jnp.asarray(rng.normal(size=(n_test,)))
    fold_test_preds = jnp.broadcast_to(test_truth, (n_folds, n_test))
    interval = advanced.cv_plus_intervals(
        fold_val_predictions=fold_val_preds,
        fold_val_targets=fold_val_targets,
        fold_test_predictions=fold_test_preds,
        alpha=0.1,
    )
    test_targets = test_truth + noise * jnp.asarray(rng.normal(size=(n_test,)))
    covered = (test_targets >= interval.lower) & (test_targets <= interval.upper)
    empirical = float(jnp.mean(covered.astype(jnp.float32)))
    assert empirical >= 0.8  # CV+ guarantees 1 - 2*alpha = 0.8


# ---------------------------------------------------------------------------
# Weighted conformal (covariate shift)
# ---------------------------------------------------------------------------


def test_weighted_conformal_quantile_changes_with_weights() -> None:
    """Skewing weight toward small scores must reduce the weighted threshold."""
    advanced = _import_advanced()
    scores = jnp.linspace(0.1, 1.2, 12)
    uniform = jnp.ones_like(scores)
    # Heavy mass on the smallest score → step-CDF crosses (1-alpha) earlier.
    skewed_toward_small = jnp.concatenate([jnp.array([20.0]), jnp.ones((11,))])
    q_uniform = float(
        advanced.weighted_conformal_quantile(
            scores=scores, weights=uniform, test_weight=1.0, alpha=0.1
        )
    )
    q_skewed = float(
        advanced.weighted_conformal_quantile(
            scores=scores, weights=skewed_toward_small, test_weight=1.0, alpha=0.1
        )
    )
    assert math.isfinite(q_uniform)
    assert q_skewed < q_uniform


def test_weighted_conformal_uniform_weights_match_step_cdf_formula() -> None:
    """Under uniform weights ``w_i = 1`` and ``test_weight = 1``, Tibshirani Eq. 5
    selects the exact ``ceil((n + 1)(1 - alpha))``-th order statistic."""
    from calibrax.statistics import conformal_quantile_rank

    from opifex.uncertainty.conformal import ipcw_conformal_quantile

    advanced = _import_advanced()
    n = 64
    alpha = 0.1
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.uniform(size=(n,)))
    uniform = jnp.ones_like(scores)
    weighted = float(
        advanced.weighted_conformal_quantile(
            scores=scores, weights=uniform, test_weight=1.0, alpha=alpha
        )
    )
    sorted_scores = np.sort(np.asarray(scores))
    expected_idx = conformal_quantile_rank(n, alpha) - 1
    assert weighted == pytest.approx(float(sorted_scores[expected_idx]), abs=1e-6)
    assert weighted == pytest.approx(
        float(
            ipcw_conformal_quantile(
                scores=scores, calibration_weights=uniform, test_weight=1.0, alpha=alpha
            )
        ),
        abs=1e-6,
    )

    # Below minimum_calibration_size (e.g. n=5, alpha=0.1 -> 5/6 < 0.9), threshold is +inf.
    small_q = advanced.weighted_conformal_quantile(
        scores=jnp.arange(1.0, 6.0),
        weights=jnp.ones((5,)),
        test_weight=1.0,
        alpha=0.1,
    )
    assert bool(jnp.isposinf(small_q))


def test_weighted_conformal_quantile_test_weight_and_zero_weight_invariants() -> None:
    """Zero-weight scores are excluded and large ``test_weight`` yields ``+inf``."""
    advanced = _import_advanced()
    scores = jnp.array([-100.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    weights = jnp.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    q_low_test = float(
        advanced.weighted_conformal_quantile(
            scores=scores, weights=weights, alpha=0.2, test_weight=1.0
        )
    )
    assert q_low_test >= 0.2
    q_high_test = advanced.weighted_conformal_quantile(
        scores=scores, weights=weights, alpha=0.2, test_weight=10.0
    )
    assert bool(jnp.isposinf(q_high_test))


def test_weighted_split_conformal_intervals_per_test_weights_and_single_trace() -> None:
    """Per-test-point ``test_weights`` map via ``vmap`` and compile in a single trace."""
    advanced = _import_advanced()
    scores = jnp.linspace(0.1, 1.0, 10, dtype=jnp.float32)
    cal_weights = jnp.ones((10,), dtype=jnp.float32)
    preds = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    test_weights = jnp.array([0.5, 1.0, 5.0], dtype=jnp.float32)

    counter = TraceCounter()
    jitted = jax.jit(
        counter.wrap(
            lambda p, tw: advanced.weighted_split_conformal_intervals(
                calibration_scores=scores,
                calibration_weights=cal_weights,
                predictions=p,
                alpha=0.1,
                test_weights=tw,
            )
        )
    )
    with counter.expect(new_traces=1):
        iv1 = jitted(preds, test_weights)
    with counter.expect(new_traces=0):
        iv2 = jitted(preds + 1.0, test_weights)

    widths = iv1.upper - iv1.lower
    assert bool(jnp.isfinite(widths[0]))
    assert bool(jnp.isfinite(widths[1]))
    assert bool(widths[1] >= widths[0])
    assert bool(jnp.isposinf(widths[2]))
    assert iv2.lower.shape == (3,)

    with pytest.raises(ValueError, match=r"test_weights"):
        advanced.weighted_split_conformal_intervals(
            calibration_scores=scores,
            calibration_weights=cal_weights,
            predictions=preds,
            alpha=0.1,
            test_weights=jnp.ones((2,), dtype=jnp.float32),
        )


def test_weighted_conformal_intervals_record_assumption_status() -> None:
    """Weighted-conformal output must declare that distribution-free
    guarantees hold only under the supplied likelihood-ratio weights."""
    advanced = _import_advanced()
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.uniform(size=(64,)))
    weights = jnp.asarray(rng.uniform(0.5, 1.5, size=(64,)))
    predictions = jnp.zeros((16,))
    interval = advanced.weighted_split_conformal_intervals(
        calibration_scores=scores,
        calibration_weights=weights,
        predictions=predictions,
        test_weights=1.0,
        alpha=0.1,
    )
    md = dict(interval.metadata)
    assert md["method"] == "weighted_split_conformal"
    assert md["assumption_status"] == "weights_required"
    assert md["alpha"] == pytest.approx(0.1)


def test_weighted_conformal_requires_explicit_test_weights_and_scale_invariance() -> None:
    """``test_weight`` / ``test_weights`` are mandatory, scale-invariant, and match
    exact unweighted ``conformal_quantile`` under equal weights."""
    from opifex.uncertainty.conformal import conformal_quantile

    advanced = _import_advanced()
    scores_20 = jnp.linspace(0.1, 2.0, 20, dtype=jnp.float32)

    with pytest.raises(TypeError):
        advanced.weighted_conformal_quantile(  # type: ignore[call-arg]
            scores=scores_20,
            weights=jnp.ones((20,), dtype=jnp.float32),
            alpha=0.1,
        )
    with pytest.raises(TypeError):
        advanced.weighted_split_conformal_intervals(  # type: ignore[call-arg]
            calibration_scores=scores_20,
            calibration_weights=jnp.ones((20,), dtype=jnp.float32),
            predictions=jnp.zeros((3,), dtype=jnp.float32),
            alpha=0.1,
        )

    # Scale invariance: c in {1e-3, 1/20, 1.0, 37.5, 1e4} all yield 1.90 on linspace(0.1, 2.0, 20)
    preds = jnp.array([0.0, 1.5, -2.0], dtype=jnp.float32)
    scales = (1e-3, 1.0 / 20.0, 1.0, 37.5, 1e4)
    for scale in scales:
        w_cal = jnp.full((20,), scale, dtype=jnp.float32)
        q_val = advanced.weighted_conformal_quantile(
            scores=scores_20,
            weights=w_cal,
            test_weight=jnp.float32(scale),
            alpha=0.1,
        )
        assert float(q_val) == pytest.approx(1.90, abs=1e-6), scale
        iv = advanced.weighted_split_conformal_intervals(
            calibration_scores=scores_20,
            calibration_weights=w_cal,
            predictions=preds,
            test_weights=jnp.full((3,), scale, dtype=jnp.float32),
            alpha=0.1,
        )
        assert jnp.allclose(iv.upper - preds, jnp.full((3,), 1.90, dtype=jnp.float32), atol=1e-6)

    # Boundary n = 19, alpha = 0.05 scale invariance in float32 and float64
    with jax.enable_x64(True):
        for dtype in (jnp.float32, jnp.float64):
            scores_19 = jnp.linspace(0.1, 2.0, 19, dtype=dtype)
            for scale in scales:
                q_19 = advanced.weighted_conformal_quantile(
                    scores=scores_19,
                    weights=jnp.full((19,), scale, dtype=dtype),
                    test_weight=jnp.asarray(scale, dtype=dtype),
                    alpha=0.05,
                )
                assert q_19.dtype == dtype
                assert float(q_19) == pytest.approx(2.0, abs=1e-6), (dtype, scale)

        # Equal-weights parity across multiple (n, alpha) pairs
        for n, alpha in ((5, 0.1), (9, 0.1), (12, 0.1), (19, 0.05), (20, 0.1), (25, 0.15)):
            s = jnp.linspace(0.1, 2.5, n, dtype=jnp.float64)
            q_w = advanced.weighted_conformal_quantile(
                scores=s,
                weights=jnp.ones((n,), dtype=jnp.float64),
                test_weight=1.0,
                alpha=alpha,
            )
            q_u = conformal_quantile(scores=s, alpha=alpha)
            if jnp.isposinf(q_u):
                assert bool(jnp.isposinf(q_w)), (n, alpha)
            else:
                assert float(q_w) == pytest.approx(float(q_u), abs=1e-12), (n, alpha)

    with pytest.raises(TypeError, match="static_argnames"):
        jax.jit(
            lambda a: advanced.weighted_conformal_quantile(
                scores=scores_20,
                weights=jnp.ones((20,), dtype=jnp.float32),
                test_weight=1.0,
                alpha=a,
            )
        )(jnp.float32(0.1))


def test_ipcw_conformal_quantile_equal_weight_grid_and_scipy_unequal_weights() -> None:
    """Float32 equal-weights parity against conformal_quantile_rank and unequal-weights
    parity against scipy.stats.quantile(..., method='inverted_cdf')."""
    from calibrax.statistics import conformal_quantile_rank
    from scipy import stats

    from opifex.uncertainty.conformal import ipcw_conformal_quantile

    q74 = ipcw_conformal_quantile(
        scores=jnp.arange(1, 75, dtype=jnp.float32),
        calibration_weights=jnp.ones(74, dtype=jnp.float32),
        test_weight=jnp.float32(1.0),
        alpha=0.84,
    )
    assert float(q74) == float(conformal_quantile_rank(74, 0.84)) == 12.0

    jitted_ipcw = jax.jit(ipcw_conformal_quantile, static_argnames=("alpha",))
    boundary_ns = (1, 2, 3, 9, 10, 19, 20, 48, 49, 74, 99, 100, 199, 299)
    grid_alphas = (0.01, 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.5, 0.84, 0.99, 1.0 / 3.0, 1.0 / 49.0)
    for n_f32 in boundary_ns:
        s_n = jnp.arange(1, n_f32 + 1, dtype=jnp.float32)
        w_n = jnp.ones(n_f32, dtype=jnp.float32)
        t_one = jnp.float32(1.0)
        for a in grid_alphas:
            got = float(
                jitted_ipcw(scores=s_n, calibration_weights=w_n, test_weight=t_one, alpha=a)
            )
            rank = conformal_quantile_rank(n_f32, a)
            expected = float(rank) if rank <= n_f32 else np.inf
            assert got == expected, (n_f32, a)

    with jax.enable_x64(True):
        rng_w = np.random.default_rng(17)
        s_uneq = rng_w.uniform(-2.0, 5.0, size=25)
        w_uneq = rng_w.uniform(0.2, 3.0, size=25)
        for tw_val in (0.1, 0.8, 2.5):
            aug_s = np.concatenate([s_uneq, [1e30]])
            aug_w = np.concatenate([w_uneq, [tw_val]])
            for a_uneq in (0.05, 0.1, 0.2, 0.35, 0.7):
                ref_q = float(
                    stats.quantile(aug_s, 1.0 - a_uneq, weights=aug_w, method="inverted_cdf")
                )
                got_q = float(
                    ipcw_conformal_quantile(
                        scores=jnp.asarray(s_uneq, dtype=jnp.float64),
                        calibration_weights=jnp.asarray(w_uneq, dtype=jnp.float64),
                        test_weight=jnp.float64(tw_val),
                        alpha=a_uneq,
                    )
                )
                if ref_q >= 1e29:
                    assert math.isinf(got_q) and got_q > 0.0
                else:
                    assert got_q == pytest.approx(ref_q, abs=1e-12)


def test_weighted_conformal_achieves_coverage_under_covariate_shift() -> None:
    """Tibshirani et al. (2019) Theorem 1: density-ratio weights ``w(x) = p_test(x) / p_cal(x)``
    restore marginal ``1 - alpha`` coverage when heteroscedastic test covariates shift."""

    advanced = _import_advanced()
    alpha = 0.1
    n_cal = 200
    n_trials = 400
    rng = np.random.default_rng(2026)

    # Calibration X ~ N(0, 1); Test X ~ N(1.2, 1) where noise scale grows strongly with X.
    shift = 1.2
    x_cal = rng.normal(loc=0.0, scale=1.0, size=(n_trials, n_cal))
    sigma_cal = 0.15 + 0.85 * np.exp(0.6 * x_cal)
    scores_cal = np.abs(sigma_cal * rng.normal(size=(n_trials, n_cal)))
    # Likelihood ratio N(x; shift, 1) / N(x; 0, 1) = exp(shift * x - 0.5 * shift^2)
    w_cal = np.exp(shift * x_cal - 0.5 * shift * shift)

    x_test = rng.normal(loc=shift, scale=1.0, size=(n_trials,))
    sigma_test = 0.15 + 0.85 * np.exp(0.6 * x_test)
    scores_test = np.abs(sigma_test * rng.normal(size=(n_trials,)))
    w_test = np.exp(shift * x_test - 0.5 * shift * shift)

    weighted_thresholds = jax.jit(
        jax.vmap(
            lambda s, wc, wt: advanced.weighted_conformal_quantile(
                scores=s, weights=wc, test_weight=wt, alpha=alpha
            )
        )
    )(
        jnp.asarray(scores_cal, dtype=jnp.float32),
        jnp.asarray(w_cal, dtype=jnp.float32),
        jnp.asarray(w_test, dtype=jnp.float32),
    )
    weighted_cov = float(np.mean(scores_test <= np.asarray(weighted_thresholds)))

    # Positive control: unweighted split conformal undercovers significantly below 1 - alpha
    # across all n_trials=400 rightward-shifted trials.
    from opifex.uncertainty.conformal import conformal_quantile

    unweighted_thresholds = jax.jit(jax.vmap(lambda s: conformal_quantile(scores=s, alpha=alpha)))(
        jnp.asarray(scores_cal, dtype=jnp.float32)
    )
    unweighted_cov = float(np.mean(scores_test <= np.asarray(unweighted_thresholds)))

    # Binomial standard error under nominal coverage 1 - alpha across n_trials=400 independent
    # draws; 3.5 * se has two-sided normal false-failure probability 2 * Phi(-3.5) = 4.65e-4.
    se = math.sqrt(alpha * (1.0 - alpha) / n_trials)
    assert weighted_cov >= (1.0 - alpha) - 3.5 * se
    assert unweighted_cov < (1.0 - alpha) - 3.5 * se


# ---------------------------------------------------------------------------
# Jackknife+
# ---------------------------------------------------------------------------


def test_jackknife_plus_is_deterministic_and_records_loo_size() -> None:
    advanced = _import_advanced()
    rng = np.random.default_rng(0)
    n_calib = 32
    n_test = 8
    loo_val_preds = jnp.asarray(rng.normal(size=(n_calib,)))
    loo_val_targets = loo_val_preds + 0.5 * jnp.asarray(rng.normal(size=(n_calib,)))
    loo_test_preds = jnp.asarray(rng.normal(size=(n_calib, n_test)))
    interval_a = advanced.jackknife_plus_intervals(
        loo_val_predictions=loo_val_preds,
        loo_val_targets=loo_val_targets,
        loo_test_predictions=loo_test_preds,
        alpha=0.1,
    )
    interval_b = advanced.jackknife_plus_intervals(
        loo_val_predictions=loo_val_preds,
        loo_val_targets=loo_val_targets,
        loo_test_predictions=loo_test_preds,
        alpha=0.1,
    )
    assert bool(jnp.allclose(interval_a.lower, interval_b.lower))
    md = dict(interval_a.metadata)
    assert md["method"] == "jackknife_plus"
    assert int(md["loo_size"]) == n_calib


def test_jackknife_plus_and_cv_plus_use_exact_barber_order_statistics() -> None:
    """Barber et al. (2021) Eqs. 2.2 & 3.2: exact ``floor(alpha*(n+1))`` and
    ``ceil((1-alpha)*(n+1))`` order statistics rather than interpolated quantiles."""
    advanced = _import_advanced()
    # n = 9, alpha = 0.1 -> k_lo = floor(1.0) = 1 (min), k_hi = ceil(9.0) = 9 (max).
    # Continuous jnp.quantile would interpolate to [-8.2, +8.2] instead of [-9.0, +9.0].
    loo_val_preds = jnp.zeros((9,), dtype=jnp.float32)
    loo_val_targets = jnp.arange(1.0, 10.0, dtype=jnp.float32)
    loo_test_preds = jnp.zeros((9, 2), dtype=jnp.float32)

    jk_iv = advanced.jackknife_plus_intervals(
        loo_val_predictions=loo_val_preds,
        loo_val_targets=loo_val_targets,
        loo_test_predictions=loo_test_preds,
        alpha=0.1,
    )
    assert jnp.allclose(jk_iv.lower, jnp.array([-9.0, -9.0]))
    assert jnp.allclose(jk_iv.upper, jnp.array([9.0, 9.0]))

    cv_iv = advanced.cv_plus_intervals(
        fold_val_predictions=loo_val_preds.reshape(3, 3),
        fold_val_targets=loo_val_targets.reshape(3, 3),
        fold_test_predictions=jnp.zeros((3, 2), dtype=jnp.float32),
        alpha=0.1,
    )
    assert jnp.allclose(cv_iv.lower, jnp.array([-9.0, -9.0]))
    assert jnp.allclose(cv_iv.upper, jnp.array([9.0, 9.0]))

    # n = 8, alpha = 0.1 -> k_lo = floor(0.9) = 0 < 1 (-inf), k_hi = ceil(8.1) = 9 > 8 (+inf).
    jk_small = advanced.jackknife_plus_intervals(
        loo_val_predictions=jnp.zeros((8,)),
        loo_val_targets=jnp.arange(1.0, 9.0),
        loo_test_predictions=jnp.zeros((8, 2)),
        alpha=0.1,
    )
    assert bool(jnp.all(jnp.isneginf(jk_small.lower)))
    assert bool(jnp.all(jnp.isposinf(jk_small.upper)))

    cv_small = advanced.cv_plus_intervals(
        fold_val_predictions=jnp.zeros((2, 4)),
        fold_val_targets=jnp.arange(1.0, 9.0).reshape(2, 4),
        fold_test_predictions=jnp.zeros((2, 2)),
        alpha=0.1,
    )
    assert bool(jnp.all(jnp.isneginf(cv_small.lower)))
    assert bool(jnp.all(jnp.isposinf(cv_small.upper)))

    for bad_alpha in (0.0, 1.0, -0.2, 1.1):
        with pytest.raises(ValueError, match=r"alpha"):
            advanced.jackknife_plus_intervals(
                loo_val_predictions=loo_val_preds,
                loo_val_targets=loo_val_targets,
                loo_test_predictions=loo_test_preds,
                alpha=bad_alpha,
            )
        with pytest.raises(ValueError, match=r"alpha"):
            advanced.cv_plus_intervals(
                fold_val_predictions=loo_val_preds.reshape(3, 3),
                fold_val_targets=loo_val_targets.reshape(3, 3),
                fold_test_predictions=jnp.zeros((3, 2)),
                alpha=bad_alpha,
            )

    # Mis-ranked array rejection and exact rational alpha boundaries
    with pytest.raises(ValueError, match="1-D"):
        advanced.jackknife_plus_intervals(
            loo_val_predictions=jnp.zeros((9, 1)),
            loo_val_targets=jnp.zeros((9,)),
            loo_test_predictions=jnp.zeros((9, 2)),
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="ndim >= 2"):
        advanced.jackknife_plus_intervals(
            loo_val_predictions=jnp.zeros((9,)),
            loo_val_targets=jnp.zeros((9,)),
            loo_test_predictions=jnp.zeros((9,)),
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="fold shapes"):
        advanced.cv_plus_intervals(
            fold_val_predictions=jnp.zeros((3, 3)),
            fold_val_targets=jnp.zeros((3, 3)),
            fold_test_predictions=jnp.zeros((3,)),
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="1-D"):
        advanced.weighted_split_conformal_intervals(
            calibration_scores=jnp.ones((10,)),
            calibration_weights=jnp.ones((10,)),
            predictions=jnp.zeros((3, 1)),
            test_weights=1.0,
            alpha=0.1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        advanced.weighted_conformal_quantile(
            scores=jnp.array([0.1, 1.0]),
            weights=jnp.array([1.0, 1.0]),
            alpha=0.1,
            test_weight=-1.0,
        )

    alpha_49 = 1.0 / 49.0
    jk_48 = advanced.jackknife_plus_intervals(
        loo_val_predictions=jnp.zeros((48,)),
        loo_val_targets=jnp.arange(1.0, 49.0),
        loo_test_predictions=jnp.zeros((48, 2)),
        alpha=alpha_49,
    )
    assert bool(jnp.all(jnp.isneginf(jk_48.lower)) and jnp.all(jnp.isposinf(jk_48.upper)))


def test_jackknife_plus_and_cv_plus_jit_vmap_grad_and_float64() -> None:
    """J+ and CV+ compose cleanly under ``jit``, ``vmap``, ``grad``, and float64."""
    advanced = _import_advanced()
    counter = TraceCounter()
    jitted_jk = jax.jit(
        counter.wrap(
            lambda vp, vt, tp: advanced.jackknife_plus_intervals(
                loo_val_predictions=vp,
                loo_val_targets=vt,
                loo_test_predictions=tp,
                alpha=0.1,
            )
        )
    )
    vp = jnp.zeros((10,), dtype=jnp.float32)
    vt = jnp.linspace(0.1, 1.0, 10, dtype=jnp.float32)
    tp = jnp.ones((10, 3), dtype=jnp.float32)
    with counter.expect(new_traces=1):
        out1 = jitted_jk(vp, vt, tp)
    with counter.expect(new_traces=0):
        _ = jitted_jk(vp + 0.1, vt, tp)
    assert out1.lower.shape == (3,)

    grad_tp = jax.grad(
        lambda x: jnp.sum(
            advanced.jackknife_plus_intervals(
                loo_val_predictions=vp,
                loo_val_targets=vt,
                loo_test_predictions=x,
                alpha=0.1,
            ).upper
        )
    )(tp)
    assert bool(jnp.all(jnp.isfinite(grad_tp)))

    with jax.enable_x64(True):
        iv64 = advanced.cv_plus_intervals(
            fold_val_predictions=jnp.zeros((2, 5), dtype=jnp.float64),
            fold_val_targets=jnp.linspace(0.1, 1.0, 10, dtype=jnp.float64).reshape(2, 5),
            fold_test_predictions=jnp.ones((2, 4), dtype=jnp.float64),
            alpha=0.1,
        )
        assert iv64.lower.dtype == jnp.float64
        assert iv64.upper.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_weighted_conformal_quantile_is_jit_and_vmap_compatible() -> None:
    advanced = _import_advanced()
    rng = np.random.default_rng(0)
    scores_batch = jnp.asarray(rng.uniform(size=(3, 64)))
    weights_batch = jnp.asarray(rng.uniform(0.5, 1.5, size=(3, 64)))
    test_weights_batch = jnp.asarray([0.8, 1.0, 1.2])
    quantiles = jax.jit(
        jax.vmap(
            lambda s, w, wt: advanced.weighted_conformal_quantile(
                scores=s, weights=w, test_weight=wt, alpha=0.1
            )
        )
    )(scores_batch, weights_batch, test_weights_batch)
    assert quantiles.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(quantiles)))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_advanced_components() -> None:
    from opifex.uncertainty import conformal

    expected = {
        "CrossCQRState",
        "CrossConformalizedQuantileRegressor",
        "cv_plus_cqr_intervals",
        "cv_plus_intervals",
        "feedback_covariate_shift_intervals",
        "feedback_covariate_shift_weights",
        "jackknife_plus_intervals",
        "weighted_conformal_quantile",
        "weighted_split_conformal_intervals",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public advanced symbols: {sorted(missing)}"


# ---------------------------------------------------------------------------
# CrossConformalizedQuantileRegressor / cv_plus_cqr_intervals
# ---------------------------------------------------------------------------


def test_cv_plus_cqr_and_cross_cqr_parity_order_stats_and_coverage() -> None:
    """``cv_plus_cqr_intervals`` and ``CrossConformalizedQuantileRegressor`` agree
    and satisfy Barber et al. (2021) exact order statistics and ``1 - 2*alpha`` coverage."""
    advanced = _import_advanced()
    rng = np.random.default_rng(41)
    n_folds, n_per_fold, n_test = 5, 128, 512
    x_val = rng.uniform(0.5, 2.0, size=(n_folds, n_per_fold))
    y_val = rng.normal(loc=0.0, scale=x_val)
    # Deliberately narrow raw bands [-0.5 * x, +0.5 * x]
    fold_val_lo = jnp.asarray(-0.5 * x_val, dtype=jnp.float32)
    fold_val_hi = jnp.asarray(0.5 * x_val, dtype=jnp.float32)
    fold_val_y = jnp.asarray(y_val, dtype=jnp.float32)

    x_test = rng.uniform(0.5, 2.0, size=(n_test,))
    y_test = jnp.asarray(rng.normal(loc=0.0, scale=x_test), dtype=jnp.float32)
    fold_test_lo = jnp.broadcast_to(
        jnp.asarray(-0.5 * x_test, dtype=jnp.float32), (n_folds, n_test)
    )
    fold_test_hi = jnp.broadcast_to(jnp.asarray(0.5 * x_test, dtype=jnp.float32), (n_folds, n_test))

    iv_fn = advanced.cv_plus_cqr_intervals(
        fold_val_lower=fold_val_lo,
        fold_val_upper=fold_val_hi,
        fold_val_targets=fold_val_y,
        fold_test_lower=fold_test_lo,
        fold_test_upper=fold_test_hi,
        alpha=0.1,
    )
    ccqr = advanced.CrossConformalizedQuantileRegressor(alpha=0.1)
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        ccqr.predict(fold_test_lower=fold_test_lo, fold_test_upper=fold_test_hi)
    state = ccqr.fit(
        fold_val_lower=fold_val_lo,
        fold_val_upper=fold_val_hi,
        fold_val_targets=fold_val_y,
    )
    iv_cls = ccqr.with_state(state).predict(
        fold_test_lower=fold_test_lo,
        fold_test_upper=fold_test_hi,
    )
    assert jnp.allclose(iv_fn.lower, iv_cls.lower)
    assert jnp.allclose(iv_fn.upper, iv_cls.upper)
    covered = (y_test >= iv_fn.lower) & (y_test <= iv_fn.upper)
    assert float(jnp.mean(covered.astype(jnp.float32))) >= 0.8

    # Unattainable finite-sample order statistic (n = 8, alpha = 0.1 -> (-inf, +inf))
    iv_small = advanced.cv_plus_cqr_intervals(
        fold_val_lower=jnp.zeros((2, 4)),
        fold_val_upper=jnp.ones((2, 4)),
        fold_val_targets=jnp.arange(8.0).reshape(2, 4),
        fold_test_lower=jnp.zeros((2, 3)),
        fold_test_upper=jnp.ones((2, 3)),
        alpha=0.1,
    )
    assert bool(jnp.all(jnp.isneginf(iv_small.lower)))
    assert bool(jnp.all(jnp.isposinf(iv_small.upper)))

    with pytest.raises(ValueError, match=r"alpha"):
        advanced.cv_plus_cqr_intervals(
            fold_val_lower=jnp.zeros((2, 5)),
            fold_val_upper=jnp.ones((2, 5)),
            fold_val_targets=jnp.zeros((2, 5)),
            fold_test_lower=jnp.zeros((2, 3)),
            fold_test_upper=jnp.ones((2, 3)),
            alpha=1.0,
        )
    with pytest.raises(ValueError, match=r"2-D"):
        ccqr.fit(
            fold_val_lower=jnp.zeros((2, 5)),
            fold_val_upper=jnp.ones((3, 5)),
            fold_val_targets=jnp.zeros((2, 5)),
        )
    with pytest.raises(ValueError, match=r"leading dim"):
        ccqr.with_state(state).predict(
            fold_test_lower=jnp.zeros((2, 4)),
            fold_test_upper=jnp.ones((2, 4)),
        )


def test_cv_plus_cqr_jit_vmap_grad_and_float64() -> None:
    """``cv_plus_cqr_intervals`` traces once and supports ``vmap``, ``grad``, and ``float64``."""
    advanced = _import_advanced()
    v_lo = jnp.zeros((2, 5), dtype=jnp.float32)
    v_hi = jnp.ones((2, 5), dtype=jnp.float32)
    v_y = jnp.linspace(-0.5, 1.5, 10, dtype=jnp.float32).reshape(2, 5)
    t_lo = jnp.zeros((2, 3), dtype=jnp.float32)
    t_hi = jnp.ones((2, 3), dtype=jnp.float32)

    counter = TraceCounter()
    jitted = jax.jit(
        counter.wrap(
            lambda tlo, thi: advanced.cv_plus_cqr_intervals(
                fold_val_lower=v_lo,
                fold_val_upper=v_hi,
                fold_val_targets=v_y,
                fold_test_lower=tlo,
                fold_test_upper=thi,
                alpha=0.1,
            )
        )
    )
    with counter.expect(new_traces=1):
        out1 = jitted(t_lo, t_hi)
    with counter.expect(new_traces=0):
        _ = jitted(t_lo + 0.1, t_hi + 0.1)
    assert out1.lower.shape == (3,)

    batched = jax.vmap(
        lambda tlo: advanced.cv_plus_cqr_intervals(
            fold_val_lower=v_lo,
            fold_val_upper=v_hi,
            fold_val_targets=v_y,
            fold_test_lower=tlo,
            fold_test_upper=tlo + 1.0,
            alpha=0.1,
        )
    )(jnp.stack([t_lo, t_lo + 0.5]))
    assert batched.lower.shape == (2, 3)

    g_hi = jax.grad(
        lambda thi: jnp.sum(
            advanced.cv_plus_cqr_intervals(
                fold_val_lower=v_lo,
                fold_val_upper=v_hi,
                fold_val_targets=v_y,
                fold_test_lower=t_lo,
                fold_test_upper=thi,
                alpha=0.1,
            ).upper
        )
    )(t_hi)
    assert bool(jnp.all(jnp.isfinite(g_hi)))

    with jax.enable_x64(True):
        iv64 = advanced.cv_plus_cqr_intervals(
            fold_val_lower=v_lo.astype(jnp.float64),
            fold_val_upper=v_hi.astype(jnp.float64),
            fold_val_targets=v_y.astype(jnp.float64),
            fold_test_lower=t_lo.astype(jnp.float64),
            fold_test_upper=t_hi.astype(jnp.float64),
            alpha=0.1,
        )
        assert iv64.lower.dtype == jnp.float64
        assert iv64.upper.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Feedback covariate shift (Fannjiang et al., PNAS 2022)
# ---------------------------------------------------------------------------


def test_feedback_covariate_shift_factorized_swapped_and_extreme_boltzmann() -> None:
    """Log-space feedback covariate-shift intervals match positive weights at moderate
    scales and remain finite without NaN at extreme Boltzmann inverse temperatures."""
    advanced = _import_advanced()
    scores = jnp.linspace(0.1, 1.2, 12, dtype=jnp.float32)
    preds = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    w_cal = jnp.linspace(0.5, 2.0, 12, dtype=jnp.float32)
    w_test = jnp.array([0.4, 1.0, 1.8], dtype=jnp.float32)

    ref_iv = advanced.weighted_split_conformal_intervals(
        calibration_scores=scores,
        calibration_weights=w_cal,
        predictions=preds,
        alpha=0.1,
        test_weights=w_test,
    )
    fcs_iv = advanced.feedback_covariate_shift_intervals(
        calibration_scores=scores,
        predictions=preds,
        alpha=0.1,
        log_calibration_ratios=jnp.log(w_cal),
        log_test_ratios=jnp.log(w_test),
    )
    assert jnp.allclose(fcs_iv.lower, ref_iv.lower, atol=1e-6)
    assert jnp.allclose(fcs_iv.upper, ref_iv.upper, atol=1e-6)
    assert dict(fcs_iv.metadata)["policy_mode"] == "factorized"

    # Extreme Boltzmann shift (+600 and -600 in float32): naive exp overflows/underflows to NaN,
    # whereas adding any per-row constant leaves normalized weights and quantiles invariant!
    shifted_iv = advanced.feedback_covariate_shift_intervals(
        calibration_scores=scores,
        predictions=preds,
        alpha=0.1,
        log_calibration_ratios=jnp.log(w_cal) + 600.0,
        log_test_ratios=jnp.log(w_test) + 600.0,
    )
    neg_shifted_iv = advanced.feedback_covariate_shift_intervals(
        calibration_scores=scores,
        predictions=preds,
        alpha=0.1,
        log_calibration_ratios=jnp.log(w_cal) - 600.0,
        log_test_ratios=jnp.log(w_test) - 600.0,
    )
    assert jnp.allclose(shifted_iv.lower, ref_iv.lower, atol=1e-6)
    assert jnp.allclose(neg_shifted_iv.lower, ref_iv.lower, atol=1e-6)

    # Swapped-policy matrix mode (Fannjiang et al. 2022, Theorem 1) + single-trace jit + float64
    swapped = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.log(w_cal)[None, :], (3, 12))
            + jnp.array([[0.0], [200.0], [-200.0]], dtype=jnp.float32),
            (jnp.log(w_test) + jnp.array([0.0, 200.0, -200.0], dtype=jnp.float32))[:, None],
        ],
        axis=-1,
    )
    counter = TraceCounter()
    jitted_swapped = jax.jit(
        counter.wrap(
            lambda s_mat: advanced.feedback_covariate_shift_intervals(
                calibration_scores=scores,
                predictions=preds,
                alpha=0.1,
                swapped_log_ratios=s_mat,
            )
        )
    )
    with counter.expect(new_traces=1):
        swapped_iv = jitted_swapped(swapped)
    with counter.expect(new_traces=0):
        _ = jitted_swapped(swapped + 10.0)
    assert jnp.allclose(swapped_iv.lower, ref_iv.lower, atol=1e-6)
    assert dict(swapped_iv.metadata)["policy_mode"] == "swapped"

    # All -inf row -> zero weights -> honest (-inf, +inf) interval with zero NaN
    all_neginf = jnp.full((1, 13), -jnp.inf, dtype=jnp.float32)
    cw_zero, tw_zero = advanced.feedback_covariate_shift_weights(swapped_log_ratios=all_neginf)
    assert bool(jnp.all(cw_zero == 0.0) and jnp.all(tw_zero == 0.0))
    iv_neginf = advanced.feedback_covariate_shift_intervals(
        calibration_scores=scores,
        predictions=jnp.array([0.0], dtype=jnp.float32),
        alpha=0.1,
        swapped_log_ratios=all_neginf,
    )
    assert bool(jnp.isneginf(iv_neginf.lower[0]) and jnp.isposinf(iv_neginf.upper[0]))

    # NaN in one row of swapped_log_ratios propagates NaN to that row without corrupting valid rows
    mixed_nan = jnp.array(
        [[jnp.nan, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    cw_nan, tw_nan = advanced.feedback_covariate_shift_weights(swapped_log_ratios=mixed_nan)
    assert bool(jnp.all(jnp.isnan(cw_nan[0])) and jnp.isnan(tw_nan[0]))
    assert bool(jnp.all(jnp.isfinite(cw_nan[1])) and jnp.isfinite(tw_nan[1]))
    iv_nan = advanced.feedback_covariate_shift_intervals(
        calibration_scores=jnp.array([1.0, 2.0], dtype=jnp.float32),
        predictions=jnp.array([0.0, 0.0], dtype=jnp.float32),
        alpha=0.5,
        swapped_log_ratios=mixed_nan,
    )
    assert bool(jnp.isnan(iv_nan.lower[0]) and jnp.isnan(iv_nan.upper[0]))
    assert bool(jnp.isfinite(iv_nan.lower[1]) and jnp.isfinite(iv_nan.upper[1]))

    with jax.enable_x64(True):
        cw64, tw64 = advanced.feedback_covariate_shift_weights(
            swapped_log_ratios=swapped.astype(jnp.float64)
        )
        assert cw64.dtype == jnp.float64
        assert tw64.dtype == jnp.float64
        assert jnp.allclose(jnp.sum(cw64, axis=-1) + tw64, jnp.ones((3,), dtype=jnp.float64))

    with pytest.raises(ValueError, match=r"not both"):
        advanced.feedback_covariate_shift_weights(
            log_calibration_ratios=jnp.zeros((4,)),
            log_test_ratios=jnp.zeros((2,)),
            swapped_log_ratios=jnp.zeros((2, 5)),
        )
    with pytest.raises(ValueError, match=r"Provide either"):
        advanced.feedback_covariate_shift_weights()
    with pytest.raises(ValueError, match=r"match"):
        advanced.feedback_covariate_shift_intervals(
            calibration_scores=scores,
            predictions=preds,
            alpha=0.1,
            swapped_log_ratios=jnp.zeros((3, 5)),
        )
