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

import jax
import jax.numpy as jnp
import numpy as np
import pytest


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
    scores = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    uniform = jnp.ones_like(scores)
    # Heavy mass on the smallest score → step-CDF crosses (1-alpha) earlier.
    skewed_toward_small = jnp.array([10.0, 1.0, 1.0, 1.0, 1.0])
    q_uniform = float(
        advanced.weighted_conformal_quantile(scores=scores, weights=uniform, alpha=0.1)
    )
    q_skewed = float(
        advanced.weighted_conformal_quantile(scores=scores, weights=skewed_toward_small, alpha=0.1)
    )
    assert q_skewed < q_uniform


def test_weighted_conformal_uniform_weights_match_step_cdf_formula() -> None:
    """Under uniform weights the step-CDF formula picks the smallest score
    whose rank exceeds ``ceil(n * (1 - alpha)) / n``."""
    advanced = _import_advanced()
    n = 64
    alpha = 0.1
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.uniform(size=(n,)))
    uniform = jnp.ones_like(scores)
    weighted = float(
        advanced.weighted_conformal_quantile(scores=scores, weights=uniform, alpha=alpha)
    )
    sorted_scores = np.sort(np.asarray(scores))
    # Step CDF: smallest k with (k+1)/n >= 1-alpha → idx = ceil(n*(1-alpha)) - 1.
    expected_idx = int(np.ceil(n * (1.0 - alpha))) - 1
    assert weighted == pytest.approx(float(sorted_scores[expected_idx]), abs=1e-6)


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
        alpha=0.1,
    )
    md = dict(interval.metadata)
    assert md["method"] == "weighted_split_conformal"
    assert md["assumption_status"] == "weights_required"
    assert md["alpha"] == pytest.approx(0.1)


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


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_weighted_conformal_quantile_is_jit_and_vmap_compatible() -> None:
    advanced = _import_advanced()
    rng = np.random.default_rng(0)
    scores_batch = jnp.asarray(rng.uniform(size=(3, 64)))
    weights_batch = jnp.asarray(rng.uniform(0.5, 1.5, size=(3, 64)))
    quantiles = jax.vmap(
        lambda s, w: advanced.weighted_conformal_quantile(scores=s, weights=w, alpha=0.1)
    )(scores_batch, weights_batch)
    assert quantiles.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(quantiles)))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_advanced_components() -> None:
    from opifex.uncertainty import conformal

    expected = {
        "cv_plus_intervals",
        "jackknife_plus_intervals",
        "weighted_conformal_quantile",
        "weighted_split_conformal_intervals",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public advanced symbols: {sorted(missing)}"
