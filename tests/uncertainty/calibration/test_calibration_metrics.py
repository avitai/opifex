"""Calibration metric numerical contracts.

Each metric must:

1. Match a closed-form reference value on a small hand-picked example.
2. Be JAX-transform-friendly (``jax.jit`` / ``jax.grad`` / ``jax.vmap``).

Wrapped CalibraX metrics (Brier, ECE, pinball) are cross-checked against
``calibrax.metrics.functional`` directly to make the wrapper a thin pass-through.
Opifex-local metrics (Gaussian NLL, PICP, MPIW, regression calibration error)
are cross-checked against handwritten NumPy formulas.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module imports (will fail until Task 4.1 implementation lands).
# ---------------------------------------------------------------------------


def _import_calibration_module():
    """Defer module import so collection still shows a clean failure."""
    from opifex.uncertainty import calibration

    return calibration


# ---------------------------------------------------------------------------
# Gaussian NLL
# ---------------------------------------------------------------------------


def test_gaussian_nll_matches_closed_form_reference() -> None:
    calibration = _import_calibration_module()
    mean = jnp.array([0.0, 1.0, -2.0])
    variance = jnp.array([1.0, 4.0, 0.25])
    target = jnp.array([0.5, 0.0, -1.5])
    # 0.5 * (log(2π σ²) + (y - μ)² / σ²), then mean over batch.
    diff = np.asarray(target) - np.asarray(mean)
    var = np.asarray(variance)
    elementwise = 0.5 * (np.log(2.0 * np.pi * var) + diff**2 / var)
    expected = float(elementwise.mean())
    out = float(calibration.gaussian_nll(mean=mean, variance=variance, target=target))
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_gaussian_nll_rejects_nonpositive_variance() -> None:
    """``validate=True`` is the eager safety path; default is jit-friendly."""
    calibration = _import_calibration_module()
    mean = jnp.zeros((3,))
    target = jnp.zeros((3,))
    variance = jnp.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="variance"):
        calibration.gaussian_nll(mean=mean, variance=variance, target=target, validate=True)


def test_gaussian_nll_is_jit_and_grad_compatible() -> None:
    calibration = _import_calibration_module()
    mean = jnp.zeros((4,))
    variance = jnp.ones((4,))
    target = jnp.array([0.0, 1.0, -1.0, 0.5])

    @jax.jit
    def jitted(m: jax.Array, v: jax.Array, y: jax.Array) -> jax.Array:
        return calibration.gaussian_nll(mean=m, variance=v, target=y)

    jit_val = jitted(mean, variance, target)
    eager_val = calibration.gaussian_nll(mean=mean, variance=variance, target=target)
    assert bool(jnp.allclose(jit_val, eager_val))
    grad = jax.grad(lambda m: calibration.gaussian_nll(mean=m, variance=variance, target=target))(
        mean
    )
    assert grad.shape == mean.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


# ---------------------------------------------------------------------------
# PICP / MPIW
# ---------------------------------------------------------------------------


def test_picp_counts_in_bound_fraction() -> None:
    calibration = _import_calibration_module()
    # 4 / 5 covered: indices 0, 1, 2, 3 in-bound; index 4 outside.
    lower = jnp.array([0.0, -1.0, 1.0, 0.5, 5.0])
    upper = jnp.array([2.0, 1.0, 3.0, 2.5, 6.0])
    target = jnp.array([1.0, 0.5, 1.5, 1.0, 0.0])
    out = float(calibration.picp(lower=lower, upper=upper, target=target))
    assert out == pytest.approx(4.0 / 5.0, abs=1e-7)


def test_mpiw_returns_mean_width() -> None:
    calibration = _import_calibration_module()
    lower = jnp.array([0.0, -2.0, 1.0])
    upper = jnp.array([1.0, 1.0, 4.0])
    # Compute expected in matching JAX float32 to keep the exact-equality contract.
    expected = float(jnp.mean(jnp.array([1.0, 3.0, 3.0])))
    out = float(calibration.mpiw(lower=lower, upper=upper))
    assert out == expected


def test_picp_rejects_inverted_bounds() -> None:
    """``validate=True`` is the eager safety path; default is jit-friendly."""
    calibration = _import_calibration_module()
    lower = jnp.array([1.0, 2.0])
    upper = jnp.array([0.5, 3.0])  # first interval inverted
    target = jnp.array([0.0, 2.5])
    with pytest.raises(ValueError, match="lower"):
        calibration.picp(lower=lower, upper=upper, target=target, validate=True)


def test_picp_and_mpiw_jit_compat() -> None:
    calibration = _import_calibration_module()
    rng = np.random.default_rng(0)
    lower = jnp.asarray(rng.normal(size=(16,)))
    upper = lower + jnp.asarray(rng.uniform(0.1, 1.0, size=(16,)))
    target = jnp.asarray(rng.normal(size=(16,)))
    jitted_picp = jax.jit(lambda lo, up, y: calibration.picp(lower=lo, upper=up, target=y))
    jitted_mpiw = jax.jit(lambda lo, up: calibration.mpiw(lower=lo, upper=up))
    assert bool(jnp.isfinite(jitted_picp(lower, upper, target)))
    assert bool(jnp.isfinite(jitted_mpiw(lower, upper)))


def test_calibration_metrics_are_vmap_compatible() -> None:
    """vmap over a leading "trial" axis traces cleanly for every jit-path metric."""
    calibration = _import_calibration_module()
    rng = np.random.default_rng(0)
    mean = jnp.asarray(rng.normal(size=(3, 16)))
    variance = jnp.asarray(rng.uniform(0.1, 1.0, size=(3, 16)))
    target = jnp.asarray(rng.normal(size=(3, 16)))
    lower = mean - 1.0
    upper = mean + 1.0
    nll_per_trial = jax.vmap(
        lambda m, v, y: calibration.gaussian_nll(mean=m, variance=v, target=y)
    )(mean, variance, target)
    picp_per_trial = jax.vmap(lambda lo, up, y: calibration.picp(lower=lo, upper=up, target=y))(
        lower, upper, target
    )
    mpiw_per_trial = jax.vmap(lambda lo, up: calibration.mpiw(lower=lo, upper=up))(lower, upper)
    assert nll_per_trial.shape == (3,)
    assert picp_per_trial.shape == (3,)
    assert mpiw_per_trial.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(nll_per_trial)))
    assert bool(jnp.all(jnp.isfinite(picp_per_trial)))
    assert bool(jnp.all(jnp.isfinite(mpiw_per_trial)))


# ---------------------------------------------------------------------------
# Regression calibration error
# ---------------------------------------------------------------------------


def test_regression_calibration_error_on_perfectly_calibrated_distribution() -> None:
    """A standard-normal predictive distribution against true samples is calibrated."""
    calibration = _import_calibration_module()
    # Predictive: N(0, 1) for every point; target sampled from same N(0, 1).
    rng = np.random.default_rng(42)
    n = 2048
    mean = jnp.zeros((n,))
    variance = jnp.ones((n,))
    target = jnp.asarray(rng.standard_normal(size=(n,)))
    # Over 10 quantile levels evenly spaced — empirical vs nominal coverage.
    levels = jnp.linspace(0.05, 0.95, 10)
    error = float(
        calibration.regression_calibration_error(
            mean=mean, variance=variance, target=target, quantile_levels=levels
        )
    )
    # Perfect calibration → near zero (sampling tolerance for n=2048).
    assert error < 0.05


def test_regression_calibration_error_detects_miscalibration() -> None:
    """A predictive distribution biased away from the data is miscalibrated."""
    calibration = _import_calibration_module()
    rng = np.random.default_rng(0)
    n = 2048
    biased_mean = jnp.full((n,), 5.0)  # Predictive far above target distribution
    target = jnp.asarray(rng.standard_normal(size=(n,)))
    variance = jnp.ones((n,))
    levels = jnp.linspace(0.05, 0.95, 10)
    error = float(
        calibration.regression_calibration_error(
            mean=biased_mean, variance=variance, target=target, quantile_levels=levels
        )
    )
    # All targets fall in the predictive lower tail → empirical CDF ≈ 0 everywhere.
    # Mean nominal level ≈ 0.5, so absolute miscalibration averages ≈ 0.5.
    assert error > 0.4


# ---------------------------------------------------------------------------
# Thin wrappers around CalibraX
# ---------------------------------------------------------------------------


def test_brier_score_wraps_calibrax() -> None:
    from calibrax.metrics.functional.calibration import brier_score as cx_brier

    calibration = _import_calibration_module()
    probs = jnp.array([0.1, 0.9, 0.2, 0.8])
    targets = jnp.array([0.0, 1.0, 0.0, 1.0])
    out = float(calibration.brier_score(probabilities=probs, targets=targets))
    expected = float(cx_brier(probs, targets))
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_ece_wraps_calibrax_with_fixed_bins() -> None:
    from calibrax.metrics.functional.calibration import (
        expected_calibration_error as cx_ece,
    )

    calibration = _import_calibration_module()
    rng = np.random.default_rng(0)
    probs = jnp.asarray(rng.uniform(size=(256,)))
    targets = jnp.asarray((rng.uniform(size=(256,)) < probs).astype(np.int32))
    out = float(
        calibration.expected_calibration_error(probabilities=probs, targets=targets, num_bins=10)
    )
    expected = float(cx_ece(probs, targets, num_bins=10))
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_pinball_loss_wraps_calibrax() -> None:
    from calibrax.metrics.functional.regression import quantile_loss as cx_pinball

    calibration = _import_calibration_module()
    preds = jnp.array([0.0, 1.0, 2.0, 3.0])
    targets = jnp.array([0.5, 0.5, 1.5, 4.0])
    out = float(calibration.pinball_loss(predictions=preds, targets=targets, quantile=0.1))
    expected = float(cx_pinball(preds, targets, quantile=0.1))
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-7)


# ---------------------------------------------------------------------------
# Module-level re-export contract
# ---------------------------------------------------------------------------


def test_public_calibration_metric_surface() -> None:
    calibration = _import_calibration_module()
    expected_surface = {
        "gaussian_nll",
        "picp",
        "mpiw",
        "regression_calibration_error",
        "brier_score",
        "expected_calibration_error",
        "pinball_loss",
    }
    missing = expected_surface - set(dir(calibration))
    assert not missing, f"missing public calibration symbols: {sorted(missing)}"
