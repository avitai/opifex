"""Tests for single-source-of-truth calibration tools (Task 12.2.2).

These tests pin two invariants that the previous duplicated / "simplified"
implementations violated:

* The pool-adjacent-violators (PAV) routine used by
  :class:`~opifex.neural.bayesian.calibration_tools.IsotonicRegression`
  must return a genuinely non-decreasing sequence. The historical
  single-pass implementation only averaged each adjacent pair once and
  therefore left order violations in place (e.g. ``[3, 2, 1]`` collapsed
  to ``[2.5, 1.75, 1.75]``).
* :meth:`CalibrationTools.platt_scaling` must delegate to
  :class:`PlattScaling` rather than maintain a divergent gradient loop,
  so the method and the class produce identical parameters.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.calibration_tools import (
    CalibrationTools,
    IsotonicRegression,
    PlattScaling,
)


_MONOTONIC_EPS = 1e-5


def test_pav_single_pass_failure_case_is_monotonic() -> None:
    """A strictly decreasing input is the canonical single-pass PAV failure.

    The single-pass implementation produced ``[2.5, 1.75, 1.75]`` for the
    input ``[3, 2, 1]`` (still decreasing). A correct PAV pools the whole
    block to its mean, ``[2, 2, 2]``.
    """
    rngs = nnx.Rngs(0)
    regressor = IsotonicRegression(n_bins=8, rngs=rngs)

    pooled = regressor._pool_adjacent_violators(jnp.array([3.0, 2.0, 1.0]))

    assert jnp.all(jnp.diff(pooled) >= -_MONOTONIC_EPS)
    # Standard PAV collapses a fully-decreasing block to its mean.
    assert jnp.allclose(pooled, jnp.full_like(pooled, 2.0), atol=_MONOTONIC_EPS)


def test_pav_longer_decreasing_sequence_is_monotonic() -> None:
    """A longer adversarial sequence the single-pass routine also botches."""
    rngs = nnx.Rngs(0)
    regressor = IsotonicRegression(n_bins=8, rngs=rngs)

    pooled = regressor._pool_adjacent_violators(jnp.array([5.0, 4.0, 3.0, 2.0, 1.0]))

    assert jnp.all(jnp.diff(pooled) >= -_MONOTONIC_EPS)
    assert jnp.allclose(pooled, jnp.full_like(pooled, 3.0), atol=_MONOTONIC_EPS)


def test_isotonic_output_is_monotonic() -> None:
    """Fitting on a noisy non-monotone target yields non-decreasing outputs.

    Confidences increase while labels are *anti*-correlated, so the raw
    per-bin accuracies are decreasing — a target the single-pass PAV
    cannot render monotone.
    """
    rngs = nnx.Rngs(7)
    regressor = IsotonicRegression(n_bins=10, rngs=rngs)

    num_samples = 400
    confidences = jnp.linspace(0.0, 1.0, num_samples)
    # Anti-correlated labels: high confidence -> low accuracy, plus noise.
    noise = 0.1 * jax.random.normal(rngs.sample(), (num_samples,))
    labels = ((1.0 - confidences) + noise) > 0.5

    regressor.fit(confidences, labels)

    # The learned calibration map must be non-decreasing.
    calibration_map = regressor.calibration_map[...]
    assert jnp.all(jnp.diff(calibration_map) >= -_MONOTONIC_EPS)

    # Outputs sorted by input confidence must also be non-decreasing.
    sorted_confidences = jnp.sort(confidences)
    calibrated = regressor(sorted_confidences)
    assert jnp.all(jnp.diff(calibrated) >= -_MONOTONIC_EPS)


def test_platt_method_matches_class() -> None:
    """``CalibrationTools.platt_scaling`` must delegate to ``PlattScaling``.

    Both paths are fitted on identical data with identical seeds; the
    returned ``(slope, intercept)`` must equal the class's ``(a, b)``.
    """
    seed = 123
    num_samples = 120
    data_rngs = nnx.Rngs(seed)
    logits = jax.random.normal(data_rngs.sample(), (num_samples,))
    labels = (logits > 0.0).astype(jnp.float32)

    # Class path (source of truth).
    class_scaler = PlattScaling(rngs=nnx.Rngs(seed))
    class_scaler.fit(logits, labels)
    class_slope = float(class_scaler.a[...])
    class_intercept = float(class_scaler.b[...])

    # Method path (must delegate to the class).
    tools = CalibrationTools(rngs=nnx.Rngs(seed))
    slope, intercept = tools.platt_scaling(logits, labels, validation_logits=logits)

    assert jnp.isclose(slope, class_slope, atol=_MONOTONIC_EPS)
    assert jnp.isclose(intercept, class_intercept, atol=_MONOTONIC_EPS)


def test_isotonic_method_matches_class() -> None:
    """``isotonic_regression_calibration`` must delegate to ``IsotonicRegression``."""
    seed = 321
    num_samples = 200
    data_rngs = nnx.Rngs(seed)
    confidences = jax.random.uniform(data_rngs.sample(), (num_samples,))
    accuracies = (confidences + 0.1 * jax.random.normal(data_rngs.sample(), (num_samples,))) > 0.5

    class_regressor = IsotonicRegression(rngs=nnx.Rngs(seed))
    class_regressor.fit(confidences, accuracies)
    expected = class_regressor(confidences)

    tools = CalibrationTools(rngs=nnx.Rngs(seed))
    calibrated = tools.isotonic_regression_calibration(confidences, accuracies)

    assert calibrated.shape == confidences.shape
    assert jnp.allclose(calibrated, expected, atol=_MONOTONIC_EPS)
