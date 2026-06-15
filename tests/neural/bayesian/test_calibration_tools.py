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
    _gradient_descent,
    CalibrationTools,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
)


_MONOTONIC_EPS = 1e-5

# Characterisation tolerance for the gradient-descent dedup (Task 12.3.8).
# The shared ``_gradient_descent`` helper must reproduce the byte-for-byte
# fitted values of the four hand-rolled SGD loops it replaces, so the
# tolerance is as tight as float32 round-trips allow.
_FIT_EQUIVALENCE_ATOL = 1e-5


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


# ---------------------------------------------------------------------------
# Task 12.3.8 — shared fixed-iteration gradient-descent loop.
#
# The four hand-rolled ``for _ in range(N): g = grad(loss)(p); p -= lr*g; ...``
# loops (simple/adaptive temperature scaling, simple/adaptive physics-aware
# temperature scaling, and Platt fitting) are replaced by a single
# ``_gradient_descent`` helper. These characterisation tests pin the exact
# fitted scalars produced by the pre-refactor loops so the dedup is proven
# behaviour-preserving (golden values were captured from HEAD before the
# helper existed).
# ---------------------------------------------------------------------------


def _temperature_logits_labels() -> tuple[jax.Array, jax.Array]:
    """Fixed (logits, labels) for the temperature-scaling characterisation."""
    logits = jax.random.normal(jax.random.PRNGKey(1), (64, 3))
    labels = jax.random.randint(jax.random.PRNGKey(2), (64,), 0, 3)
    return logits, labels


def _physics_predictions_targets_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fixed (predictions, targets, inputs) for the physics characterisation."""
    inputs = jax.random.uniform(jax.random.PRNGKey(3), (40, 2))
    predictions = jax.random.normal(jax.random.PRNGKey(4), (40, 1))
    targets = jnp.abs(jax.random.normal(jax.random.PRNGKey(5), (40, 1)))
    return predictions, targets, inputs


def test_optimize_temperature_simple_matches_golden() -> None:
    """Plain-SGD temperature scaling reproduces its pre-dedup fitted value."""
    logits, labels = _temperature_logits_labels()
    calibrator = TemperatureScaling(rngs=nnx.Rngs(0))

    fitted = calibrator.optimize_temperature(logits, labels)

    assert jnp.isclose(fitted, 1.2994803190231323, atol=_FIT_EQUIVALENCE_ATOL)


def test_optimize_temperature_adaptive_matches_golden() -> None:
    """Momentum-SGD temperature scaling reproduces its pre-dedup fitted value."""
    logits, labels = _temperature_logits_labels()
    calibrator = TemperatureScaling(adaptive=True, rngs=nnx.Rngs(0))

    fitted = calibrator.optimize_temperature(logits, labels)

    assert jnp.isclose(fitted, 2.5726025104522705, atol=_FIT_EQUIVALENCE_ATOL)


def test_optimize_temperature_physics_simple_matches_golden() -> None:
    """Plain-SGD physics-aware scaling reproduces value and history length."""
    predictions, targets, inputs = _physics_predictions_targets_inputs()
    calibrator = TemperatureScaling(physics_constraints=("positivity",), rngs=nnx.Rngs(0))

    fitted = calibrator.optimize_temperature_with_physics_constraints(predictions, targets, inputs)

    assert jnp.isclose(fitted, 1.4705876111984253, atol=_FIT_EQUIVALENCE_ATOL)
    # The 100-step loop appends one penalty per step; history stays bounded.
    assert len(calibrator.constraint_penalty_history) == 100


def test_optimize_temperature_physics_adaptive_matches_golden() -> None:
    """Momentum-SGD physics-aware scaling reproduces value and bounded history."""
    predictions, targets, inputs = _physics_predictions_targets_inputs()
    calibrator = TemperatureScaling(
        physics_constraints=("positivity", "boundedness"),
        adaptive=True,
        constraint_strength=0.5,
        rngs=nnx.Rngs(0),
    )

    fitted = calibrator.optimize_temperature_with_physics_constraints(predictions, targets, inputs)

    assert jnp.isclose(fitted, 1.5693259239196777, atol=_FIT_EQUIVALENCE_ATOL)
    # The 150-step loop truncates the penalty history to the last 100 entries.
    assert len(calibrator.constraint_penalty_history) == 100


def test_platt_fit_matches_golden() -> None:
    """Two-parameter Platt SGD reproduces its pre-dedup (a, b) fit."""
    logits = jax.random.normal(jax.random.PRNGKey(6), (100,))
    labels = (logits > 0).astype(jnp.float32)
    scaler = PlattScaling(rngs=nnx.Rngs(0))

    scaler.fit(logits, labels)

    assert jnp.isclose(float(scaler.a[...]), -0.4867066740989685, atol=_FIT_EQUIVALENCE_ATOL)
    assert jnp.isclose(float(scaler.b[...]), 0.07027804851531982, atol=_FIT_EQUIVALENCE_ATOL)


def test_gradient_descent_helper_minimises_quadratic() -> None:
    """The shared helper drives a convex scalar loss toward its minimum."""

    def loss_fn(x: jax.Array) -> jax.Array:
        return (x - 3.0) ** 2

    fitted = _gradient_descent(loss_fn, jnp.asarray(0.0), n_steps=200, lr=0.1)

    assert jnp.isclose(fitted, 3.0, atol=1e-3)


def test_gradient_descent_helper_applies_projection() -> None:
    """A projection callback constrains every iterate (here, a lower bound)."""

    def loss_fn(x: jax.Array) -> jax.Array:
        # Minimiser at x = -5, but the projection floors iterates at 0.5.
        return (x + 5.0) ** 2

    fitted = _gradient_descent(
        loss_fn,
        jnp.asarray(1.0),
        n_steps=100,
        lr=0.1,
        project=lambda x: jnp.maximum(x, 0.5),
    )

    assert float(fitted) >= 0.5


def test_gradient_descent_helper_is_jit_compatible() -> None:
    """The pure-JAX helper composes under ``jax.jit`` with identical output."""

    def loss_fn(x: jax.Array) -> jax.Array:
        return (x - 2.0) ** 2

    def run(init: jax.Array) -> jax.Array:
        return _gradient_descent(loss_fn, init, n_steps=50, lr=0.1)

    eager = run(jnp.asarray(0.0))
    jitted = jax.jit(run)(jnp.asarray(0.0))

    assert bool(jnp.allclose(eager, jitted, rtol=1e-6, atol=1e-7))
