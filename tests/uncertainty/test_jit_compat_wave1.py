"""JIT-compatibility regression tests for Wave-1 audit fixes.

Pins the fixes made in the Wave-1 audit cleanup:

* `aggregators._bin_calibration_stats` and the three `CalibrationAssessment`
  methods that call it must trace under ``jax.jit`` (previously they used
  Python ``if bin_count > 0`` branches on traced arrays + boolean fancy
  indexing — un-jittable).
* `aggregators.EpistemicUncertainty.compute_variance_of_expected` must
  return the actual per-sample variance, not a degenerate zero
  (previous implementation broadcast the mean back then took variance,
  giving 0 always).
* `priors_physics.PhysicsInformedPriors.compute_violation_penalty` and
  `check_physical_plausibility` must trace under ``jax.jit`` (previously
  used Python ``if jnp.any(...)`` branches on traced arrays — un-jittable).
* `priors_physics` must access `nnx.Param` values via canonical ``[...]``
  indexing — not the deprecated ``.value`` property.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# ---------------------------------------------------------------------------
# aggregators._bin_calibration_stats jit-compat (B2)
# ---------------------------------------------------------------------------


def test_calibration_assessment_methods_are_jit_compatible() -> None:
    """ECE / MCE / reliability-diagram all trace under jax.jit on the
    canonical (confidences, accuracies, n_bins=10) signature."""
    from opifex.uncertainty.aggregators import CalibrationAssessment

    confidences = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.6, 0.4, 0.1])
    accuracies = jnp.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    ca = CalibrationAssessment()

    @jax.jit
    def jitted_ece(c: jax.Array, a: jax.Array) -> jax.Array:
        # `expected_calibration_error` returns a Python float at the boundary.
        # Inside jit we let the kernel produce its scalar array result.
        from opifex.uncertainty.aggregators import _bin_calibration_stats

        boundaries = jnp.linspace(0.0, 1.0, 11)
        bin_c, bin_a, counts = _bin_calibration_stats(
            confidences=c, accuracies=a, bin_boundaries=boundaries
        )
        total = jnp.maximum(jnp.sum(counts), 1.0)
        return jnp.sum((counts / total) * jnp.abs(bin_c - bin_a))

    ece_jit = float(jitted_ece(confidences, accuracies))
    ece_eager = ca.expected_calibration_error(confidences, accuracies, n_bins=10)
    assert ece_jit == pytest.approx(ece_eager, rel=1e-5, abs=1e-6)


def test_bin_stats_zeroes_empty_bins() -> None:
    """When no samples fall in a bin, the returned stats are zero (not nan)."""
    from opifex.uncertainty.aggregators import _bin_calibration_stats

    confidences = jnp.array([0.1, 0.1, 0.1])  # all in bin 0
    accuracies = jnp.array([1.0, 1.0, 1.0])
    boundaries = jnp.linspace(0.0, 1.0, 6)  # 5 bins
    bin_c, bin_a, counts = _bin_calibration_stats(
        confidences=confidences, accuracies=accuracies, bin_boundaries=boundaries
    )
    # Bin 0 has all 3 samples; bins 1..4 are empty.
    assert int(counts[0]) == 3
    assert int(jnp.sum(counts[1:])) == 0
    assert bool(jnp.all(jnp.isfinite(bin_c)))
    assert bool(jnp.all(jnp.isfinite(bin_a)))


# ---------------------------------------------------------------------------
# EpistemicUncertainty.compute_variance_of_expected fix (B2 / M18)
# ---------------------------------------------------------------------------


def test_variance_of_expected_returns_actual_variance() -> None:
    """Previously broadcast-then-var trick returned 0; canonical fix uses
    jnp.var(predictions, axis=0) directly."""
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    rng = jax.random.key(0)
    predictions = jax.random.normal(rng, (16, 4, 1))
    var = EpistemicUncertainty.compute_variance_of_expected(predictions)
    assert var.shape == (4, 1)
    assert bool(jnp.all(var > 0))  # spread across 16 samples → nonzero variance


# ---------------------------------------------------------------------------
# priors_physics jit-compat (B3)
# ---------------------------------------------------------------------------


def test_compute_violation_penalty_is_jit_compatible() -> None:
    """`compute_violation_penalty` traces under jax.jit; non-finite inputs
    return the large finite fallback via jnp.where instead of a Python branch."""
    from opifex.uncertainty.priors_physics import PhysicsInformedPriors

    rngs = nnx.Rngs(42)
    prior = PhysicsInformedPriors(
        conservation_laws=["energy", "momentum"],
        boundary_conditions=("dirichlet",),
        rngs=rngs,
    )

    @jax.jit
    def jitted(params: jax.Array) -> jax.Array:
        return prior.compute_violation_penalty(params)

    finite_params = jnp.array([1.0, 2.0, -1.0, 0.5])
    finite_penalty = float(jitted(finite_params))
    assert finite_penalty > 0.0
    assert finite_penalty < 1e6  # not the non-finite fallback

    non_finite_params = jnp.array([jnp.nan, 1.0, jnp.inf, 0.0])
    fallback_penalty = float(jitted(non_finite_params))
    assert fallback_penalty == pytest.approx(1e6 * prior.penalty_weight)


def test_check_physical_plausibility_is_jit_compatible() -> None:
    from opifex.uncertainty.priors_physics import PhysicsInformedPriors

    rngs = nnx.Rngs(42)
    prior = PhysicsInformedPriors(conservation_laws=["positivity", "boundedness"], rngs=rngs)

    @jax.jit
    def jitted(params: jax.Array) -> jax.Array:
        return prior.check_physical_plausibility(params)

    # Finite, small magnitude → high plausibility.
    p_good = jnp.array([0.5, 1.2, 2.1, 0.3])
    # Magnitude > 1e3 (×0.1) AND any |p|>10 → boundedness penalty (×0.5) → 0.05.
    p_bad_magnitude = jnp.array([2000.0, 1.0, 1.0, 1.0])
    p_negative = jnp.array([0.5, -1.0, 0.3, 0.4])  # triggers positivity (×0.3)
    p_non_finite = jnp.array([jnp.nan, 1.0, 2.0])  # → 0

    assert float(jitted(p_good)) == pytest.approx(1.0)
    assert float(jitted(p_bad_magnitude)) == pytest.approx(0.05, abs=1e-6)
    assert float(jitted(p_negative)) == pytest.approx(0.3, abs=1e-6)
    assert float(jitted(p_non_finite)) == pytest.approx(0.0)


def test_apply_constraints_is_jit_compatible() -> None:
    """`apply_constraints` uses `self.constraint_weights[i]` (canonical NNX
    indexing) — must trace under jax.jit since traced weights are NOT
    converted to Python floats anymore."""
    from opifex.uncertainty.priors_physics import PhysicsInformedPriors

    rngs = nnx.Rngs(42)
    prior = PhysicsInformedPriors(conservation_laws=["energy", "momentum"], rngs=rngs)

    @jax.jit
    def jitted(params: jax.Array) -> jax.Array:
        return prior.apply_constraints(params)

    p = jnp.array([1.0, 2.0, 3.0, 4.0])
    out = jitted(p)
    assert out.shape == p.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_priors_physics_no_deprecated_value_property() -> None:
    """Source-level guard that .value property access has been migrated to
    canonical [...] indexing — flax NNX has deprecated `.value`."""
    import inspect

    from opifex.uncertainty import priors_physics

    source = inspect.getsource(priors_physics)
    # Allow `optax.value_and_grad_from_state` and `jax.value_and_grad` patterns
    # (those are jax/optax APIs, not Variable.value access).
    lines_with_value = [
        line.strip()
        for line in source.splitlines()
        if ".value" in line
        and "value_and_grad" not in line
        and "values()" not in line
        and not line.strip().startswith("#")
    ]
    assert not lines_with_value, (
        f"Found {len(lines_with_value)} `.value` references in priors_physics.py — "
        f"flax NNX has deprecated this property; use `[...]` indexing instead. "
        f"Lines: {lines_with_value[:3]}"
    )
