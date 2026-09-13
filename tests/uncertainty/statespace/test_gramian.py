"""Tests for the batched exponential-and-Gramian discretisation.

``exponential_and_gramian`` discretises one LTI SDE at a sequence of steps. It always applies 16
doublings and applies 16 more, under one ``lax.cond`` for the whole sequence, only when some step
needs them; steps needing more than 32 doublings are NaN. These tests pin that the batching and the
conditional block change no value, that NaN is confined to the offending step, and that gradients
survive a zero step, whose Gramian factor is the zero matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty.statespace._gramian import exponential_and_gramian
from tests.uncertainty.statespace._process_noise_references import REFERENCES


def _matern72_inputs() -> tuple[jax.Array, jax.Array, float]:
    """Return ``(F, B, lambda)`` of the unit Matern-7/2 SDE, with ``B B^T = L Q_c L^T``."""
    reference = REFERENCES["matern72"]
    drift = jnp.asarray(reference.drift, dtype=jnp.float32)
    factor = jnp.asarray(reference.dispersion, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(reference.diffusion[0][0], dtype=jnp.float32)
    )
    return drift, factor, float(np.sqrt(7.0))


def test_sequence_matches_each_step_discretised_alone() -> None:
    """Batching, including a step that triggers the conditional block, changes no value."""
    drift, factor, rate = _matern72_inputs()
    steps = jnp.asarray([1e-4, 1e-2, 1.0, 1e2, 1e4], dtype=jnp.float32) / rate
    transitions, process_noises = jax.jit(exponential_and_gramian)(drift, factor, steps)
    for index in range(steps.shape[0]):
        alone_transition, alone_noise = jax.jit(exponential_and_gramian)(
            drift, factor, steps[index : index + 1]
        )
        np.testing.assert_array_equal(
            np.asarray(transitions[index]), np.asarray(alone_transition[0])
        )
        np.testing.assert_array_equal(np.asarray(process_noises[index]), np.asarray(alone_noise[0]))


def test_nan_is_confined_to_the_step_beyond_the_doubling_limit() -> None:
    """A step needing more than 32 doublings is NaN; its neighbours are unaffected."""
    drift, factor, _ = _matern72_inputs()
    steps = jnp.asarray([0.5, 1e12, 2.0], dtype=jnp.float32)
    transitions, process_noises = exponential_and_gramian(drift, factor, steps)
    assert bool(jnp.all(jnp.isnan(transitions[1])))
    assert bool(jnp.all(jnp.isnan(process_noises[1])))
    for index in (0, 2):
        assert bool(jnp.all(jnp.isfinite(transitions[index])))
        assert bool(jnp.all(jnp.isfinite(process_noises[index])))


def test_gradients_are_finite_through_a_zero_step() -> None:
    """The first step of a Kalman sequence is ``dt = 0``; the drift gradient stays finite."""
    drift, factor, _ = _matern72_inputs()
    steps = jnp.asarray([0.0, 0.5, 2.0], dtype=jnp.float32)

    def total_noise(drift_matrix: jax.Array) -> jax.Array:
        _, process_noises = exponential_and_gramian(drift_matrix, factor, steps)
        return jnp.sum(process_noises)

    gradient = jax.grad(total_noise)(drift)
    assert bool(jnp.all(jnp.isfinite(gradient)))
