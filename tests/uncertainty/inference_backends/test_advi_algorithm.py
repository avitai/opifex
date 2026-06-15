"""Tests for the mean-field ADVI algorithm primitives.

Tests for :mod:`opifex.uncertainty.inference_backends._advi_algorithm`.

Regression guard for R5 (immutability / no ``field()`` as a function
default): :func:`step` must default ``objective`` to a real ``KL()``
instance when called without the keyword, rather than leaking a
``dataclasses.Field`` sentinel into the objective dispatch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from opifex.uncertainty.inference_backends._advi_algorithm import (
    init,
    KL,
    step,
)


def _standard_normal_log_density(position: jax.Array) -> jax.Array:
    """``log N(x; 0, I)`` up to an additive constant."""
    return -0.5 * jnp.sum(position**2)


def test_step_without_objective_uses_real_kl_instance() -> None:
    """``step`` called without ``objective=`` must default to ``KL()``.

    Previously the default was ``field(default_factory=KL)`` so the
    parameter held a ``dataclasses.Field`` sentinel, which the objective
    dispatch rejects with ``TypeError``. A correct default makes the step
    run and return a finite ELBO.
    """
    optimizer = optax.adam(1e-2)
    initial_state = init(jnp.zeros((3,)), optimizer)

    new_state, info = step(
        jax.random.key(0),
        initial_state,
        _standard_normal_log_density,
        optimizer,
        num_samples=8,
    )

    assert jnp.isfinite(info.elbo)
    assert new_state.mu.shape == (3,)


def test_step_default_objective_matches_explicit_kl() -> None:
    """Omitting ``objective`` is equivalent to passing ``KL()`` explicitly."""
    optimizer = optax.adam(1e-2)
    initial_state = init(jnp.zeros((3,)), optimizer)
    key = jax.random.key(1)

    _, info_default = step(
        key,
        initial_state,
        _standard_normal_log_density,
        optimizer,
        num_samples=8,
    )
    _, info_explicit = step(
        key,
        initial_state,
        _standard_normal_log_density,
        optimizer,
        num_samples=8,
        objective=KL(),
    )

    assert jnp.allclose(info_default.elbo, info_explicit.elbo)
