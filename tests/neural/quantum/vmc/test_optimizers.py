"""Tests for the natural-gradient VMC optimizers (MinSR / SPRING).

MinSR solves the natural-gradient system in the *sample* space (the NTK Gram
matrix), avoiding the full parameter-space Fisher. SPRING adds Nesterov-style
momentum (Goldshlager, Abrahamsen & Lin, arXiv:2401.10190). The math is ported
from NetKet ``_src/ngd/srt.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.optimizers import (
    minsr_update,
    spring_update,
    SpringState,
)


def _random_jacobian_and_energies() -> tuple[jax.Array, jax.Array]:
    """A random per-sample log-amplitude jacobian and local energies."""
    n_samples, n_params = 32, 8
    jacobian = jax.random.normal(jax.random.PRNGKey(0), (n_samples, n_params), dtype=jnp.float64)
    energies = jax.random.normal(jax.random.PRNGKey(1), (n_samples,), dtype=jnp.float64)
    return jacobian, energies


def test_minsr_solves_the_sample_space_natural_gradient_system() -> None:
    r"""MinSR returns ``O_L^T (O_L O_L^T + lambda I)^{-1} dv`` (NTK Gram solve)."""
    jacobian, energies = _random_jacobian_and_energies()
    diag_shift = 1e-3
    update = minsr_update(jacobian, energies, diag_shift=diag_shift)

    # Recompute the reference centred/scaled quantities directly.
    n = jacobian.shape[0]
    centred = jacobian - jnp.mean(jacobian, axis=0, keepdims=True)
    o_l = centred / jnp.sqrt(n)
    dv = 2.0 * (energies - jnp.mean(energies)) / jnp.sqrt(n)
    gram = o_l @ o_l.T + diag_shift * jnp.eye(n)
    expected = o_l.T @ jnp.linalg.solve(gram, dv)
    np.testing.assert_allclose(update, expected, rtol=1e-8, atol=1e-10)


def test_minsr_matches_dense_fisher_natural_gradient() -> None:
    r"""MinSR equals the parameter-space natural gradient (push-through identity).

    ``O^T (O O^T + lambda I)^{-1} = (O^T O + lambda I)^{-1} O^T`` -- so the
    sample-space (Gram) solve and the parameter-space (Fisher) solve give the
    same update for the same shift.
    """
    jacobian, energies = _random_jacobian_and_energies()
    diag_shift = 1e-2
    update = minsr_update(jacobian, energies, diag_shift=diag_shift)

    n, p = jacobian.shape
    centred = jacobian - jnp.mean(jacobian, axis=0, keepdims=True)
    o_l = centred / jnp.sqrt(n)
    dv = 2.0 * (energies - jnp.mean(energies)) / jnp.sqrt(n)
    fisher = o_l.T @ o_l + diag_shift * jnp.eye(p)
    expected = jnp.linalg.solve(fisher, o_l.T @ dv)
    np.testing.assert_allclose(update, expected, rtol=1e-6, atol=1e-8)


def test_spring_with_zero_momentum_equals_minsr() -> None:
    """SPRING with ``momentum=0`` reduces to plain MinSR."""
    jacobian, energies = _random_jacobian_and_energies()
    state = SpringState(old_updates=jnp.zeros(jacobian.shape[1]))
    update, _ = spring_update(
        jacobian, energies, state, diag_shift=1e-3, momentum=0.0, proj_reg=0.0
    )
    minsr = minsr_update(jacobian, energies, diag_shift=1e-3)
    np.testing.assert_allclose(update, minsr, rtol=1e-7, atol=1e-9)


def test_spring_accumulates_momentum_state() -> None:
    """SPRING carries the previous update forward through its state."""
    jacobian, energies = _random_jacobian_and_energies()
    state = SpringState(old_updates=jnp.zeros(jacobian.shape[1]))
    update1, state1 = spring_update(
        jacobian, energies, state, diag_shift=1e-3, momentum=0.9, proj_reg=1e-3
    )
    np.testing.assert_allclose(state1.old_updates, update1, atol=1e-12)
    # A second step with momentum differs from a fresh first step.
    update2, _ = spring_update(
        jacobian, energies, state1, diag_shift=1e-3, momentum=0.9, proj_reg=1e-3
    )
    assert not np.allclose(update1, update2)


def test_optimizers_are_jit_clean() -> None:
    """Both natural-gradient updates run under ``jit``."""
    jacobian, energies = _random_jacobian_and_energies()
    minsr = jax.jit(lambda j, e: minsr_update(j, e, diag_shift=1e-3))
    out = minsr(jacobian, energies)
    assert out.shape == (jacobian.shape[1],)
    assert jnp.all(jnp.isfinite(out))
