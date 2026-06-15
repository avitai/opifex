"""Tests for the FermiNet-core generalized-Slater wavefunction ansatz.

The ansatz exposes a single-walker ``log|psi|`` / ``sign`` evaluation that must
be antisymmetric under same-spin electron exchange and fully ``jit`` / ``grad``
/ ``vmap`` clean. Exact energies are validated in :mod:`test_energy`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.quantum.vmc.wavefunctions import FermiNet


def _make_h2_ansatz(*, full_det: bool = True) -> tuple[FermiNet, jax.Array]:
    """Return a small FermiNet for H2 and a single-walker electron config."""
    atoms = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    ansatz = FermiNet(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        hidden_one=(16, 16),
        hidden_two=(8, 8),
        determinants=2,
        full_det=full_det,
        rngs=nnx.Rngs(0),
    )
    positions = jax.random.normal(jax.random.PRNGKey(5), (2, 3), dtype=jnp.float64)
    return ansatz, positions


def test_log_psi_returns_finite_scalar_sign_and_logabs() -> None:
    """A single-walker evaluation returns a finite scalar ``(sign, log|psi|)``."""
    ansatz, positions = _make_h2_ansatz()
    sign, log_abs = ansatz(positions)
    assert sign.shape == ()
    assert log_abs.shape == ()
    assert jnp.isfinite(log_abs)
    assert jnp.abs(jnp.abs(sign) - 1.0) < 1e-10


def test_antisymmetry_under_same_spin_exchange() -> None:
    """Swapping two same-spin electrons flips the sign and preserves ``log|psi|``."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([3.0])
    ansatz = FermiNet(
        nspins=(2, 0),
        atoms=atoms,
        charges=charges,
        hidden_one=(16, 16),
        hidden_two=(8, 8),
        determinants=2,
        full_det=True,
        rngs=nnx.Rngs(1),
    )
    positions = jax.random.normal(jax.random.PRNGKey(6), (2, 3), dtype=jnp.float64)
    sign_a, log_a = ansatz(positions)
    swapped = positions[jnp.array([1, 0])]
    sign_b, log_b = ansatz(swapped)
    np.testing.assert_allclose(log_a, log_b, atol=1e-9)
    np.testing.assert_allclose(sign_a, -sign_b, atol=1e-9)


def test_log_psi_is_jit_and_grad_clean() -> None:
    """The log-magnitude is differentiable under ``jit``."""
    ansatz, positions = _make_h2_ansatz()
    graphdef, state = nnx.split(ansatz)

    def log_abs(pos: jax.Array) -> jax.Array:
        model = nnx.merge(graphdef, state)
        return model(pos)[1]

    grad = jax.jit(jax.grad(log_abs))(positions)
    assert grad.shape == positions.shape
    assert jnp.all(jnp.isfinite(grad))


def test_log_psi_vmaps_over_walkers() -> None:
    """The single-walker ansatz vmaps cleanly over a batch of walkers."""
    ansatz, _ = _make_h2_ansatz()
    graphdef, state = nnx.split(ansatz)
    walkers = jax.random.normal(jax.random.PRNGKey(7), (32, 2, 3), dtype=jnp.float64)

    def log_abs(pos: jax.Array) -> jax.Array:
        return nnx.merge(graphdef, state)(pos)[1]

    out = jax.vmap(log_abs)(walkers)
    assert out.shape == (32,)
    assert jnp.all(jnp.isfinite(out))


def test_spin_factored_determinant_runs() -> None:
    """The block-diagonal (``full_det=False``) path also produces finite output."""
    ansatz, positions = _make_h2_ansatz(full_det=False)
    _, log_abs = ansatz(positions)
    assert jnp.isfinite(log_abs)
