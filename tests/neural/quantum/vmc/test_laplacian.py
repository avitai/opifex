"""Tests for the kinetic-energy Laplacian operators.

The native forward-Laplacian must agree with the obviously-correct
``jvp``-over-``grad`` oracle to ~1e-6 on a random ansatz, and both must be
``jit`` clean. They compute ``laplacian(log|psi|)`` and ``|grad log|psi||^2``,
the two terms of the local kinetic energy.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.laplacian import (
    forward_laplacian,
    jvp_grad_laplacian,
)


def _random_scalar_fn(in_dim: int) -> Callable[[jax.Array], jax.Array]:
    """A smooth non-separable scalar function R^{in_dim} -> R for testing."""
    key = jax.random.PRNGKey(11)
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = jax.random.normal(k1, (in_dim, 12), dtype=jnp.float64)
    w2 = jax.random.normal(k2, (12, 1), dtype=jnp.float64)
    b1 = jax.random.normal(k3, (12,), dtype=jnp.float64)

    def fn(x: jax.Array) -> jax.Array:
        flat = x.reshape(-1)
        hidden = jnp.tanh(flat @ w1 + b1)
        return jnp.sum(jnp.tanh(hidden @ w2))

    return fn


def _dense_laplacian(fn: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
    """Reference Laplacian via the trace of the dense Hessian."""
    hessian = jax.hessian(fn)(x.reshape(-1))
    return jnp.trace(hessian)


def test_oracle_matches_dense_hessian_trace() -> None:
    """The jvp-over-grad oracle reproduces the dense Hessian trace."""
    x = jax.random.normal(jax.random.PRNGKey(12), (2, 3), dtype=jnp.float64)
    fn = _random_scalar_fn(x.size)
    _, lap, grad = jvp_grad_laplacian(fn, x)
    np.testing.assert_allclose(lap, _dense_laplacian(fn, x), atol=1e-8)
    np.testing.assert_allclose(grad, jax.grad(fn)(x), atol=1e-8)


def test_forward_laplacian_matches_oracle() -> None:
    """The native forward-Laplacian agrees with the oracle to 1e-6."""
    x = jax.random.normal(jax.random.PRNGKey(13), (2, 3), dtype=jnp.float64)
    fn = _random_scalar_fn(x.size)
    val_o, lap_o, grad_o = jvp_grad_laplacian(fn, x)
    val_f, lap_f, grad_f = forward_laplacian(fn, x)
    np.testing.assert_allclose(val_f, val_o, atol=1e-6)
    np.testing.assert_allclose(lap_f, lap_o, atol=1e-6)
    np.testing.assert_allclose(grad_f, grad_o, atol=1e-6)


def test_forward_laplacian_matches_dense_hessian_trace() -> None:
    """The forward-Laplacian reproduces the dense Hessian trace directly."""
    x = jax.random.normal(jax.random.PRNGKey(14), (3, 3), dtype=jnp.float64)
    fn = _random_scalar_fn(x.size)
    _, lap, _ = forward_laplacian(fn, x)
    np.testing.assert_allclose(lap, _dense_laplacian(fn, x), atol=1e-6)


def test_laplacians_are_jit_clean() -> None:
    """Both Laplacian operators run under ``jit``."""
    x = jax.random.normal(jax.random.PRNGKey(15), (2, 3), dtype=jnp.float64)
    fn = _random_scalar_fn(x.size)
    jit_oracle = jax.jit(lambda v: jvp_grad_laplacian(fn, v)[1])
    jit_fwd = jax.jit(lambda v: forward_laplacian(fn, v)[1])
    np.testing.assert_allclose(jit_oracle(x), jit_fwd(x), atol=1e-6)
