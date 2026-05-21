"""Tests for higher-moment stochastic trace UQ.

Sibling reference: ``matfree/matfree/stochtrace.py::integrand_wrap_moments``
wraps any Hutchinson-style integrand to compute multiple raw moments of
the per-probe quadratic-form distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import trace_moments


def test_trace_moments_returns_first_moment_equals_hutchinson_trace() -> None:
    """The first moment equals the Hutchinson trace estimate."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    moments = trace_moments(
        matvec=matvec,
        dim=4,
        num_samples=512,
        powers=(1,),
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(moments[0], jnp.trace(matrix), atol=1e-5)


def test_trace_moments_variance_is_zero_on_diagonal_matrix() -> None:
    """For Rademacher probes the per-probe trace estimator has zero variance on diag(A).

    Cite: Hutchinson 1990. ``Var(v^T A v) = 2 * (||A||_F^2 - ||diag(A)||^2)``
    for Rademacher probes, which vanishes for diagonal ``A``. The second
    raw moment equals ``mean^2`` and the variance (second central moment)
    is zero.
    """
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    raw_moments = trace_moments(
        matvec=matvec,
        dim=4,
        num_samples=256,
        powers=(1, 2),
        key=jax.random.PRNGKey(1),
    )
    mean = raw_moments[0]
    second_moment = raw_moments[1]
    variance = second_moment - mean**2
    assert jnp.allclose(variance, 0.0, atol=1e-3)


def test_trace_moments_has_positive_variance_on_dense_matrix() -> None:
    """For a non-diagonal symmetric matrix the variance is positive."""
    rng = jax.random.PRNGKey(2)
    raw = jax.random.normal(rng, (6, 6))
    matrix = 0.5 * (raw + raw.T)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    raw_moments = trace_moments(
        matvec=matvec,
        dim=6,
        num_samples=256,
        powers=(1, 2),
        key=jax.random.PRNGKey(3),
    )
    variance = raw_moments[1] - raw_moments[0] ** 2
    assert variance > 1e-2


def test_trace_moments_is_jit_compatible() -> None:
    """``trace_moments`` compiles under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda k: trace_moments(
            matvec=matvec, dim=3, num_samples=64, powers=(1, 2, 3), key=k
        )
    )
    moments = jitted(jax.random.PRNGKey(4))
    assert len(moments) == 3
    assert all(jnp.isfinite(m) for m in moments)
