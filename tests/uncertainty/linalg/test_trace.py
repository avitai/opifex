"""Tests for matrix-free trace estimators.

References
----------
* Hutchinson 1990 — A stochastic estimator of the trace of the influence matrix.
* Meyer et al. arXiv:2010.09649 — Hutch++.
* Epperly et al. arXiv:2301.07825 — XTrace + XNysTrace.

Sibling reference implementations under ``/mnt/ssd2/Works/`` (not opifex deps):
``matfree/stochtrace.py`` (Hutchinson + Rademacher sampler),
``traceax/src/traceax/_estimators.py`` (Hutch++, XTrace, XNysTrace).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import hutch_plus_plus_trace, hutchinson_trace


def test_hutchinson_trace_converges_to_exact_trace_on_small_spd_matrix() -> None:
    """Hutchinson estimator converges to ``jnp.trace`` within a variance bound.

    Cite: Hutchinson 1990. The estimator is unbiased with variance scaling
    as ``2 * (||A||_F^2 - sum(diag(A)^2)) / num_samples`` for Rademacher
    probes; we check the empirical mean is within 3 standard errors.
    """
    diag_values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = jnp.diag(diag_values)
    exact_trace = jnp.trace(matrix)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    estimate = hutchinson_trace(
        matvec=matvec,
        dim=5,
        num_samples=4096,
        key=jax.random.PRNGKey(0),
    )
    # Diagonal matrix has off-diagonal Frobenius norm 0, so variance is 0.
    assert jnp.allclose(estimate, exact_trace, atol=1e-5)


def test_hutchinson_trace_is_jit_compatible() -> None:
    """The estimator must be jittable so it can live in trained loops."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(lambda key: hutchinson_trace(matvec=matvec, dim=3, num_samples=512, key=key))
    estimate = jitted(jax.random.PRNGKey(1))
    assert jnp.allclose(estimate, jnp.trace(matrix), atol=1e-5)


def test_hutchinson_trace_unbiased_on_dense_symmetric_matrix() -> None:
    """Averaging across independent runs reduces variance toward zero."""
    rng = jax.random.PRNGKey(42)
    a = jax.random.normal(rng, (8, 8))
    matrix = 0.5 * (a + a.T)
    exact_trace = jnp.trace(matrix)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    keys = jax.random.split(jax.random.PRNGKey(7), 32)
    estimates = jax.vmap(lambda k: hutchinson_trace(matvec=matvec, dim=8, num_samples=512, key=k))(
        keys
    )
    average_estimate = jnp.mean(estimates)
    assert jnp.allclose(average_estimate, exact_trace, atol=1e-1)


def test_hutch_plus_plus_beats_hutchinson_variance_on_rank_decaying_matrix() -> None:
    """Hutch++ exploits leading subspace for lower variance than Hutchinson.

    Cite: Meyer et al. arXiv:2010.09649. On a matrix with rapidly-decaying
    spectrum, the leading eigenspace captures most of the trace and Hutch++
    recovers it exactly; only the residual contributes stochastic variance.
    """
    eigenvalues = jnp.asarray([10.0, 5.0, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    rng = jax.random.PRNGKey(123)
    raw = jax.random.normal(rng, (8, 8))
    orthogonal, _ = jnp.linalg.qr(raw)
    matrix = orthogonal @ jnp.diag(eigenvalues) @ orthogonal.T
    exact_trace = jnp.trace(matrix)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    keys = jax.random.split(jax.random.PRNGKey(99), 16)

    hutchinson_estimates = jax.vmap(
        lambda k: hutchinson_trace(matvec=matvec, dim=8, num_samples=48, key=k)
    )(keys)
    hutch_pp_estimates = jax.vmap(
        lambda k: hutch_plus_plus_trace(matvec=matvec, dim=8, num_samples=48, key=k)
    )(keys)

    hutchinson_variance = jnp.var(hutchinson_estimates)
    hutch_pp_variance = jnp.var(hutch_pp_estimates)
    assert hutch_pp_variance < hutchinson_variance
    assert jnp.allclose(jnp.mean(hutch_pp_estimates), exact_trace, atol=1e-1)


def test_hutch_plus_plus_is_jit_compatible() -> None:
    """The Hutch++ estimator passes ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda key: hutch_plus_plus_trace(matvec=matvec, dim=4, num_samples=24, key=key)
    )
    estimate = jitted(jax.random.PRNGKey(2))
    assert jnp.isfinite(estimate)
