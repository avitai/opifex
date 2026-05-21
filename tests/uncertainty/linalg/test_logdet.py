"""Tests for stochastic Lanczos quadrature log-determinant estimator.

References
----------
* Ubaru, Chen, Saad 2017 — *Fast estimation of tr(f(A)) via stochastic
  Lanczos quadrature*.
* Krämer arXiv:2405.17277 — *Gradients of matrix functions in JAX*.

Sibling reference (line-by-line port): ``matfree/matfree/funm.py:178
integrand_funm_sym_logdet`` plus the Hutchinson-style outer averaging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import slq_logdet


def test_slq_logdet_matches_exact_logdet_on_diagonal_spd_matrix() -> None:
    """SLQ converges to ``sum(log(eigvals))`` within a variance bound.

    Cite: Ubaru+ 2017. The estimator is unbiased; with a diagonal matrix
    the Lanczos basis is exact after one step per probe and the variance
    over Rademacher probes vanishes for diagonal A.
    """
    eigenvalues = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = jnp.diag(eigenvalues)
    exact_logdet = jnp.sum(jnp.log(eigenvalues))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    estimate = slq_logdet(
        matvec=matvec,
        dim=5,
        num_samples=32,
        num_matvecs=5,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(estimate, exact_logdet, atol=1e-1)


def test_slq_logdet_is_zero_on_identity_matrix() -> None:
    """``log det(I) = 0`` is the trivial baseline."""
    identity = jnp.eye(4)

    def matvec(vector: jax.Array) -> jax.Array:
        return identity @ vector

    estimate = slq_logdet(
        matvec=matvec,
        dim=4,
        num_samples=16,
        num_matvecs=4,
        key=jax.random.PRNGKey(1),
    )
    assert jnp.allclose(estimate, 0.0, atol=1e-4)


def test_slq_logdet_converges_to_dense_value_on_random_spd_matrix() -> None:
    """SLQ converges to ``slogdet`` over many probes on a random SPD matrix."""
    rng = jax.random.PRNGKey(2)
    raw = jax.random.normal(rng, (6, 6))
    matrix = raw @ raw.T + 1.0 * jnp.eye(6)
    sign, exact_logdet = jnp.linalg.slogdet(matrix)
    assert sign > 0  # sanity

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    estimate = slq_logdet(
        matvec=matvec,
        dim=6,
        num_samples=256,
        num_matvecs=6,
        key=jax.random.PRNGKey(3),
    )
    assert jnp.allclose(estimate, exact_logdet, atol=2.0)


def test_slq_logdet_is_jit_compatible() -> None:
    """SLQ compiles under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda k: slq_logdet(matvec=matvec, dim=3, num_samples=8, num_matvecs=3, key=k)
    )
    estimate = jitted(jax.random.PRNGKey(4))
    assert jnp.isfinite(estimate)
