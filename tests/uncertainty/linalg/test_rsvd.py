"""Tests for randomised SVD (Halko-Martinsson-Tropp).

References
----------
* Halko, Martinsson, Tropp arXiv:0909.4061 — *Finding structure with
  randomness*. Algorithm 5.1 is the canonical randomised SVD.

Sibling reference: ``matfree`` provides Lanczos-bidiag-based partial SVD
in ``matfree/matfree/eig.py:svd_partial``; HMT-style randomised SVD with
subspace iteration is NOT in matfree (cola has a stub in ``tbd/``).
opifex implements it directly from the paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import randomized_svd


def test_randomized_svd_recovers_singular_values_of_low_rank_matrix() -> None:
    """HMT randomised SVD recovers the top-k singular values of a rank-k matrix.

    Cite: Halko+ arXiv:0909.4061. For an exact rank-k matrix the
    randomised range finder captures the row space and the subsequent
    small SVD returns the exact singular triplets.
    """
    rng = jax.random.PRNGKey(0)
    left = jax.random.normal(rng, (8, 3))
    right = jax.random.normal(jax.random.PRNGKey(1), (3, 6))
    matrix = left @ right  # rank 3 in 8x6

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    left_factor, singvals, right_factor = randomized_svd(
        matvec=matvec,
        matvec_transpose=matvec_t,
        dim_rows=8,
        dim_cols=6,
        rank=3,
        oversampling=5,
        num_iterations=2,
        key=jax.random.PRNGKey(2),
    )

    reference_svals = jnp.sort(jnp.linalg.svd(matrix, compute_uv=False))[-3:]
    estimated_svals = jnp.sort(singvals)[-3:]
    # Atol calibrated for float32 randomised SVD with 2 subspace iters.
    assert jnp.allclose(estimated_svals, reference_svals, atol=1e-2)
    assert left_factor.shape == (8, 3)
    assert right_factor.shape == (6, 3)


def test_randomized_svd_subspace_iteration_reduces_error() -> None:
    """Subspace iteration improves accuracy on matrices with slow spectral decay.

    Cite: Halko+ arXiv:0909.4061 §4.5. Without subspace iteration the
    range finder may miss directions associated with small singular
    values; ``num_iterations >= 1`` recovers them.
    """
    eigenvalues = jnp.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    rng = jax.random.PRNGKey(11)
    raw = jax.random.normal(rng, (6, 6))
    orthogonal, _ = jnp.linalg.qr(raw)
    matrix = orthogonal @ jnp.diag(eigenvalues) @ orthogonal.T

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    _, singvals_zero_iter, _ = randomized_svd(
        matvec=matvec,
        matvec_transpose=matvec_t,
        dim_rows=6,
        dim_cols=6,
        rank=3,
        oversampling=2,
        num_iterations=0,
        key=jax.random.PRNGKey(13),
    )
    _, singvals_two_iter, _ = randomized_svd(
        matvec=matvec,
        matvec_transpose=matvec_t,
        dim_rows=6,
        dim_cols=6,
        rank=3,
        oversampling=2,
        num_iterations=2,
        key=jax.random.PRNGKey(13),
    )

    reference_top3 = jnp.sort(eigenvalues)[-3:]
    error_zero = jnp.linalg.norm(jnp.sort(singvals_zero_iter) - reference_top3)
    error_two = jnp.linalg.norm(jnp.sort(singvals_two_iter) - reference_top3)
    assert error_two <= error_zero + 1e-6


def test_randomized_svd_is_jit_compatible() -> None:
    """The HMT chain compiles under ``jax.jit``."""
    rng = jax.random.PRNGKey(21)
    matrix = jax.random.normal(rng, (5, 4))

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    def call(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return randomized_svd(
            matvec=matvec,
            matvec_transpose=matvec_t,
            dim_rows=5,
            dim_cols=4,
            rank=2,
            oversampling=1,
            num_iterations=1,
            key=key,
        )

    jitted = jax.jit(call)
    left, sv, right = jitted(jax.random.PRNGKey(22))
    assert left.shape == (5, 2)
    assert sv.shape == (2,)
    assert right.shape == (4, 2)
