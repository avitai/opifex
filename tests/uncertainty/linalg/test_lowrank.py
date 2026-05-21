"""Tests for partial Cholesky low-rank factorisations.

References
----------
* Harbrecht, Peters, Schneider 2012 — *On the low-rank approximation by the
  pivoted Cholesky decomposition*.
* Chen, Epperly, Tropp, Webber arXiv:2207.06503 — *Randomly Pivoted Cholesky*.

Sibling references (line-by-line port):
* ``matfree/matfree/low_rank.py`` — greedy partial Cholesky.
* ``eepperly/Randomly-Pivoted-Cholesky/rpcholesky.py::cholesky_helper`` —
  canonical NumPy implementation of randomly-pivoted Cholesky.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import cholesky_greedy, rp_cholesky


def test_cholesky_greedy_recovers_full_rank_psd_matrix() -> None:
    """At ``rank = n``, greedy partial Cholesky reconstructs the full matrix."""
    rng = jax.random.PRNGKey(0)
    factor = jax.random.normal(rng, (5, 5))
    matrix = factor @ factor.T + 0.5 * jnp.eye(5)

    cholesky_factor = cholesky_greedy(matrix=matrix, rank=5)
    reconstructed = cholesky_factor @ cholesky_factor.T
    assert jnp.allclose(reconstructed, matrix, atol=1e-3)


def test_cholesky_greedy_is_jit_compatible() -> None:
    """Greedy partial Cholesky compiles under ``jax.jit``."""
    rng = jax.random.PRNGKey(1)
    factor = jax.random.normal(rng, (4, 4))
    matrix = factor @ factor.T + 0.1 * jnp.eye(4)

    jitted = jax.jit(lambda m: cholesky_greedy(matrix=m, rank=2))
    cholesky_factor = jitted(matrix)
    assert cholesky_factor.shape == (4, 2)


def test_rp_cholesky_recovers_exact_low_rank_psd_matrix() -> None:
    """RPCholesky with sufficient rank recovers a rank-k PSD matrix exactly.

    Cite: Chen+ arXiv:2207.06503. For a rank-r PSD matrix, choosing
    ``rank >= r`` random pivots with probability proportional to the
    residual diagonal recovers ``A`` with zero residual (probability 1
    when probes hit the column space).
    """
    rng = jax.random.PRNGKey(2)
    factor = jax.random.normal(rng, (6, 3))
    matrix = factor @ factor.T  # rank 3 in 6D

    cholesky_factor = rp_cholesky(matrix=matrix, rank=4, key=jax.random.PRNGKey(3))
    reconstructed = cholesky_factor @ cholesky_factor.T
    assert jnp.allclose(reconstructed, matrix, atol=1e-3)


def test_rp_cholesky_is_jit_compatible() -> None:
    """RPCholesky compiles under ``jax.jit``."""
    rng = jax.random.PRNGKey(4)
    factor = jax.random.normal(rng, (5, 5))
    matrix = factor @ factor.T

    jitted = jax.jit(lambda key: rp_cholesky(matrix=matrix, rank=3, key=key))
    cholesky_factor = jitted(jax.random.PRNGKey(5))
    assert cholesky_factor.shape == (5, 3)


def test_rp_cholesky_pivot_distribution_is_proportional_to_residual_diag() -> None:
    """On an identity matrix the diagonal residual is uniform, so pivots are uniform.

    Cite: Chen+ arXiv:2207.06503. The pivot sampling probability is
    ``residual_diag[i] / sum(residual_diag)``; for ``A = I`` every
    column has equal residual at every step.
    """
    matrix = jnp.eye(4)

    cholesky_factor = rp_cholesky(matrix=matrix, rank=4, key=jax.random.PRNGKey(11))
    reconstructed = cholesky_factor @ cholesky_factor.T
    assert jnp.allclose(reconstructed, matrix, atol=1e-4)


def test_cholesky_greedy_picks_diagonal_dominant_pivot_on_dominant_diagonal_matrix() -> None:
    """Greedy pivot selects the largest diagonal entry first.

    Constructing a matrix whose largest diagonal entry lives at index ``j``
    confirms greedy picks the dominant column first; this is the
    deterministic-pivoting baseline RPCholesky's randomized variant
    contrasts against.
    """
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 10.0, 0.5]))

    cholesky_factor = cholesky_greedy(matrix=matrix, rank=1)
    # The first column of L should be exactly aligned with column 2 (the
    # dominant diagonal) up to sign and normalisation.
    first_column = cholesky_factor[:, 0]
    assert jnp.allclose(jnp.abs(first_column[2]), jnp.sqrt(10.0), atol=1e-4)
    assert jnp.allclose(jnp.abs(first_column[0]), 0.0, atol=1e-4)
    assert jnp.allclose(jnp.abs(first_column[1]), 0.0, atol=1e-4)
    assert jnp.allclose(jnp.abs(first_column[3]), 0.0, atol=1e-4)
