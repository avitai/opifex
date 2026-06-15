"""Partial Cholesky low-rank factorisations of positive-semidefinite matrices.

Two algorithms produce a low-rank factor ``L`` of shape ``(dim, rank)``
such that ``L @ L.T`` approximates a PSD operator ``A``:

* ``cholesky_greedy`` — Harbrecht, Peters, Schneider 2012. Pivot is the
  index of the largest current residual diagonal entry. Deterministic;
  exact when ``rank == dim`` for full-rank PSD matrices.
* ``rp_cholesky`` — Chen, Epperly, Tropp, Webber 2024 (arXiv:2207.06503).
  Pivot is sampled from a categorical distribution proportional to the
  residual diagonal. Provides theoretical guarantees on the spectral and
  trace approximation error in expectation.

Sibling references (line-by-line ports):
* ``matfree/matfree/low_rank.py`` (greedy variant, matrix-element API).
* ``eepperly/Randomly-Pivoted-Cholesky/rpcholesky.py::cholesky_helper``
  with ``alg='rp'``.

References
----------
* Harbrecht, Peters, Schneider — *On the low-rank approximation by the
  pivoted Cholesky decomposition*, Appl. Numer. Math. 2012.
* Chen, Epperly, Tropp, Webber arXiv:2207.06503 — *Randomly Pivoted
  Cholesky*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cholesky_greedy(*, matrix: jax.Array, rank: int) -> jax.Array:
    """Compute a partial Cholesky factorisation with greedy pivoting.

    At each step the algorithm selects the diagonal entry of the residual
    with the largest current value, then performs the corresponding
    Cholesky-style column update on the running factor.

    Args:
        matrix: dense ``(dim, dim)`` symmetric positive-semidefinite
            matrix.
        rank: number of Cholesky iterations (static); ``rank <= dim``.

    Returns:
        Factor ``L`` of shape ``(dim, rank)``. ``L @ L.T`` is the
        rank-``rank`` greedy partial Cholesky approximation to
        ``matrix``.
    """
    dim = matrix.shape[0]
    factor = jnp.zeros((dim, rank), dtype=matrix.dtype)
    residual_diag = jnp.diag(matrix)

    def body(step: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        factor_state, diag_state = state
        pivot = jnp.argmax(diag_state)
        pivot_value = diag_state[pivot]
        pivot_value_safe = jnp.where(pivot_value > 0, pivot_value, 1.0)
        column = matrix[:, pivot] - factor_state @ factor_state[pivot, :]
        new_column = column / jnp.sqrt(pivot_value_safe)
        new_column = jnp.where(pivot_value > 0, new_column, jnp.zeros_like(new_column))
        factor_next = factor_state.at[:, step].set(new_column)
        diag_next = jnp.clip(diag_state - new_column**2, min=0.0)
        return factor_next, diag_next

    factor, _ = jax.lax.fori_loop(0, rank, body, (factor, residual_diag))
    return factor


def rp_cholesky(*, matrix: jax.Array, rank: int, key: jax.Array) -> jax.Array:
    """Compute a partial Cholesky factorisation with randomly-pivoted Cholesky.

    Pivots are sampled at each step from a categorical distribution
    whose probability mass is proportional to the current residual
    diagonal. Per Chen+ arXiv:2207.06503, this scheme yields strong
    expected-error guarantees for low-rank approximation of PSD matrices
    (in particular, optimal expected Frobenius error after the rank-r
    iteration for a rank-r ground truth).

    Args:
        matrix: dense ``(dim, dim)`` symmetric positive-semidefinite
            matrix.
        rank: number of Cholesky iterations (static); ``rank <= dim``.
        key: PRNG key, split into one substream per pivot draw.

    Returns:
        Factor ``L`` of shape ``(dim, rank)``. ``L @ L.T`` is the
        rank-``rank`` randomly-pivoted Cholesky approximation.
    """
    dim = matrix.shape[0]
    factor = jnp.zeros((dim, rank), dtype=matrix.dtype)
    residual_diag = jnp.diag(matrix)

    def body(
        step: int, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        factor_state, diag_state, key_state = state
        key_state, subkey = jax.random.split(key_state)
        diag_sum = jnp.sum(diag_state)
        diag_sum_safe = jnp.where(diag_sum > 0, diag_sum, 1.0)
        probabilities = diag_state / diag_sum_safe
        pivot = jax.random.choice(subkey, dim, p=probabilities)
        pivot_value = diag_state[pivot]
        pivot_value_safe = jnp.where(pivot_value > 0, pivot_value, 1.0)
        column = matrix[:, pivot] - factor_state @ factor_state[pivot, :]
        new_column = column / jnp.sqrt(pivot_value_safe)
        new_column = jnp.where(pivot_value > 0, new_column, jnp.zeros_like(new_column))
        factor_next = factor_state.at[:, step].set(new_column)
        diag_next = jnp.clip(diag_state - new_column**2, min=0.0)
        return factor_next, diag_next, key_state

    factor, _, _ = jax.lax.fori_loop(0, rank, body, (factor, residual_diag, key))
    return factor


__all__ = ["cholesky_greedy", "rp_cholesky"]
