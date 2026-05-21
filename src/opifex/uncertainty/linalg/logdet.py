"""Stochastic Lanczos quadrature for log-determinant estimation.

For a symmetric positive-definite operator ``A`` accessed only through a
``matvec`` callable, this module estimates ``log det(A) = trace(log(A))``
using stochastic Lanczos quadrature (SLQ): for each Rademacher probe
``v``, the quadratic form ``v.T @ log(A) @ v`` is approximated via the
Lanczos tridiagonalisation, and the trace estimator averages over probes.

Algorithm
---------
1. Sample ``num_samples`` Rademacher probes (per-entry ±1).
2. For each probe, normalise to unit length, build a Lanczos
   tridiagonalisation of dimension ``num_matvecs``, and compute
   ``length^2 * e_1.T @ log(T) @ e_1`` using ``dense_funm_sym_eigh(log)``.
3. Return the mean of the per-probe estimates.

Sibling reference (line-by-line port): ``matfree/matfree/funm.py:178
integrand_funm_sym_logdet``.

References
----------
* Ubaru, Chen, Saad 2017 — *Fast estimation of tr(f(A)) via stochastic
  Lanczos quadrature*, SIAM J. Matrix Anal. Appl.
* Krämer arXiv:2405.17277.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg.funm import dense_funm_sym_eigh
from opifex.uncertainty.linalg.krylov import lanczos_tridiag


if TYPE_CHECKING:
    from collections.abc import Callable


def slq_logdet(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    num_matvecs: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate ``log det(A)`` for symmetric positive-definite ``A``.

    Args:
        matvec: SPD matvec callable.
        dim: matrix dimension ``n`` (static).
        num_samples: number of Rademacher probes (static). Variance of
            the estimator scales as ``O(1 / num_samples)``.
        num_matvecs: Lanczos depth per probe (static). Higher depth
            improves the per-probe quadratic-form approximation.
        key: PRNG key, split into one substream per probe.

    Returns:
        Scalar JAX array containing the SLQ estimate of ``log det(A)``.
    """
    bernoulli = jax.random.bernoulli(key, shape=(num_samples, dim))
    probes = 2.0 * bernoulli.astype(jnp.float32) - 1.0

    def per_probe_estimate(probe: jax.Array) -> jax.Array:
        length = jnp.linalg.norm(probe)
        _basis, diag, off_diag = lanczos_tridiag(
            matvec=matvec, init_vec=probe, num_matvecs=num_matvecs
        )
        # Lucky-breakdown handling under static-shape constraints.
        #
        # When Lanczos hits an A-invariant subspace at step k (signalled
        # by ``off_diag[k] = 0``), the leading (k+1)x(k+1) tridiagonal
        # ``T_k`` is similar to A restricted to that invariant subspace
        # and ``e_1.T @ log(T_k) @ e_1 = probe.T @ log(A) @ probe`` holds
        # EXACTLY — there is no truncation error (Saad, *Iterative
        # Methods for Sparse Linear Systems*, §6.6).
        #
        # Under ``jax.lax.fori_loop`` the iteration count is static, so
        # we cannot dynamically shrink the tridiagonal. Instead we
        # *complete* it block-diagonally: trailing rows that were
        # zero-padded by ``_safe_normalise`` get their diagonal entries
        # replaced by ``1`` so the bottom-right block becomes the
        # identity ``I_{n-k-1}``. The completed matrix is
        # ``T_complete = diag(T_k, I_{n-k-1})``, and since matrix
        # functions act block-diagonally on block-diagonal matrices
        # (Higham, *Functions of Matrices*, Theorem 1.13),
        # ``log(T_complete) = diag(log(T_k), log(I) = 0)``. Because
        # ``e_1`` lies in the first block, ``e_1.T @ log(T_complete) @
        # e_1 = e_1.T @ log(T_k) @ e_1`` exactly. The completion is a
        # JIT-friendly re-expression of "use the truncated Lanczos
        # space", with no approximation introduced.
        connected_prefix = jnp.cumprod(off_diag != 0.0)
        diag_active_mask = jnp.concatenate([jnp.ones(1), connected_prefix]) > 0
        completed_diag = jnp.where(diag_active_mask, diag, 1.0)
        off_diag_arr = jnp.asarray(off_diag)
        tridiagonal = (
            jnp.diag(completed_diag) + jnp.diag(off_diag_arr, k=1) + jnp.diag(off_diag_arr, k=-1)
        )
        log_tridiagonal = dense_funm_sym_eigh(matrix=tridiagonal, matfun=jnp.log)
        e1 = jnp.eye(num_matvecs)[0, :]
        return length**2 * (e1 @ log_tridiagonal @ e1)

    estimates = jax.vmap(per_probe_estimate)(probes)
    return jnp.mean(estimates)


__all__ = ["slq_logdet"]
