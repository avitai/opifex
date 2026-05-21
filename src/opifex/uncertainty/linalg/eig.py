"""Partial eigen- and singular-value decompositions of matrix-free operators.

Each function combines one of the Krylov decompositions from
:mod:`opifex.uncertainty.linalg.krylov` with a *dense* eigen / SVD of the
small projected matrix and returns the eigenpairs / singular triplets in
the ambient ``dim``-dimensional space.

Algorithms
----------
* ``eigh_partial`` — symmetric case. Lanczos → tridiagonal ``T`` →
  ``jnp.linalg.eigh(T)`` → lift via ``basis @ vecs``.
* ``eig_partial`` — non-symmetric case. Arnoldi → upper Hessenberg ``H`` →
  ``jnp.linalg.eig(H)`` → lift via ``basis @ vecs``. Eigenvalues may be
  complex.
* ``svd_partial`` — rectangular case. Golub-Kahan bidiagonalisation →
  bidiagonal ``B`` → ``jnp.linalg.svd(B)`` → lift left / right vectors via
  ``left_basis @ U`` and ``right_basis @ V``.

Sibling reference (line-by-line port): ``matfree/matfree/eig.py``.

References
----------
* Lanczos 1950, Arnoldi 1951, Golub & Kahan 1965 (see :mod:`.krylov`).
* Golub & Van Loan §10.1.4 — projection-based partial eigenvalue methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg.krylov import (
    arnoldi_hessenberg,
    golub_kahan_bidiag,
    lanczos_tridiag,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def eigh_partial(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array]:
    """Partial eigendecomposition of a symmetric matrix-free operator.

    Args:
        matvec: symmetric matvec callable.
        init_vec: starting vector for the Krylov sequence.
        num_matvecs: number of Lanczos iterations.

    Returns:
        ``(eigenvalues, eigenvectors)`` where ``eigenvalues`` has shape
        ``(num_matvecs,)`` and ``eigenvectors`` has shape
        ``(dim, num_matvecs)`` with each column a unit eigenvector of the
        projected tridiagonal lifted into the full space.
    """
    basis, diag, off_diag = lanczos_tridiag(
        matvec=matvec, init_vec=init_vec, num_matvecs=num_matvecs
    )
    tridiagonal = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    eigenvalues, small_eigenvectors = jnp.linalg.eigh(tridiagonal)
    eigenvectors = basis @ small_eigenvectors
    return eigenvalues, eigenvectors


def eig_partial(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array]:
    """Partial eigendecomposition of a general (non-symmetric) operator.

    Args:
        matvec: matvec callable for a (possibly non-symmetric) operator.
        init_vec: starting vector for the Krylov sequence.
        num_matvecs: number of Arnoldi iterations.

    Returns:
        ``(eigenvalues, eigenvectors)`` where ``eigenvalues`` has shape
        ``(num_matvecs,)`` and may be complex; ``eigenvectors`` has shape
        ``(dim, num_matvecs)`` with each column lifted from the Hessenberg
        spectrum into the full space.
    """
    basis, hessenberg = arnoldi_hessenberg(
        matvec=matvec, init_vec=init_vec, num_matvecs=num_matvecs
    )
    eigenvalues, small_eigenvectors = jnp.linalg.eig(hessenberg)
    eigenvectors = basis @ small_eigenvectors
    return eigenvalues, eigenvectors


def svd_partial(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Partial SVD of a matrix-free operator.

    Args:
        matvec: callable computing ``A @ v``.
        matvec_transpose: callable computing ``A.T @ v``.
        init_vec: starting vector for the Golub-Kahan sequence.
        num_matvecs: number of bidiagonalisation iterations.

    Returns:
        ``(left_singvecs, singvals, right_singvecs)`` where ``singvals`` has
        shape ``(num_matvecs,)`` and the singular-vector matrices have
        shape ``(dim, num_matvecs)``.
    """
    left_basis, right_basis, bidiag = golub_kahan_bidiag(
        matvec=matvec,
        matvec_transpose=matvec_transpose,
        init_vec=init_vec,
        num_matvecs=num_matvecs,
    )
    small_left, singvals, small_right_t = jnp.linalg.svd(bidiag, full_matrices=False)
    left_singvecs = left_basis @ small_left
    right_singvecs = right_basis @ small_right_t.T
    return left_singvecs, singvals, right_singvecs


__all__ = ["eig_partial", "eigh_partial", "svd_partial"]
