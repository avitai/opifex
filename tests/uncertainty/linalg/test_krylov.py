"""Tests for matrix-free Krylov decompositions.

References
----------
* Lanczos 1950 — *An iteration method for the solution of the eigenvalue problem
  of linear differential and integral operators*.
* Arnoldi 1951 — *The principle of minimized iterations in the solution of the
  matrix eigenvalue problem*.
* Golub & Kahan 1965 — *Calculating the singular values and pseudo-inverse of a
  matrix*.
* Krämer arXiv:2405.17277 — *Gradients of matrix functions in JAX*
  (differentiable Lanczos / Arnoldi).

Sibling reference (line-by-line port): ``matfree/matfree/decomp.py`` —
``tridiag_sym``, ``hessenberg``, ``bidiag``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import (
    arnoldi_hessenberg,
    golub_kahan_bidiag,
    lanczos_tridiag,
)


def test_lanczos_tridiag_basis_is_orthonormal() -> None:
    """Lanczos vectors form an orthonormal basis of the Krylov subspace.

    Cite: Lanczos 1950. After ``num_matvecs`` iterations the columns of the
    returned basis should satisfy ``basis.T @ basis ≈ I``.
    """
    rng = jax.random.PRNGKey(0)
    raw = jax.random.normal(rng, (6, 6))
    matrix = raw @ raw.T + 0.1 * jnp.eye(6)  # SPD

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(1), (6,))
    basis, diag, off_diag = lanczos_tridiag(matvec=matvec, init_vec=init_vec, num_matvecs=6)
    gram = basis.T @ basis
    assert basis.shape == (6, 6)
    assert diag.shape == (6,)
    assert off_diag.shape == (5,)
    assert jnp.allclose(gram, jnp.eye(6), atol=1e-3)


def test_lanczos_tridiag_reconstructs_symmetric_operator() -> None:
    """Full-dimension Lanczos recovers ``A = V T V^T`` exactly (modulo roundoff)."""
    rng = jax.random.PRNGKey(2)
    raw = jax.random.normal(rng, (5, 5))
    matrix = 0.5 * (raw + raw.T) + 0.5 * jnp.eye(5)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(3), (5,))
    basis, diag, off_diag = lanczos_tridiag(matvec=matvec, init_vec=init_vec, num_matvecs=5)
    tridiagonal = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    reconstructed = basis @ tridiagonal @ basis.T
    assert jnp.allclose(reconstructed, matrix, atol=1e-3)


def test_lanczos_tridiag_is_jit_compatible() -> None:
    """The Lanczos forward pass must compile under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(lambda v: lanczos_tridiag(matvec=matvec, init_vec=v, num_matvecs=4))
    basis, diag, off_diag = jitted(jnp.asarray([1.0, 1.0, 1.0, 1.0]))
    assert basis.shape == (4, 4)
    assert jnp.all(jnp.isfinite(diag))
    assert jnp.all(jnp.isfinite(off_diag))


def test_arnoldi_hessenberg_basis_is_orthonormal_on_non_symmetric_matrix() -> None:
    """Arnoldi produces an orthonormal basis of the Krylov subspace.

    Cite: Arnoldi 1951. For non-symmetric matrices the projection is upper
    Hessenberg, not tridiagonal.
    """
    rng = jax.random.PRNGKey(11)
    matrix = jax.random.normal(rng, (5, 5))  # non-symmetric

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(12), (5,))
    basis, hessenberg = arnoldi_hessenberg(matvec=matvec, init_vec=init_vec, num_matvecs=5)
    assert basis.shape == (5, 5)
    assert hessenberg.shape == (5, 5)
    assert jnp.allclose(basis.T @ basis, jnp.eye(5), atol=1e-3)
    # Hessenberg structure: zero below the first sub-diagonal.
    below_subdiag = jnp.tril(hessenberg, k=-2)
    assert jnp.allclose(below_subdiag, jnp.zeros((5, 5)), atol=1e-5)


def test_arnoldi_hessenberg_reconstructs_non_symmetric_operator() -> None:
    """Full-dimension Arnoldi recovers ``A = V H V^T`` exactly."""
    rng = jax.random.PRNGKey(20)
    matrix = jax.random.normal(rng, (4, 4))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(21), (4,))
    basis, hessenberg = arnoldi_hessenberg(matvec=matvec, init_vec=init_vec, num_matvecs=4)
    reconstructed = basis @ hessenberg @ basis.T
    assert jnp.allclose(reconstructed, matrix, atol=1e-3)


def test_arnoldi_hessenberg_is_jit_compatible() -> None:
    """The Arnoldi forward pass must compile under ``jax.jit``."""
    matrix = jnp.asarray([[2.0, 1.0, 0.5], [0.3, 2.0, 1.0], [0.1, 0.4, 2.0]])

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(lambda v: arnoldi_hessenberg(matvec=matvec, init_vec=v, num_matvecs=3))
    init_vec = jnp.asarray([1.0, 1.0, 1.0])
    basis, hessenberg = jitted(init_vec)
    assert basis.shape == (3, 3)
    assert jnp.all(jnp.isfinite(hessenberg))


def test_arnoldi_hessenberg_handles_lucky_breakdown_without_nan() -> None:
    """When the Krylov subspace is A-invariant, the algorithm must not produce NaN.

    Cite: Saad, *Iterative Methods for Sparse Linear Systems*, §6.5 — *lucky
    breakdown*. An exact A-invariant subspace yields ``beta = 0`` at some
    step ``k``; the canonical handling is to zero-pad the basis past ``k``
    and set ``hessenberg[k+1:, k+1:] = 0`` so callers can detect breakdown
    via the zero sub-diagonal entry. Implementation must remain
    ``jax.jit``-compatible (static loop bounds).
    """
    # The standard basis vector ``e_0`` is in an invariant subspace of an
    # upper-triangular matrix, so Arnoldi terminates after one step.
    matrix = jnp.asarray([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]])

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    basis, hessenberg = arnoldi_hessenberg(
        matvec=matvec, init_vec=jnp.asarray([1.0, 0.0, 0.0]), num_matvecs=3
    )
    assert jnp.all(jnp.isfinite(basis))
    assert jnp.all(jnp.isfinite(hessenberg))
    # Sub-diagonal entry [1, 0] signals lucky breakdown at step 0.
    assert jnp.allclose(hessenberg[1, 0], 0.0, atol=1e-6)


def test_lanczos_tridiag_handles_lucky_breakdown_without_nan() -> None:
    """Lanczos must zero-pad when the Krylov subspace is A-invariant.

    Cite: Saad, *Iterative Methods*, §6.5. An eigenvector of ``A`` triggers
    immediate lucky breakdown: ``r_0 = A v_0 - alpha_0 v_0 = 0``.
    """
    eigenvalue = 3.0
    matrix = jnp.diag(jnp.asarray([eigenvalue, 1.0, 2.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    # init_vec is exactly an eigenvector — Lanczos breaks down at step 0.
    basis, diag, off_diag = lanczos_tridiag(
        matvec=matvec, init_vec=jnp.asarray([1.0, 0.0, 0.0]), num_matvecs=3
    )
    assert jnp.all(jnp.isfinite(basis))
    assert jnp.all(jnp.isfinite(diag))
    assert jnp.all(jnp.isfinite(off_diag))
    # The first diagonal entry is the eigenvalue; off-diagonals are zero.
    assert jnp.allclose(diag[0], eigenvalue, atol=1e-6)
    assert jnp.allclose(off_diag, jnp.zeros_like(off_diag), atol=1e-6)


def test_golub_kahan_bidiag_factors_recover_singular_values() -> None:
    """Bidiagonalisation produces left/right bases whose bidiag matches ``A``.

    Cite: Golub & Kahan 1965. ``A = U B V^T`` where ``B`` is bidiagonal.
    The singular values of ``A`` equal the singular values of ``B``.
    """
    rng = jax.random.PRNGKey(31)
    matrix = jax.random.normal(rng, (4, 4))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    def matvec_t(vector: jax.Array) -> jax.Array:
        return matrix.T @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(32), (4,))
    left_basis, right_basis, bidiag = golub_kahan_bidiag(
        matvec=matvec,
        matvec_transpose=matvec_t,
        init_vec=init_vec,
        num_matvecs=4,
    )
    assert left_basis.shape == (4, 4)
    assert right_basis.shape == (4, 4)
    assert bidiag.shape == (4, 4)
    # Singular values of bidiag should match singular values of the matrix.
    matrix_singular_values = jnp.sort(jnp.linalg.svd(matrix, compute_uv=False))
    bidiag_singular_values = jnp.sort(jnp.linalg.svd(bidiag, compute_uv=False))
    assert jnp.allclose(matrix_singular_values, bidiag_singular_values, atol=1e-3)


def test_golub_kahan_bidiag_is_jit_compatible() -> None:
    """The Golub-Kahan bidiag pass compiles under ``jax.jit``."""
    matrix = jnp.eye(3) * 2.0 + jnp.diag(jnp.asarray([0.5, 0.3]), k=1)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    def matvec_t(vector: jax.Array) -> jax.Array:
        return matrix.T @ vector

    jitted = jax.jit(
        lambda v: golub_kahan_bidiag(
            matvec=matvec, matvec_transpose=matvec_t, init_vec=v, num_matvecs=3
        )
    )
    left_basis, right_basis, bidiag = jitted(jnp.asarray([1.0, 0.0, 0.0]))
    assert left_basis.shape == (3, 3)
    assert right_basis.shape == (3, 3)
    assert bidiag.shape == (3, 3)


def test_golub_kahan_bidiag_handles_lucky_breakdown_without_nan() -> None:
    """Bidiagonalisation must zero-pad on a rank-deficient operator.

    Cite: Golub & Van Loan, *Matrix Computations*, §10.4. A rank-1 operator
    triggers ``alpha = 0`` after the first step; the algorithm should set
    the rest of the basis and bidiagonal to zero rather than producing NaN.
    """
    column = jnp.asarray([1.0, 2.0, 3.0])
    matrix = jnp.outer(column, column) / 14.0  # rank-1 symmetric PSD

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    def matvec_t(vector: jax.Array) -> jax.Array:
        return matrix.T @ vector

    left_basis, _right_basis, bidiag = golub_kahan_bidiag(
        matvec=matvec,
        matvec_transpose=matvec_t,
        init_vec=column,
        num_matvecs=3,
    )
    assert jnp.all(jnp.isfinite(left_basis))
    assert jnp.all(jnp.isfinite(bidiag))
    # Beyond rank 1, the bidiagonal entries are zero — the algorithm has
    # detected the invariant subspace.
    assert jnp.allclose(bidiag[1:, 1:], jnp.zeros_like(bidiag[1:, 1:]), atol=1e-6)
