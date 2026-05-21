"""Tests for matrix-free partial eigendecomposition / SVD.

The ``eig`` module wraps the Krylov decompositions in :mod:`opifex.uncertainty.linalg.krylov`
to produce partial eigen / singular value decompositions of matrix-free
operators.

Sibling reference (line-by-line port): ``matfree/matfree/eig.py`` —
``eigh_partial``, ``eig_partial``, ``svd_partial``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg import eig_partial, eigh_partial, svd_partial


def test_eigh_partial_recovers_full_spectrum_when_num_matvecs_equals_dim() -> None:
    """Full-dimension Lanczos + small ``eigh`` recovers the matrix's spectrum.

    Cite: Lanczos 1950 + Golub & Van Loan §10.1.4. When ``num_matvecs == dim``
    and Lanczos does not break down early, the projected tridiagonal matrix
    is similar to ``A`` and shares its eigenvalues.
    """
    rng = jax.random.PRNGKey(0)
    raw = jax.random.normal(rng, (6, 6))
    matrix = raw @ raw.T + 0.1 * jnp.eye(6)  # SPD

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(1), (6,))
    eigenvalues, eigenvectors = eigh_partial(matvec=matvec, init_vec=init_vec, num_matvecs=6)
    reference = jnp.sort(jnp.linalg.eigvalsh(matrix))
    assert jnp.allclose(jnp.sort(eigenvalues), reference, atol=1e-3)
    # Eigenvectors stored column-wise; check approximate orthogonality.
    assert jnp.allclose(eigenvectors.T @ eigenvectors, jnp.eye(6), atol=1e-3)


def test_eigh_partial_eigenpairs_satisfy_av_equals_lambda_v() -> None:
    """Each returned eigenpair satisfies the defining equation ``A v = λ v``."""
    rng = jax.random.PRNGKey(2)
    raw = jax.random.normal(rng, (5, 5))
    matrix = 0.5 * (raw + raw.T) + 1.0 * jnp.eye(5)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(3), (5,))
    eigenvalues, eigenvectors = eigh_partial(matvec=matvec, init_vec=init_vec, num_matvecs=5)
    for k in range(5):
        lhs = matrix @ eigenvectors[:, k]
        rhs = eigenvalues[k] * eigenvectors[:, k]
        assert jnp.allclose(lhs, rhs, atol=1e-2)


def test_eigh_partial_is_jit_compatible() -> None:
    """The combined Lanczos + dense ``eigh`` chain compiles under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(lambda v: eigh_partial(matvec=matvec, init_vec=v, num_matvecs=4))
    eigenvalues, eigenvectors = jitted(jnp.asarray([1.0, 1.0, 1.0, 1.0]))
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == (4, 4)


def test_svd_partial_recovers_singular_values_when_num_matvecs_equals_dim() -> None:
    """Full-dimension Golub-Kahan + small SVD recovers the singular spectrum.

    Cite: Golub & Kahan 1965. The bidiagonal matrix from Golub-Kahan has
    the same singular values as ``A``.
    """
    rng = jax.random.PRNGKey(11)
    matrix = jax.random.normal(rng, (4, 4))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    def matvec_t(vector: jax.Array) -> jax.Array:
        return matrix.T @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(12), (4,))
    left_singvecs, singvals, right_singvecs = svd_partial(
        matvec=matvec,
        matvec_transpose=matvec_t,
        init_vec=init_vec,
        num_matvecs=4,
    )
    reference = jnp.sort(jnp.linalg.svd(matrix, compute_uv=False))
    assert jnp.allclose(jnp.sort(singvals), reference, atol=1e-3)
    assert left_singvecs.shape == (4, 4)
    assert right_singvecs.shape == (4, 4)


def test_svd_partial_is_jit_compatible() -> None:
    """SVD partial decomposition passes ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([3.0, 2.0, 1.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    def matvec_t(vector: jax.Array) -> jax.Array:
        return matrix.T @ vector

    jitted = jax.jit(
        lambda v: svd_partial(
            matvec=matvec,
            matvec_transpose=matvec_t,
            init_vec=v,
            num_matvecs=3,
        )
    )
    left, sv, right = jitted(jnp.asarray([1.0, 1.0, 1.0]))
    assert left.shape == (3, 3)
    assert sv.shape == (3,)
    assert right.shape == (3, 3)


def test_eig_partial_recovers_full_spectrum_for_non_symmetric_matrix() -> None:
    """Full-dimension Arnoldi + small dense ``eig`` recovers the spectrum.

    Cite: Arnoldi 1951. For non-symmetric matrices the eigenvalues may be
    complex; the projected Hessenberg matrix is similar to ``A`` and
    shares its eigenvalues.
    """
    rng = jax.random.PRNGKey(21)
    matrix = jax.random.normal(rng, (4, 4))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(22), (4,))
    eigenvalues, _eigenvectors = eig_partial(matvec=matvec, init_vec=init_vec, num_matvecs=4)
    reference = jnp.sort(jnp.abs(jnp.linalg.eigvals(matrix)))
    assert jnp.allclose(jnp.sort(jnp.abs(eigenvalues)), reference, atol=1e-3)


def test_eig_partial_is_jit_compatible() -> None:
    """The Arnoldi + dense ``eig`` chain compiles under ``jax.jit``."""
    matrix = jnp.asarray([[2.0, 1.0, 0.5], [0.3, 2.0, 1.0], [0.1, 0.4, 2.0]])

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(lambda v: eig_partial(matvec=matvec, init_vec=v, num_matvecs=3))
    eigenvalues, eigenvectors = jitted(jnp.asarray([1.0, 1.0, 1.0]))
    assert eigenvalues.shape == (3,)
    assert eigenvectors.shape == (3, 3)
