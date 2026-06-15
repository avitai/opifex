"""Matrix-function-vector products via Krylov projection and Chebyshev expansion.

Given a symmetric (or general square) operator ``A`` accessed only through
``matvec`` and a scalar function ``matfun`` analytic on the spectrum of
``A``, these routines approximate ``f(A) @ v`` without ever materialising
``A`` or ``f(A)``.

Algorithms
----------
* ``dense_funm_sym_eigh`` — apply ``matfun`` to a small dense symmetric
  matrix via eigendecomposition. Building block for the Krylov-projection
  matrix functions below.
* ``funm_lanczos_sym`` — symmetric Krylov projection: Lanczos →
  tridiagonal ``T`` → dense ``f(T)`` → lift via ``length * Q @ f(T) @ e_1``.
* ``funm_arnoldi`` — non-symmetric Krylov projection via Arnoldi.
* ``funm_chebyshev`` — Chebyshev polynomial expansion on a spectrum
  contained in ``(-1, 1)``. Cheaper than Lanczos / Arnoldi when ``matfun``
  is analytic on a bounded interval and the spectrum is known a priori.

Sibling reference (line-by-line port): ``matfree/matfree/funm.py``.

References
----------
* Higham — *Functions of Matrices: Theory and Computation* (2008).
* Krämer arXiv:2405.17277.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg.krylov import arnoldi_hessenberg, lanczos_tridiag


if TYPE_CHECKING:
    from collections.abc import Callable


def dense_funm_sym_eigh(
    *,
    matrix: jax.Array,
    matfun: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Apply ``matfun`` to a dense symmetric matrix via eigendecomposition.

    Computes ``U @ diag(matfun(eigvals)) @ U.T`` where ``eigvals, U`` are
    the eigenvalues and orthonormal eigenvectors of ``matrix``. ``matfun``
    must be jittable and broadcast over the eigenvalue array.

    Sibling reference: ``matfree/matfree/funm.py:300 dense_funm_sym_eigh``.
    """
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    transformed = matfun(eigvals)
    return eigvecs @ jnp.diag(transformed) @ eigvecs.T


def funm_lanczos_sym(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
    matfun: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Compute ``matfun(A) @ init_vec`` via Lanczos for symmetric ``A``.

    Pipeline: Lanczos → tridiagonal ``T`` → dense ``f(T)`` → lift.

    Args:
        matvec: symmetric matvec callable.
        init_vec: vector ``v`` such that the function returns ``f(A) @ v``.
        num_matvecs: number of Lanczos iterations. Static.
        matfun: scalar function applied to eigenvalues of the tridiagonal
            projection. Must broadcast over a 1-D array.

    Returns:
        The approximation of ``matfun(A) @ init_vec`` as a ``(dim,)`` array.

    Sibling reference: ``matfree/matfree/funm.py:116 funm_lanczos_sym``.
    """
    length = jnp.linalg.norm(init_vec)
    basis, diag, off_diag = lanczos_tridiag(
        matvec=matvec, init_vec=init_vec, num_matvecs=num_matvecs
    )
    tridiagonal = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    transformed = dense_funm_sym_eigh(matrix=tridiagonal, matfun=matfun)
    e1 = jnp.eye(num_matvecs)[0, :]
    return length * (basis @ (transformed @ e1))


def funm_arnoldi(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
    matfun: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Compute ``matfun(A) @ init_vec`` via Arnoldi for general square ``A``.

    Pipeline: Arnoldi → Hessenberg ``H`` → dense ``f(H)`` via
    eigendecomposition → lift. Eigenvalues of ``H`` may be complex; the
    dense ``f`` is taken via ``jnp.linalg.eig`` and reconstruction uses
    the right-eigenvector matrix.

    Sibling reference: ``matfree/matfree/funm.py:147 funm_arnoldi``.
    """
    length = jnp.linalg.norm(init_vec)
    basis, hessenberg = arnoldi_hessenberg(
        matvec=matvec, init_vec=init_vec, num_matvecs=num_matvecs
    )
    eigvals, eigvecs = jnp.linalg.eig(hessenberg)
    transformed = eigvecs @ jnp.diag(matfun(eigvals)) @ jnp.linalg.inv(eigvecs)
    e1 = jnp.eye(num_matvecs)[0, :]
    result = length * (basis @ (transformed @ e1))
    return result.real


def funm_chebyshev(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
    matfun: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Compute ``matfun(A) @ init_vec`` via Chebyshev polynomial expansion.

    Assumes the spectrum of ``A`` is contained in ``(-1, 1)`` and
    ``matfun`` is analytic on that interval. Faster than Lanczos / Arnoldi
    when both conditions hold because no Krylov basis is built.

    Sibling reference (port of the Clenshaw-style recurrence):
    ``matfree/matfree/funm.py:44 funm_chebyshev``.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)``.
        init_vec: vector ``v`` such that the function returns ``f(A) @ v``.
        num_matvecs: degree of the Chebyshev expansion (static).
        matfun: scalar function broadcasting over the Chebyshev nodes.

    Returns:
        Approximation of ``matfun(A) @ init_vec``.
    """
    indices = jnp.arange(num_matvecs, dtype=init_vec.dtype) + 1
    nodes = jnp.cos((2 * indices - 1) / (2 * num_matvecs) * jnp.pi)
    fx_nodes = matfun(nodes)

    t2_n = nodes
    t1_n = jnp.ones_like(nodes)
    c1 = jnp.mean(fx_nodes * t1_n)
    c2 = 2.0 * jnp.mean(fx_nodes * t2_n)

    t2_x = matvec(init_vec)
    t1_x = init_vec
    value = c1 * t1_x + c2 * t2_x

    def body(
        _step: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        value_state, t2_n_state, t1_n_state, t2_x_state, t1_x_state = state
        next_t2_n = 2.0 * nodes * t2_n_state - t1_n_state
        next_t1_n = t2_n_state
        next_c2 = 2.0 * jnp.mean(fx_nodes * next_t2_n)
        next_t2_x = 2.0 * matvec(t2_x_state) - t1_x_state
        next_t1_x = t2_x_state
        next_value = value_state + next_c2 * next_t2_x
        return next_value, next_t2_n, next_t1_n, next_t2_x, next_t1_x

    final = jax.lax.fori_loop(0, num_matvecs - 1, body, (value, t2_n, t1_n, t2_x, t1_x))
    return final[0]


__all__ = [
    "dense_funm_sym_eigh",
    "funm_arnoldi",
    "funm_chebyshev",
    "funm_lanczos_sym",
]
