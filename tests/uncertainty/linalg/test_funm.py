"""Tests for matrix-function evaluators (Lanczos / Arnoldi / Chebyshev).

References
----------
* Higham — *Functions of Matrices: Theory and Computation* (2008).
* Krämer arXiv:2405.17277 — *Gradients of matrix functions in JAX*.

Sibling reference (line-by-line port): ``matfree/matfree/funm.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from opifex.uncertainty.linalg import (
    dense_funm_sym_eigh,
    funm_arnoldi,
    funm_chebyshev,
    funm_lanczos_sym,
)


def test_dense_funm_sym_eigh_exp_matches_jax_expm() -> None:
    """``f(A) = exp(A)`` via eigh on a symmetric matrix matches ``jsp.linalg.expm``."""
    rng = jax.random.PRNGKey(0)
    raw = jax.random.normal(rng, (4, 4))
    matrix = 0.5 * (raw + raw.T)  # symmetric

    actual = dense_funm_sym_eigh(matrix=matrix, matfun=jnp.exp)
    reference = jsp_linalg.expm(matrix)
    assert jnp.allclose(actual, reference, atol=1e-3)


def test_funm_lanczos_sym_exp_matches_matrix_exponential_action() -> None:
    """``funm_lanczos_sym(exp)(A) @ v`` matches ``expm(A) @ v`` on SPD A."""
    rng = jax.random.PRNGKey(1)
    raw = jax.random.normal(rng, (5, 5))
    matrix = raw @ raw.T + 0.5 * jnp.eye(5)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(2), (5,))
    actual = funm_lanczos_sym(matvec=matvec, init_vec=init_vec, num_matvecs=5, matfun=jnp.exp)
    reference = jsp_linalg.expm(matrix) @ init_vec
    assert jnp.allclose(actual, reference, atol=1e-2)


def test_funm_lanczos_sym_sqrt_matches_matrix_sqrt_action() -> None:
    """``funm_lanczos_sym(sqrt)(A) @ v`` matches ``sqrtm(A) @ v`` on SPD A."""
    rng = jax.random.PRNGKey(3)
    raw = jax.random.normal(rng, (4, 4))
    matrix = raw @ raw.T + 1.0 * jnp.eye(4)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(4), (4,))
    actual = funm_lanczos_sym(matvec=matvec, init_vec=init_vec, num_matvecs=4, matfun=jnp.sqrt)
    # Build sqrt(A) via eigh — jsp.linalg.sqrtm is not supported on CUDA.
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    sqrt_matrix = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
    reference = sqrt_matrix @ init_vec
    assert jnp.allclose(actual, reference, atol=1e-2)


def test_funm_lanczos_sym_returns_eigenvector_action_under_lucky_breakdown() -> None:
    """When ``init_vec`` is an eigenvector, ``funm(λ) * v`` is returned exactly.

    Cite: Lanczos 1950 + Saad §6.5. Lucky breakdown at step 0 yields a
    1x1 tridiagonal ``T = [[λ]]``, and ``f(T) @ e_1 = f(λ)``.
    """
    eigenvalue = 3.0
    matrix = jnp.diag(jnp.asarray([eigenvalue, 1.0, 2.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jnp.asarray([1.0, 0.0, 0.0])
    actual = funm_lanczos_sym(matvec=matvec, init_vec=init_vec, num_matvecs=3, matfun=jnp.exp)
    assert jnp.allclose(actual, jnp.exp(eigenvalue) * init_vec, atol=1e-4)


def test_funm_lanczos_sym_is_jit_compatible() -> None:
    """The combined Lanczos + dense funm chain compiles under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda v: funm_lanczos_sym(matvec=matvec, init_vec=v, num_matvecs=3, matfun=jnp.exp)
    )
    result = jitted(jnp.asarray([1.0, 1.0, 1.0]))
    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))


def test_funm_arnoldi_exp_matches_matrix_exponential_action_on_non_symmetric() -> None:
    """``funm_arnoldi(exp)(A) @ v`` matches ``expm(A) @ v`` on a non-symmetric A."""
    rng = jax.random.PRNGKey(11)
    matrix = jax.random.normal(rng, (4, 4))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jax.random.normal(jax.random.PRNGKey(12), (4,))
    actual = funm_arnoldi(matvec=matvec, init_vec=init_vec, num_matvecs=4, matfun=jnp.exp)
    reference = jsp_linalg.expm(matrix) @ init_vec
    assert jnp.allclose(actual, reference, atol=5e-2)


def test_funm_arnoldi_is_jit_compatible() -> None:
    """The combined Arnoldi + dense funm chain compiles under ``jax.jit``."""
    matrix = jnp.asarray([[2.0, 1.0, 0.5], [0.3, 2.0, 1.0], [0.1, 0.4, 2.0]])

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda v: funm_arnoldi(matvec=matvec, init_vec=v, num_matvecs=3, matfun=jnp.exp)
    )
    result = jitted(jnp.asarray([1.0, 1.0, 1.0]))
    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))


def test_funm_chebyshev_recovers_linear_polynomial_exactly() -> None:
    """``funm_chebyshev`` with ``f(x) = x`` on spectrum ⊂ (-1, 1) gives ``A @ v``.

    Cite: matfree ``funm_chebyshev``. The Chebyshev expansion truncated to
    ``num_matvecs`` terms is exact when ``f`` is a polynomial of degree at
    most ``num_matvecs - 1``.
    """
    matrix = jnp.diag(jnp.asarray([0.5, -0.3, 0.1]))  # spectrum in (-1, 1)

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    init_vec = jnp.asarray([1.0, 1.0, 1.0])
    actual = funm_chebyshev(matvec=matvec, init_vec=init_vec, num_matvecs=4, matfun=lambda x: x)
    reference = matrix @ init_vec
    assert jnp.allclose(actual, reference, atol=1e-3)


def test_funm_chebyshev_is_jit_compatible() -> None:
    """``funm_chebyshev`` compiles under ``jax.jit``."""
    matrix = jnp.diag(jnp.asarray([0.2, -0.4, 0.6]))

    def matvec(vector: jax.Array) -> jax.Array:
        return matrix @ vector

    jitted = jax.jit(
        lambda v: funm_chebyshev(matvec=matvec, init_vec=v, num_matvecs=8, matfun=jnp.exp)
    )
    result = jitted(jnp.asarray([1.0, 1.0, 1.0]))
    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))
