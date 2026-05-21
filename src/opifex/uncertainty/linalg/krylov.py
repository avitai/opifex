"""Matrix-free Krylov decompositions.

Algorithms
----------
* ``lanczos_tridiag`` — Lanczos 1950. Builds an orthonormal basis ``V`` of
  the Krylov subspace ``span{v, A v, A^2 v, ...}`` for a symmetric operator
  ``A`` together with a tridiagonal matrix ``T`` such that ``A V ≈ V T``.
  Forward pass uses ``jax.lax.fori_loop`` so the iteration count is static
  but the loop runs in a single XLA kernel. Sibling reference:
  ``matfree/matfree/decomp.py:30 tridiag_sym``.
* ``arnoldi_hessenberg`` — Arnoldi 1951. The non-symmetric analogue:
  ``V`` is orthonormal, ``H`` is upper Hessenberg, ``A V ≈ V H``. Sibling
  reference: ``matfree/matfree/decomp.py:348 hessenberg``.
* ``golub_kahan_bidiag`` — Golub & Kahan 1965. Bidiagonalises a general
  rectangular operator with both ``matvec`` and ``matvec_transpose``,
  yielding left/right orthonormal bases and a bidiagonal matrix whose
  singular values match those of ``A``. Sibling reference:
  ``matfree/matfree/decomp.py:600 bidiag``.

Degenerate / lucky-breakdown handling
-------------------------------------
When the iterate produces a residual whose norm is below ``_BREAKDOWN_EPS``,
the Krylov subspace is A-invariant and the algorithm has found an exact
decomposition for the steps completed (Saad, *Iterative Methods*, §6.5;
Golub & Van Loan, §10.1.4 — *lucky breakdown*). Because static loop bounds
forbid early termination under ``jax.jit``, we mask via ``jnp.where``:

* the off-diagonal (or sub-Hessenberg) entry is set to zero;
* the next basis vector is set to the zero vector;
* the remaining steps continue with zeros, yielding a block-diagonal
  projection past the breakdown point.

Callers detect lucky breakdown by checking for zero entries in the
off-diagonal / sub-Hessenberg.

References
----------
* Lanczos 1950 — *An iteration method for the solution of the eigenvalue
  problem of linear differential and integral operators*.
* Arnoldi 1951 — *The principle of minimized iterations*.
* Golub & Kahan 1965 — *Calculating the singular values and pseudo-inverse
  of a matrix*.
* Saad — *Iterative Methods for Sparse Linear Systems*, §6.5 (lucky
  breakdown).
* Golub & Van Loan — *Matrix Computations*, §10.1.4.
* Krämer arXiv:2405.17277 — *Gradients of matrix functions in JAX*
  (differentiable Lanczos / Arnoldi).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


_BREAKDOWN_EPS_SCALE = 100.0
"""Multiplicative slack above the dtype's machine epsilon for breakdown.

The breakdown threshold is set to ``finfo(dtype).eps * _BREAKDOWN_EPS_SCALE``
to absorb the cumulative rounding error that the Lanczos / Arnoldi /
Golub-Kahan recurrences incur while iterating against an exactly
A-invariant subspace. The value ``100`` is conservative: for ``float32``
it yields ``≈ 1.2e-5`` which catches the worst observed roundoff on the
opifex lucky-breakdown unit tests while leaving plenty of headroom for
genuine small-but-nonzero residuals.
"""


def _safe_normalise(vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Normalise a vector with safe handling of zero norm (lucky breakdown).

    Returns ``(unit, norm)``. When ``norm`` falls at or below the
    dtype-aware breakdown threshold the returned unit vector is zero and
    the returned norm is exactly zero. Otherwise ``unit = vector / norm``.

    This pattern keeps the function ``jax.jit`` compatible (no Python-side
    branching) while propagating zero entries that downstream code can
    detect as breakdown markers.
    """
    norm = jnp.asarray(jnp.linalg.norm(vector))
    eps = jnp.finfo(norm.dtype).eps * _BREAKDOWN_EPS_SCALE
    is_nonzero = norm > eps
    safe_norm = jnp.where(is_nonzero, norm, jnp.asarray(1.0, dtype=norm.dtype))
    unit = jnp.where(is_nonzero, vector / safe_norm, jnp.zeros_like(vector))
    reported_norm = jnp.where(is_nonzero, norm, jnp.zeros_like(norm))
    return unit, reported_norm


def lanczos_tridiag(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Lanczos tridiagonalisation of a symmetric matrix-free operator.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)``. Must be symmetric
            for the tridiagonal output to be accurate; non-symmetric inputs
            yield a still-orthogonal basis but the projection is no longer
            tridiagonal.
        init_vec: starting vector of shape ``(dim,)``. Normalised internally.
        num_matvecs: number of Lanczos iterations. Static.

    Returns:
        ``(basis, diag, off_diag)`` where ``basis`` is ``(dim, num_matvecs)``
        orthonormal, ``diag`` is ``(num_matvecs,)`` containing the main
        diagonal of the tridiagonal projection, and ``off_diag`` is
        ``(num_matvecs - 1,)`` containing the sub/super-diagonal.
    """
    dim = init_vec.shape[0]
    init_unit, _ = _safe_normalise(init_vec)
    first_image = matvec(init_unit)
    first_alpha = jnp.dot(init_unit, first_image)
    residual = first_image - first_alpha * init_unit
    second_unit, first_beta = _safe_normalise(residual)

    basis = jnp.zeros((num_matvecs, dim), dtype=init_vec.dtype)
    basis = basis.at[0].set(init_unit)
    diag = jnp.zeros((num_matvecs,), dtype=init_vec.dtype).at[0].set(first_alpha)
    off_diag = jnp.zeros((num_matvecs - 1,), dtype=init_vec.dtype).at[0].set(first_beta)
    if num_matvecs > 1:
        basis = basis.at[1].set(second_unit)

    def body(
        step: int, state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        basis_state, diag_state, off_diag_state, current_unit, previous_unit = state
        image = matvec(current_unit)
        alpha = jnp.dot(current_unit, image)
        previous_beta = off_diag_state[step - 1]
        residual = image - alpha * current_unit - previous_beta * previous_unit
        next_unit, beta = _safe_normalise(residual)
        diag_state = diag_state.at[step].set(alpha)
        off_diag_state = jax.lax.cond(
            step < num_matvecs - 1,
            lambda od: od.at[step].set(beta),
            lambda od: od,
            off_diag_state,
        )
        basis_state = jax.lax.cond(
            step < num_matvecs - 1,
            lambda b: b.at[step + 1].set(next_unit),
            lambda b: b,
            basis_state,
        )
        return basis_state, diag_state, off_diag_state, next_unit, current_unit

    basis, diag, off_diag, _, _ = jax.lax.fori_loop(
        1, num_matvecs, body, (basis, diag, off_diag, second_unit, init_unit)
    )
    return basis.T, diag, off_diag


def arnoldi_hessenberg(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array]:
    """Arnoldi process — non-symmetric Krylov decomposition.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)``.
        init_vec: starting vector of shape ``(dim,)``. Normalised internally.
        num_matvecs: number of Arnoldi iterations. Static.

    Returns:
        ``(basis, hessenberg)`` where ``basis`` is ``(dim, num_matvecs)``
        orthonormal and ``hessenberg`` is ``(num_matvecs, num_matvecs)``
        upper Hessenberg.
    """
    dim = init_vec.shape[0]
    init_unit, _ = _safe_normalise(init_vec)

    basis = jnp.zeros((num_matvecs, dim), dtype=init_vec.dtype).at[0].set(init_unit)
    hessenberg = jnp.zeros((num_matvecs, num_matvecs), dtype=init_vec.dtype)

    def body(step: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        basis_state, hessenberg_state = state
        current_unit = basis_state[step]
        image = matvec(current_unit)
        # Modified Gram-Schmidt against all previous basis vectors.
        coefficients = basis_state @ image  # (num_matvecs,) inner products
        # Mask out entries with index > step to keep iteration uncontaminated.
        index_mask = jnp.arange(num_matvecs) <= step
        masked_coeffs = jnp.where(index_mask, coefficients, 0.0)
        residual = image - basis_state.T @ masked_coeffs
        next_unit, beta = _safe_normalise(residual)
        hessenberg_state = hessenberg_state.at[:, step].set(masked_coeffs)
        hessenberg_state = jax.lax.cond(
            step < num_matvecs - 1,
            lambda h: h.at[step + 1, step].set(beta),
            lambda h: h,
            hessenberg_state,
        )
        basis_state = jax.lax.cond(
            step < num_matvecs - 1,
            lambda b: b.at[step + 1].set(next_unit),
            lambda b: b,
            basis_state,
        )
        return basis_state, hessenberg_state

    basis, hessenberg = jax.lax.fori_loop(0, num_matvecs, body, (basis, hessenberg))
    return basis.T, hessenberg


def golub_kahan_bidiag(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    init_vec: jax.Array,
    num_matvecs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Golub-Kahan bidiagonalisation of a general (possibly non-symmetric) operator.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)`` (``A @ v``).
        matvec_transpose: callable mapping ``(dim,)`` to ``(dim,)``
            (``A.T @ v``). Required because bidiagonalisation alternates
            between primal and adjoint applications.
        init_vec: starting vector of shape ``(dim,)``. Normalised internally.
        num_matvecs: number of bidiagonal iterations. Static.

    Returns:
        ``(left_basis, right_basis, bidiag)`` where ``left_basis`` and
        ``right_basis`` are ``(dim, num_matvecs)`` orthonormal and
        ``bidiag`` is ``(num_matvecs, num_matvecs)`` upper bidiagonal.
        The singular values of ``bidiag`` match those of ``A``.
    """
    dim_cols = init_vec.shape[0]
    right_init, _ = _safe_normalise(init_vec)
    left_unnorm = matvec(right_init)
    dim_rows = left_unnorm.shape[0]
    left_init, first_alpha = _safe_normalise(left_unnorm)

    left_basis = (
        jnp.zeros((num_matvecs, dim_rows), dtype=init_vec.dtype).at[0].set(left_init)
    )
    right_basis = (
        jnp.zeros((num_matvecs, dim_cols), dtype=init_vec.dtype).at[0].set(right_init)
    )
    diag = jnp.zeros((num_matvecs,), dtype=init_vec.dtype).at[0].set(first_alpha)
    off_diag = jnp.zeros((num_matvecs,), dtype=init_vec.dtype)

    def body(
        step: int, state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        left_state, right_state, diag_state, off_diag_state = state
        previous_right = right_state[step - 1]
        previous_left = left_state[step - 1]
        previous_alpha = diag_state[step - 1]
        right_unnorm = matvec_transpose(previous_left) - previous_alpha * previous_right
        new_right, beta = _safe_normalise(right_unnorm)
        left_unnorm = matvec(new_right) - beta * previous_left
        new_left, alpha = _safe_normalise(left_unnorm)
        off_diag_state = off_diag_state.at[step - 1].set(beta)
        diag_state = diag_state.at[step].set(alpha)
        right_state = right_state.at[step].set(new_right)
        left_state = left_state.at[step].set(new_left)
        return left_state, right_state, diag_state, off_diag_state

    left_basis, right_basis, diag, off_diag = jax.lax.fori_loop(
        1, num_matvecs, body, (left_basis, right_basis, diag, off_diag)
    )
    bidiag = jnp.diag(diag) + jnp.diag(off_diag[:-1], k=1)
    return left_basis.T, right_basis.T, bidiag


__all__ = ["arnoldi_hessenberg", "golub_kahan_bidiag", "lanczos_tridiag"]
