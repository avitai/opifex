"""Matrix-free least-squares via Golub-Kahan bidiagonalisation + small solve.

Solves the (damped) least-squares problem

    min_x || A x - b ||^2  +  damping^2 || x ||^2

where ``A`` is accessed only through ``matvec`` and ``matvec_transpose``,
by reducing it to a small ``num_matvecs``-dimensional system on the
Lanczos right-basis ``V`` of the Golub-Kahan bidiagonalisation.

Algorithm
---------
1. Build a Golub-Kahan bidiagonalisation of ``A`` from the starting
   vector ``A.T b`` (so the right Krylov subspace contains the
   normal-equation gradient).
2. Reduce the original problem to ``min_z ||A V z - b||^2 +
   damping^2 ||V z||^2``. Because ``V`` is orthonormal,
   ``||V z||^2 = ||z||^2``, and ``A V z`` is captured by the dense
   bidiagonal ``B`` plus the projection onto the left basis ``U``.
3. Solve the normal equations on the projected system:
   ``(B.T B + damping^2 I) z = B.T (U.T b)``.
4. Lift back: ``x = V z``.

Compared to the canonical LSMR recurrence (Fong, Saunders 2011) this
formulation lacks the Givens-rotation update that delivers the iterative
residual norms, but produces the same minimum-residual solution at the
``num_matvecs`` truncation step. matfree's ``lsmr`` exposes an
equivalent solution via a ``custom_vjp``-decorated Clenshaw-style
recursion; a JIT-compatible version of that recursion is deferred to a
follow-up commit when reverse-mode gradient support is needed.

Sibling reference: ``matfree/matfree/lstsq.py:18 lsmr``.

References
----------
* Fong, Saunders 2011 — *LSMR: An iterative algorithm for sparse
  least-squares problems*, SIAM J. Sci. Comput.
* Roy, Krämer et al. arXiv:2510.19634 — *Gradients of LSMR in JAX*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.linalg.krylov import golub_kahan_bidiag


if TYPE_CHECKING:
    from collections.abc import Callable


def lsmr(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    dim_cols: int,
    num_matvecs: int,
    damping: float = 0.0,
) -> jax.Array:
    """Solve ``min ||A x - rhs||^2 + damping^2 ||x||^2`` via bidiagonalised reduction.

    Args:
        matvec: callable computing ``A @ v`` for ``v`` of length ``dim_cols``.
        matvec_transpose: callable computing ``A.T @ u`` for ``u`` of
            length ``len(rhs)``.
        rhs: right-hand-side vector ``b``.
        dim_cols: number of columns of ``A`` (static); ``x`` has this length.
        num_matvecs: number of bidiagonalisation iterations (static).
            Larger values yield more accurate truncated solutions; for
            full-rank ``A`` with ``num_matvecs >= rank``, the routine
            recovers the exact least-squares solution.
        damping: Tikhonov damping coefficient ``λ`` (default ``0`` for
            plain least squares). Regularises ill-conditioned systems.

    Returns:
        Approximate solution ``x`` of shape ``(dim_cols,)``.
    """
    seed_right = matvec_transpose(rhs)
    left_basis, right_basis, bidiag = golub_kahan_bidiag(
        matvec=matvec,
        matvec_transpose=matvec_transpose,
        init_vec=seed_right,
        num_matvecs=num_matvecs,
    )
    projected_rhs = left_basis.T @ rhs
    damping_arr = jnp.asarray(damping, dtype=bidiag.dtype)
    normal_lhs = bidiag.T @ bidiag + damping_arr**2 * jnp.eye(num_matvecs)
    normal_rhs = bidiag.T @ projected_rhs
    small_solution = jnp.linalg.solve(normal_lhs, normal_rhs)
    return right_basis @ small_solution


__all__ = ["lsmr"]
