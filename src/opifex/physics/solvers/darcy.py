"""Darcy flow equation solver.

Solves the variable-coefficient elliptic PDE

    -∇·(a(x) ∇u(x)) = f(x),    u = 0 on ∂Ω,

with a constant unit source ``f ≡ 1`` and homogeneous Dirichlet boundary
conditions, on a uniform ``resolution × resolution`` grid.

The spatial operator is the standard conservative 5-point finite-difference
scheme with arithmetic-mean interface coefficients

    a_{i+1/2,j} = (a_{i,j} + a_{i+1,j}) / 2,

assembled into a sparse symmetric-positive-definite system and solved with a
**direct** ``float64`` sparse factorization (:func:`scipy.sparse.linalg.spsolve`).

A direct float64 solve is used deliberately: this routine generates *ground-truth*
labels for operator-learning datasets, and an iterative ``float32`` solver (the
previous Jacobi implementation) cannot produce accurate solutions for
high-contrast permeability — the system is ill-conditioned and ``float32``
conjugate-gradient stalls around a ``1e-4`` relative residual (≈30 % solution
error), silently corrupting the training labels. The direct solve converges the
PDE residual to machine precision, so the labels actually satisfy the PDE.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _assemble_system(coeff_field: np.ndarray, resolution: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Assemble the SPD sparse system ``A u_interior = h^2 f`` for the interior nodes.

    Args:
        coeff_field: Permeability ``a(x)`` of shape ``(resolution, resolution)``.
        resolution: Grid resolution.

    Returns:
        Tuple of the ``(m^2, m^2)`` CSR matrix and the right-hand side, where
        ``m = resolution - 2`` is the number of interior nodes per axis.
    """
    a = coeff_field
    m = resolution - 2
    h2 = (1.0 / (resolution - 1)) ** 2

    # Arithmetic-mean interface coefficients at the interior nodes.
    a_e = 0.5 * (a[1:-1, 1:-1] + a[1:-1, 2:])
    a_w = 0.5 * (a[1:-1, :-2] + a[1:-1, 1:-1])
    a_n = 0.5 * (a[:-2, 1:-1] + a[1:-1, 1:-1])
    a_s = 0.5 * (a[1:-1, 1:-1] + a[2:, 1:-1])
    a_sum = a_e + a_w + a_n + a_s

    index = np.arange(m * m).reshape(m, m)
    rows = [
        index.ravel(),
        index[:, :-1].ravel(),
        index[:, 1:].ravel(),
        index[1:, :].ravel(),
        index[:-1, :].ravel(),
    ]
    cols = [
        index.ravel(),
        index[:, 1:].ravel(),
        index[:, :-1].ravel(),
        index[:-1, :].ravel(),
        index[1:, :].ravel(),
    ]
    data = [
        a_sum.ravel(),
        -a_e[:, :-1].ravel(),
        -a_w[:, 1:].ravel(),
        -a_n[1:, :].ravel(),
        -a_s[:-1, :].ravel(),
    ]
    matrix = sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(m * m, m * m),
    )
    rhs = np.full(m * m, h2, dtype=np.float64)
    return matrix, rhs


def solve_darcy_flow(
    coeff_field: jax.Array,
    resolution: int,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> jax.Array:
    """Solve the Darcy flow equation ``-∇·(a∇u) = 1`` with zero Dirichlet BCs.

    Uses an exact ``float64`` sparse direct solve of the conservative 5-point
    discretization (see the module docstring for why a direct solve is required).

    Args:
        coeff_field: Positive permeability field ``a(x)`` of shape
            ``(resolution, resolution)``.
        resolution: Grid resolution.
        max_iter: Retained for API compatibility; ignored by the direct solver
            (an exact factorization needs no iteration count).
        tolerance: Retained for API compatibility; ignored by the direct solver.

    Returns:
        Solution field ``u(x)`` of shape ``(resolution, resolution)`` with zero
        Dirichlet boundary values.

    Raises:
        ValueError: If ``coeff_field`` is not a 2-D array, ``resolution`` is not a
            positive integer, or the shapes are inconsistent.
    """
    del max_iter, tolerance  # exact direct solve; accepted only for API compatibility

    if not (isinstance(coeff_field, jax.Array) and coeff_field.ndim == 2):
        raise ValueError("coeff_field must be a 2D array")
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if coeff_field.shape != (resolution, resolution):
        raise ValueError(
            f"coeff_field shape {coeff_field.shape} != expected ({resolution}, {resolution})"
        )

    matrix, rhs = _assemble_system(np.asarray(coeff_field, dtype=np.float64), resolution)
    interior = np.asarray(spla.spsolve(matrix.tocsc(), rhs))

    solution = np.zeros((resolution, resolution), dtype=np.float64)
    solution[1:-1, 1:-1] = interior.reshape(resolution - 2, resolution - 2)
    return jnp.asarray(solution)


__all__ = ["solve_darcy_flow"]
