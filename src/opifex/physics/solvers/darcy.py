"""
Darcy Flow Equation Solver

JAX-native implementation of Darcy flow solver using iterative methods.

Equation: -∇·(a(x)∇u(x)) = f(x)
where a(x) is the permeability coefficient field.

This is the standard elliptic form where positive source gives positive pressure.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(1, 2))
def solve_darcy_flow(
    coeff_field: jax.Array,
    resolution: int,
    max_iter: int = 50,
    tolerance: float = 1e-6,
) -> jax.Array:
    """
    Solve Darcy flow equation using Jacobi iterative method.

    Solves: -∇·(a(x)∇u(x)) = f(x) where a(x) is the permeability field.

    This is the standard elliptic PDE form where positive source terms f > 0
    produce positive solutions (with zero Dirichlet BCs).

    Uses a conservative finite difference scheme that properly accounts for
    the spatially varying permeability coefficient at cell interfaces.

    The discretization uses arithmetic mean for interface coefficients:
        a_{i+1/2,j} = (a_{i,j} + a_{i+1,j}) / 2

    Args:
        coeff_field: Permeability coefficient field (resolution x resolution).
                     Must be positive everywhere.
        resolution: Grid resolution
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance (currently not used for early stopping)

    Returns:
        Solution field u(x) with zero Dirichlet boundary conditions
    """
    if not (isinstance(coeff_field, jax.Array) and coeff_field.ndim == 2):
        raise ValueError("coeff_field must be a 2D array")
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if coeff_field.shape != (resolution, resolution):
        raise ValueError(
            f"coeff_field shape {coeff_field.shape} != expected "
            f"({resolution}, {resolution})"
        )

    h = 1.0 / (resolution - 1)
    h2 = h**2

    # Forcing term (constant source)
    f = jnp.ones((resolution, resolution))

    # Initialize solution with zero Dirichlet BC
    u = jnp.zeros((resolution, resolution))

    # Precompute interface coefficients using arithmetic mean
    # a_east[i,j] = (a[i,j] + a[i,j+1]) / 2  (coefficient at east interface)
    # a_west[i,j] = (a[i,j-1] + a[i,j]) / 2  (coefficient at west interface)
    # a_north[i,j] = (a[i-1,j] + a[i,j]) / 2 (coefficient at north interface)
    # a_south[i,j] = (a[i,j] + a[i+1,j]) / 2 (coefficient at south interface)

    # For interior points [1:-1, 1:-1]:
    a = coeff_field
    a_east = 0.5 * (a[1:-1, 1:-1] + a[1:-1, 2:])  # (i, j+1) neighbor
    a_west = 0.5 * (a[1:-1, :-2] + a[1:-1, 1:-1])  # (i, j-1) neighbor
    a_north = 0.5 * (a[:-2, 1:-1] + a[1:-1, 1:-1])  # (i-1, j) neighbor
    a_south = 0.5 * (a[1:-1, 1:-1] + a[2:, 1:-1])  # (i+1, j) neighbor

    # Sum of interface coefficients for the denominator
    a_sum = a_east + a_west + a_north + a_south

    # Forcing term at interior points
    f_interior = f[1:-1, 1:-1]

    # Jacobi iterative solver using lax.fori_loop for JAX compatibility
    def body_fn(_, u):
        # Get neighbor values
        u_east = u[1:-1, 2:]  # u[i, j+1]
        u_west = u[1:-1, :-2]  # u[i, j-1]
        u_north = u[:-2, 1:-1]  # u[i-1, j]
        u_south = u[2:, 1:-1]  # u[i+1, j]

        # Conservative finite difference scheme for -∇·(a∇u) = f
        # u_new = (a_E*u_E + a_W*u_W + a_N*u_N + a_S*u_S + h^2*f) / (a_E+a_W+a_N+a_S)
        numerator = (
            a_east * u_east
            + a_west * u_west
            + a_north * u_north
            + a_south * u_south
            + h2 * f_interior
        )
        u_new_inner = numerator / a_sum

        # Update interior points (boundary stays at zero)
        return u.at[1:-1, 1:-1].set(u_new_inner)

    # Run iterations using JAX's fori_loop (JIT-compatible)
    return jax.lax.fori_loop(0, max_iter, body_fn, u)


__all__ = ["solve_darcy_flow"]
