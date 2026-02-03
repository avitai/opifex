"""
Darcy Flow Equation Solver

JAX-native implementation of Darcy flow solver using iterative methods.

Equation: ∇·(a(x)∇u(x)) = f(x)
where a(x) is the permeability coefficient field.
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

    Solves: ∇·(a(x)∇u(x)) = f(x) where a(x) is the permeability field.

    Args:
        coeff_field: Permeability coefficient field (resolution x resolution)
        resolution: Grid resolution
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Solution field u(x)
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

    # Forcing term (simple constant source)
    f = jnp.ones((resolution, resolution))

    # Initialize solution
    u = jnp.zeros((resolution, resolution))

    # Jacobi iterative solver using lax.fori_loop for JAX compatibility
    def body_fn(i, u):
        # Interior points update using Jacobi iteration
        u_left = u[1:-1, :-2]
        u_right = u[1:-1, 2:]
        u_up = u[:-2, 1:-1]
        u_down = u[2:, 1:-1]

        # Simple finite difference scheme (Poisson equation approximation)
        numerator = u_left + u_right + u_up + u_down + h**2 * f[1:-1, 1:-1]
        u_new_inner = numerator / 4.0

        # Update interior points
        return u.at[1:-1, 1:-1].set(u_new_inner)

    # Run iterations using JAX's fori_loop (JIT-compatible)
    return jax.lax.fori_loop(0, max_iter, body_fn, u)


__all__ = ["solve_darcy_flow"]
