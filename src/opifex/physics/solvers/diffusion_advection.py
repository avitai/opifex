"""
Diffusion-advection equation solver.

This module provides JAX-native solvers for diffusion-advection PDEs.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3, 4))
def _solve_diffusion_advection_2d_jit(
    initial_condition: jax.Array,
    diffusion_coeff: float,
    advection_vel: tuple[float, float],
    n_steps: int,
    grid_spacing: float,
    dt: float,
) -> jax.Array:
    """
    JIT-compiled core solver for 2D diffusion-advection equation.

    Internal function - use solve_diffusion_advection_2d for public API.
    """
    vx, vy = advection_vel
    dx = dy = grid_spacing

    def step_fn(u, _):
        """Single time step of diffusion-advection equation."""
        # Compute spatial derivatives using finite differences
        # Advection terms (upwind scheme for stability)
        u_x = jnp.where(
            vx >= 0,
            (u - jnp.roll(u, 1, axis=1)) / dx,
            (jnp.roll(u, -1, axis=1) - u) / dx,
        )
        u_y = jnp.where(
            vy >= 0,
            (u - jnp.roll(u, 1, axis=0)) / dy,
            (jnp.roll(u, -1, axis=0) - u) / dy,
        )

        # Diffusion terms (central differences)
        u_xx = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / (dx**2)
        u_yy = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / (dy**2)

        # Diffusion-advection equation: ∂u/∂t = D∇²u - v·∇u
        dudt = diffusion_coeff * (u_xx + u_yy) - vx * u_x - vy * u_y

        return u + dt * dudt, None

    final_state, _ = jax.lax.scan(step_fn, initial_condition, None, length=n_steps)
    return final_state


def solve_diffusion_advection_2d(
    initial_condition: jax.Array,
    diffusion_coeff: float,
    advection_vel: tuple[float, float],
    dt: float = 0.001,
    n_steps: int = 1000,
    grid_spacing: float = 1.0,
) -> jax.Array:
    """
    Solve 2D diffusion-advection equation using finite differences.

    JIT-compiled for optimal performance on the core PDE solver.

    Args:
        initial_condition: Initial field values
        diffusion_coeff: Diffusion coefficient
        advection_vel: (vx, vy) advection velocities
        dt: Time step
        n_steps: Number of time steps
        grid_spacing: Spatial grid spacing

    Returns:
        Solution at final time
    """
    # Input validation (outside JIT for proper error handling)
    if not (isinstance(initial_condition, jax.Array) and initial_condition.ndim == 2):
        raise ValueError("initial_condition must be a 2D array")
    if not (isinstance(diffusion_coeff, (int, float)) and diffusion_coeff > 0):
        raise ValueError("diffusion_coeff must be a positive number")
    if not (isinstance(dt, (int, float)) and dt > 0):
        raise ValueError("dt must be a positive number")
    if not (isinstance(n_steps, int) and n_steps > 0):
        raise ValueError("n_steps must be a positive integer")
    if not (isinstance(grid_spacing, (int, float)) and grid_spacing > 0):
        raise ValueError("grid_spacing must be a positive number")
    if not (
        isinstance(advection_vel, tuple)
        and len(advection_vel) == 2
        and all(isinstance(v, (int, float)) for v in advection_vel)
    ):
        raise ValueError("advection_vel must be a tuple of two numbers")

    # Call JIT-compiled solver
    return _solve_diffusion_advection_2d_jit(
        initial_condition, diffusion_coeff, advection_vel, n_steps, grid_spacing, dt
    )


__all__ = ["solve_diffusion_advection_2d"]
