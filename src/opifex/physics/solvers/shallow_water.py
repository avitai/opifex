"""
Shallow water equations solver.

This module provides JAX-native solvers for shallow water equations.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def solve_shallow_water_2d(
    h_initial: jax.Array,
    u_initial: jax.Array,
    v_initial: jax.Array,
    g: float = 9.81,
    dt: float = 0.001,
    n_steps: int = 1000,
    grid_spacing: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Solve 2D shallow water equations using finite differences.

    JIT-compiled for optimal performance on the shallow water PDE system.

    Args:
        h_initial: Initial height field
        u_initial: Initial u-velocity field
        v_initial: Initial v-velocity field
        g: Gravitational acceleration
        dt: Time step
        n_steps: Number of time steps
        grid_spacing: Spatial grid spacing

    Returns:
        Tuple of (height, u_velocity, v_velocity) at final time
    """
    # Input validation
    if not (isinstance(h_initial, jax.Array) and h_initial.ndim == 2):
        raise ValueError("h_initial must be a 2D array")
    if not (isinstance(u_initial, jax.Array) and u_initial.ndim == 2):
        raise ValueError("u_initial must be a 2D array")
    if not (isinstance(v_initial, jax.Array) and v_initial.ndim == 2):
        raise ValueError("v_initial must be a 2D array")
    if not (h_initial.shape == u_initial.shape == v_initial.shape):
        raise ValueError("All input fields must have the same shape")
    if not (isinstance(g, (int, float)) and g > 0):
        raise ValueError("g must be a positive number")
    if not (isinstance(dt, (int, float)) and dt > 0):
        raise ValueError("dt must be a positive number")
    if not (isinstance(n_steps, int) and n_steps > 0):
        raise ValueError("n_steps must be a positive integer")
    if not (isinstance(grid_spacing, (int, float)) and grid_spacing > 0):
        raise ValueError("grid_spacing must be a positive number")

    dx = dy = grid_spacing

    def step_fn(state):
        """Single time step of shallow water equations."""
        h, u, v = state

        # Compute spatial derivatives using finite differences
        # Height derivatives
        h_x = (jnp.roll(h, -1, axis=1) - jnp.roll(h, 1, axis=1)) / (2 * dx)
        h_y = (jnp.roll(h, -1, axis=0) - jnp.roll(h, 1, axis=0)) / (2 * dy)

        # Velocity derivatives
        u_x = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dx)
        u_y = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dy)
        v_x = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dx)
        v_y = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * dy)

        # Shallow water equations:
        # ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
        # ∂u/∂t + u∂u/∂x + v∂u/∂y + g∂h/∂x = 0
        # ∂v/∂t + u∂v/∂x + v∂v/∂y + g∂h/∂y = 0

        dhdt = -(h * (u_x + v_y) + u * h_x + v * h_y)
        dudt = -(u * u_x + v * u_y + g * h_x)
        dvdt = -(u * v_x + v * v_y + g * h_y)

        # Update using forward Euler
        h_new = h + dt * dhdt
        u_new = u + dt * dudt
        v_new = v + dt * dvdt

        return (h_new, u_new, v_new)

    # Time integration using scan for memory efficiency
    def scan_step(state, _):
        new_state = step_fn(state)
        return new_state, None

    initial_state = (h_initial, u_initial, v_initial)
    final_state, _ = jax.lax.scan(scan_step, initial_state, None, length=n_steps)

    return final_state


__all__ = ["solve_shallow_water_2d"]
