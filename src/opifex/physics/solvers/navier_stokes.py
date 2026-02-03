"""
2D Incompressible Navier-Stokes Equation Solver

JAX-native implementation of the 2D incompressible Navier-Stokes equations
using a projection method for pressure correction.

Equations:
    du/dt + (u·∇)u = -∇p/ρ + ν∇²u
    ∇·u = 0 (incompressibility)

The solver uses a fractional step (projection) method:
1. Compute tentative velocity without pressure
2. Solve pressure Poisson equation
3. Correct velocity to be divergence-free
"""

import jax
import jax.numpy as jnp


def _laplacian(f: jax.Array, dx: float) -> jax.Array:
    """Compute Laplacian with periodic boundaries."""
    return (
        jnp.roll(f, 1, axis=0)
        + jnp.roll(f, -1, axis=0)
        + jnp.roll(f, 1, axis=1)
        + jnp.roll(f, -1, axis=1)
        - 4 * f
    ) / dx**2


def _divergence(u: jax.Array, v: jax.Array, dx: float) -> jax.Array:
    """Compute divergence of velocity field."""
    du_dx = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx)
    dv_dy = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dx)
    return du_dx + dv_dy


def _gradient(p: jax.Array, dx: float) -> tuple[jax.Array, jax.Array]:
    """Compute gradient of scalar field."""
    dp_dx = (jnp.roll(p, -1, axis=0) - jnp.roll(p, 1, axis=0)) / (2 * dx)
    dp_dy = (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1)) / (2 * dx)
    return dp_dx, dp_dy


def _solve_pressure_poisson(
    div_u: jax.Array, dx: float, n_iters: int = 50
) -> jax.Array:
    """Solve pressure Poisson equation: nabla^2 p = nabla dot u using Jacobi."""
    p = jnp.zeros_like(div_u)
    for _ in range(n_iters):
        p_new = (
            jnp.roll(p, 1, axis=0)
            + jnp.roll(p, -1, axis=0)
            + jnp.roll(p, 1, axis=1)
            + jnp.roll(p, -1, axis=1)
            - dx**2 * div_u
        ) / 4
        p_new = p_new - jnp.mean(p_new)
        p = p_new
    return p


def _advection_term(f: jax.Array, u: jax.Array, v: jax.Array, dx: float) -> jax.Array:
    """Compute advection (u dot nabla)f using upwind scheme."""
    f_x_backward = (f - jnp.roll(f, 1, axis=0)) / dx
    f_x_forward = (jnp.roll(f, -1, axis=0) - f) / dx
    f_x = jnp.where(u >= 0, f_x_backward, f_x_forward)

    f_y_backward = (f - jnp.roll(f, 1, axis=1)) / dx
    f_y_forward = (jnp.roll(f, -1, axis=1) - f) / dx
    f_y = jnp.where(v >= 0, f_y_backward, f_y_forward)

    return u * f_x + v * f_y


def solve_navier_stokes_2d(
    u0: jax.Array,
    v0: jax.Array,
    nu: float,
    time_range: tuple[float, float] = (0.0, 1.0),
    time_steps: int = 5,
    resolution: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """
    Solve 2D incompressible Navier-Stokes equations.

    Uses a projection method with finite differences on a periodic domain.
    The domain is [0, 2π] × [0, 2π] with periodic boundary conditions.

    Args:
        u0: Initial x-velocity field, shape (resolution, resolution)
        v0: Initial y-velocity field, shape (resolution, resolution)
        nu: Kinematic viscosity (ν = μ/ρ)
        time_range: (start_time, end_time)
        time_steps: Number of time steps to save
        resolution: Grid resolution (should match u0, v0)

    Returns:
        Tuple of (u_trajectory, v_trajectory) each of shape
        (time_steps+1, resolution, resolution) including initial condition
    """
    # Grid setup
    L = 2 * jnp.pi  # Domain size
    dx = L / resolution
    save_times = jnp.linspace(time_range[0], time_range[1], time_steps + 1)

    def compute_cfl_dt(u, v):
        """Compute stable time step based on CFL condition."""
        max_vel = jnp.maximum(jnp.max(jnp.abs(u)), jnp.max(jnp.abs(v))) + 1e-8
        dt_advection = 0.3 * dx / max_vel
        dt_diffusion = 0.2 * dx**2 / (nu + 1e-8)
        return jnp.minimum(dt_advection, dt_diffusion)

    def step_forward(u, v, dt):
        """Single time step using projection method."""
        # Step 1: Compute tentative velocity (without pressure)
        u_star = u + dt * (-_advection_term(u, u, v, dx) + nu * _laplacian(u, dx))
        v_star = v + dt * (-_advection_term(v, u, v, dx) + nu * _laplacian(v, dx))

        # Step 2: Solve pressure Poisson equation
        div_u_star = _divergence(u_star, v_star, dx)
        p = _solve_pressure_poisson(div_u_star / dt, dx)

        # Step 3: Correct velocity to be divergence-free
        dp_dx, dp_dy = _gradient(p, dx)
        u_new = u_star - dt * dp_dx
        v_new = v_star - dt * dp_dy

        return u_new, v_new

    # Time stepping with sub-stepping for stability
    u, v = u0, v0
    u_trajectory = [u]
    v_trajectory = [v]

    for i in range(time_steps):
        t_current = float(save_times[i])
        t_target = float(save_times[i + 1])
        t = t_current

        while t < t_target - 1e-12:
            dt_sub = compute_cfl_dt(u, v)
            dt_sub = jnp.minimum(dt_sub, t_target - t)
            u, v = step_forward(u, v, float(dt_sub))
            t += float(dt_sub)

        u_trajectory.append(u)
        v_trajectory.append(v)

    return jnp.stack(u_trajectory), jnp.stack(v_trajectory)


def create_taylor_green_vortex(
    resolution: int,
    amplitude: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Create Taylor-Green vortex initial condition.

    The Taylor-Green vortex is an exact solution of the NS equations at t=0
    and decays exponentially due to viscosity. It satisfies incompressibility.

    u = A * sin(x) * cos(y)
    v = -A * cos(x) * sin(y)

    Args:
        resolution: Grid resolution
        amplitude: Velocity amplitude

    Returns:
        Tuple of (u0, v0) initial velocity fields
    """
    x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    u0 = amplitude * jnp.sin(X) * jnp.cos(Y)
    v0 = -amplitude * jnp.cos(X) * jnp.sin(Y)

    return u0, v0


def create_lid_driven_cavity_ic(
    resolution: int,
    lid_velocity: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Create lid-driven cavity initial condition.

    For lid-driven cavity, the top boundary has a specified velocity
    while all other boundaries are no-slip. This is an approximation
    using a smooth profile since we use periodic boundaries.

    Args:
        resolution: Grid resolution
        lid_velocity: Velocity of the lid (top boundary)

    Returns:
        Tuple of (u0, v0) initial velocity fields
    """
    # Start with quiescent flow
    v0 = jnp.zeros((resolution, resolution))

    # Add a smooth velocity profile near the top
    # Using tanh to create a smooth boundary layer
    # y-axis is the second dimension in (x, y) = (axis 0, axis 1)
    y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    y_profile = 0.5 * (1 + jnp.tanh(10 * (y / (2 * jnp.pi) - 0.9)))

    # Broadcast to full 2D array: (1, res) * (res, 1) = (res, res)
    # But we want constant in x direction, varying in y
    # So we use ones for x and y_profile for y
    u0 = lid_velocity * jnp.ones((resolution, 1)) * y_profile[None, :]

    return u0, v0


def create_double_shear_layer(
    resolution: int,
    shear_thickness: float = 0.05,
    perturbation: float = 0.05,
) -> tuple[jax.Array, jax.Array]:
    """
    Create double shear layer initial condition.

    A classic test case for 2D turbulence that develops
    Kelvin-Helmholtz instabilities.

    Args:
        resolution: Grid resolution
        shear_thickness: Thickness of the shear layers
        perturbation: Amplitude of initial perturbation

    Returns:
        Tuple of (u0, v0) initial velocity fields
    """
    x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Double shear layer in x-velocity
    delta = shear_thickness * 2 * jnp.pi
    u0 = jnp.where(
        jnp.pi > Y,
        jnp.tanh((Y - jnp.pi / 2) / delta),
        jnp.tanh((3 * jnp.pi / 2 - Y) / delta),
    )

    # Small perturbation in y-velocity
    v0 = perturbation * jnp.sin(X)

    return u0, v0


__all__ = [
    "create_double_shear_layer",
    "create_lid_driven_cavity_ic",
    "create_taylor_green_vortex",
    "solve_navier_stokes_2d",
]
