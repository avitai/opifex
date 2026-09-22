"""
2D Burgers Equation Solver

JAX-native implementation of the 2D viscous Burgers equation solver.
Uses finite difference methods with adaptive time stepping for stability.

Equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)
"""

from functools import partial

import jax
import jax.numpy as jnp


def solve_burgers_2d(
    initial_condition: jax.Array,
    viscosity: float | jax.Array,
    time_range: tuple[float, float] = (0.0, 2.0),
    time_steps: int = 5,
    resolution: int = 64,
) -> jax.Array:
    """
    Solve 2D Burgers equation using finite difference scheme.

    Uses forward Euler with upwind advection and central diffusion,
    with CFL-adaptive sub-stepping for numerical stability.

    Args:
        initial_condition: Initial condition u(x, y, 0)
        viscosity: Viscosity parameter
        time_range: (start_time, end_time)
        time_steps: Number of time steps to save
        resolution: Grid resolution

    Returns:
        Solution trajectory (time_steps+1, resolution, resolution) including initial
    """
    dx = 2.0 / (resolution - 1)
    save_times = jnp.linspace(time_range[0], time_range[1], time_steps + 1)

    def step_forward(u, dt_sub):
        """Single sub-step for 2D Burgers equation."""
        # Periodic boundary conditions
        u_padded = jnp.pad(u, ((1, 1), (1, 1)), mode="wrap")

        # Second derivatives (Laplacian)
        u_xx = (u_padded[2:, 1:-1] - 2 * u_padded[1:-1, 1:-1] + u_padded[:-2, 1:-1]) / dx**2
        u_yy = (u_padded[1:-1, 2:] - 2 * u_padded[1:-1, 1:-1] + u_padded[1:-1, :-2]) / dx**2

        # First derivatives (upwind scheme)
        u_x = jnp.where(
            u >= 0,
            (u - u_padded[1:-1, :-2]) / dx,  # Backward difference
            (u_padded[1:-1, 2:] - u) / dx,  # Forward difference
        )

        u_y = jnp.where(
            u >= 0,
            (u - u_padded[:-2, 1:-1]) / dx,  # Backward difference
            (u_padded[2:, 1:-1] - u) / dx,  # Forward difference
        )

        # Update equation: du/dt + u*du/dx + u*du/dy = nu*(d²u/dx² + d²u/dy²)
        return u - dt_sub * (u * u_x + u * u_y) + dt_sub * viscosity * (u_xx + u_yy)

    def compute_stable_dt(u):
        """Compute CFL-stable time step."""
        max_u = jnp.max(jnp.abs(u)) + 1e-8
        dt_advection = 0.4 * dx / max_u
        dt_diffusion = 0.25 * dx**2 / (viscosity + 1e-8)
        return jnp.minimum(dt_advection, dt_diffusion)

    # Adaptive sub-stepping between save times, expressed with lax.while_loop +
    # lax.scan so the solver is jit/vmap-compatible. The previous version called
    # float(dt_sub)/float(save_times[i]) inside a Python while loop, which forces
    # a host<->device sync every sub-step and cannot be jit-compiled (breaking
    # vmapped on-device data generation).
    def integrate_interval(u: jax.Array, bounds: jax.Array) -> tuple[jax.Array, jax.Array]:
        t_target = bounds[1]

        def not_done(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            return state[1] < t_target - 1e-12

        def sub_step(
            state: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            u_cur, t = state
            dt_sub = jnp.minimum(compute_stable_dt(u_cur), t_target - t)
            return step_forward(u_cur, dt_sub), t + dt_sub

        u_next, _ = jax.lax.while_loop(not_done, sub_step, (u, bounds[0]))
        return u_next, u_next

    interval_bounds = jnp.stack([save_times[:-1], save_times[1:]], axis=1)
    _, saved = jax.lax.scan(integrate_interval, initial_condition, interval_bounds)
    return jnp.concatenate([initial_condition[None], saved], axis=0)


class Burgers2DSolver:
    """
    2D viscous Burgers equation solver using JAX.

    Solves the nonlinear 2D Burgers equation using finite difference schemes
    with adaptive time stepping for numerical stability.

    Args:
        resolution: Grid resolution (number of points per dimension)
        domain_size: Physical domain size (default: 2π x 2π)
        viscosity: Kinematic viscosity coefficient
        dt_max: Maximum time step (adaptive stepping will use smaller values if needed)
    """

    def __init__(
        self,
        resolution: int = 64,
        domain_size: tuple[float, float] = (2.0 * jnp.pi, 2.0 * jnp.pi),
        viscosity: float = 0.01,
        dt_max: float = 0.001,
    ) -> None:
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        if not (isinstance(viscosity, int | float) and viscosity > 0):
            raise ValueError("viscosity must be a positive number")
        if not (isinstance(dt_max, int | float) and dt_max > 0):
            raise ValueError("dt_max must be a positive number")
        self.resolution = resolution
        self.domain_size = domain_size
        self.viscosity = viscosity
        self.dt_max = dt_max

        # Grid spacing
        self.dx = domain_size[0] / resolution
        self.dy = domain_size[1] / resolution

        # Create coordinate grids
        x = jnp.linspace(0, domain_size[0], resolution, endpoint=False)
        y = jnp.linspace(0, domain_size[1], resolution, endpoint=False)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

    def _compute_derivatives(self, u: jax.Array, v: jax.Array) -> tuple[jax.Array, ...]:
        """Compute spatial derivatives using finite differences with periodic BC."""
        # First derivatives (central difference with periodic boundaries)
        u_x = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * self.dx)
        u_y = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * self.dy)

        v_x = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * self.dx)
        v_y = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * self.dy)

        # Second derivatives (central difference)
        u_xx = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / (self.dx**2)
        u_yy = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / (self.dy**2)

        v_xx = (jnp.roll(v, -1, axis=1) - 2 * v + jnp.roll(v, 1, axis=1)) / (self.dx**2)
        v_yy = (jnp.roll(v, -1, axis=0) - 2 * v + jnp.roll(v, 1, axis=0)) / (self.dy**2)

        return u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy

    def _adaptive_time_step(self, u: jax.Array, v: jax.Array) -> jax.Array:
        """The largest stable step for this state, as a device value.

        Reading it on the host would sync every sub-step and leave the solve untraceable.
        """
        u_max = jnp.max(jnp.abs(u))
        v_max = jnp.max(jnp.abs(v))

        # CFL condition for advection
        dt_cfl = 0.5 * jnp.minimum(self.dx / (u_max + 1e-8), self.dy / (v_max + 1e-8))

        # Diffusion stability condition
        dt_diff = 0.25 * min(self.dx**2, self.dy**2) / self.viscosity

        # Use minimum of all constraints
        return jnp.minimum(jnp.asarray(self.dt_max), jnp.minimum(dt_cfl, dt_diff))

    @partial(jax.jit, static_argnums=(0,))
    def _rk4_step(
        self, state: tuple[jax.Array, jax.Array], dt: float
    ) -> tuple[jax.Array, jax.Array]:
        """Single Runge-Kutta 4th order time step."""
        u, v = state

        def rhs(u_curr, v_curr):
            """Right-hand side of the Burgers equation."""
            u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy = self._compute_derivatives(u_curr, v_curr)

            # Burgers equation RHS
            du_dt = -u_curr * u_x - v_curr * u_y + self.viscosity * (u_xx + u_yy)
            dv_dt = -u_curr * v_x - v_curr * v_y + self.viscosity * (v_xx + v_yy)

            return du_dt, dv_dt

        # RK4 stages
        k1_u, k1_v = rhs(u, v)
        k2_u, k2_v = rhs(u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v)
        k3_u, k3_v = rhs(u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v)
        k4_u, k4_v = rhs(u + dt * k3_u, v + dt * k3_v)

        # Update
        u_new = u + (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return u_new, v_new

    def solve(
        self,
        initial_condition: tuple[jax.Array, jax.Array],
        time_final: float,
        num_saves: int = 1,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Integrate from the initial condition to ``time_final``.

        The saved times are fixed, and the CFL-limited sub-steps between them run in a
        ``lax.while_loop``, so the whole solve traces under ``jit`` and maps under ``vmap``.

        Args:
            initial_condition: Tuple of (u0, v0) initial velocity fields.
            time_final: Final time to integrate to.
            num_saves: How many states after the initial one to return, equally spaced
                in time.

        Returns:
            Tuple of (times, u_trajectory, v_trajectory), each of length ``num_saves + 1``.

        Raises:
            ValueError: If a field does not have the solver's resolution, or ``num_saves``
                is not positive.
        """
        u, v = initial_condition
        expected_shape = (self.resolution, self.resolution)
        if u.shape != expected_shape:
            raise ValueError(f"u shape {u.shape} != expected {expected_shape}")
        if v.shape != expected_shape:
            raise ValueError(f"v shape {v.shape} != expected {expected_shape}")
        if num_saves < 1:
            raise ValueError(f"num_saves must be positive, got {num_saves}")

        save_times = jnp.linspace(0.0, time_final, num_saves + 1)

        def integrate_interval(
            state: tuple[jax.Array, jax.Array], bounds: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            target = bounds[1]

            def not_done(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
                return carry[2] < target - 1e-12

            def sub_step(
                carry: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                u_current, v_current, time = carry
                dt = jnp.minimum(self._adaptive_time_step(u_current, v_current), target - time)
                u_next, v_next = self._rk4_step((u_current, v_current), dt)
                return u_next, v_next, time + dt

            u_next, v_next, _ = jax.lax.while_loop(
                not_done, sub_step, (state[0], state[1], bounds[0])
            )
            return (u_next, v_next), (u_next, v_next)

        interval_bounds = jnp.stack([save_times[:-1], save_times[1:]], axis=1)
        _, (u_saved, v_saved) = jax.lax.scan(integrate_interval, (u, v), interval_bounds)
        return (
            save_times,
            jnp.concatenate([u[None], u_saved], axis=0),
            jnp.concatenate([v[None], v_saved], axis=0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def create_vortex_initial_condition(
        self, strength: float = 1.0, center: tuple[float, float] | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Create a vortex initial condition."""
        if center is None:
            center = (self.domain_size[0] / 2, self.domain_size[1] / 2)

        cx, cy = center

        # Distance from center
        r = jnp.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)

        # Vortex velocity field
        u = -strength * (self.Y - cy) * jnp.exp(-(r**2))
        v = strength * (self.X - cx) * jnp.exp(-(r**2))

        return u, v

    @partial(jax.jit, static_argnums=(0,))
    def create_shear_layer_initial_condition(
        self, shear_strength: float = 1.0
    ) -> tuple[jax.Array, jax.Array]:
        """Create a shear layer initial condition."""
        u = shear_strength * jnp.tanh((self.Y - self.domain_size[1] / 2) / 0.1)
        v = 0.1 * shear_strength * jnp.sin(2 * jnp.pi * self.X / self.domain_size[0])

        return u, v


__all__ = ["Burgers2DSolver", "solve_burgers_2d"]
