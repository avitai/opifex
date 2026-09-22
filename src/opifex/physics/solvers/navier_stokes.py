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

The differential operators and the projection are the ones in ``opifex.fields``, which owns
their boundary handling and the consistency between the divergence the projection removes
and the gradient it removes it with. The domain is the periodic ``[0, 2 pi]`` square.

Reference:
    Chorin 1968 -- *Numerical solution of the Navier-Stokes equations*,
    Math. Comp. 22(104), 745.
"""

import jax
import jax.numpy as jnp

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import StaggeredGrid
from opifex.fields.staggered_navier_stokes import integrate


def solve_navier_stokes_2d(
    u0: jax.Array,
    v0: jax.Array,
    nu: float | jax.Array,
    time_range: tuple[float, float] = (0.0, 1.0),
    time_steps: int = 5,
    resolution: int = 64,
    substeps: int = 25,
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
        substeps: Equal sub-steps taken between consecutive save times (static)

    Returns:
        Tuple of (u_trajectory, v_trajectory) each of shape
        (time_steps+1, resolution, resolution) including initial condition
    """
    # The fields arrive on the node grid the initial-condition builders use, x_i = i*h,
    # and are carried on a staggered layout whose cells are centred on those nodes, so the
    # faces sit at (i+0.5)*h. Getting that alignment wrong costs O(h) silently: the node
    # grid and `CenteredGrid.cell_centers()` differ by exactly half a cell.
    extent = 2.0 * jnp.pi
    domain = Box(lower=(0.0, 0.0), upper=(float(extent), float(extent)))
    save_times = jnp.linspace(time_range[0], time_range[1], time_steps + 1)
    interval = (time_range[1] - time_range[0]) / time_steps

    def translate(values: jax.Array, axis: int, cells: float) -> jax.Array:
        """Translate by a fraction of a cell along ``axis``.

        A translation is a phase in Fourier space, so on a periodic grid this is exact for
        any field the grid resolves, where averaging the two neighbours is second order
        and -- being a mean -- smooths. That smoothing reads as decay: measured on a
        Taylor-Green vortex at 32 cells, averaging costs the viscous decay rate 9.5e-03
        relative where the shift costs 6.4e-05, a factor of a hundred and fifty, because
        the error of an amplitude-losing interpolation is indistinguishable from physical
        dissipation.
        """
        count = values.shape[axis]
        phase = jnp.exp(jnp.fft.fftfreq(count) * 2j * jnp.pi * cells)
        shape = [1] * values.ndim
        shape[axis] = count
        spectrum = jnp.fft.fft(values, axis=axis) * phase.reshape(shape)
        return jnp.real(jnp.fft.ifft(spectrum, axis=axis))

    def to_faces(u: jax.Array, v: jax.Array) -> StaggeredGrid:
        """Nodes to the faces half a cell above them."""
        return StaggeredGrid(
            (translate(u, 0, 0.5), translate(v, 1, 0.5)),
            domain,
            Extrapolation.PERIODIC,
            (resolution, resolution),
        )

    def to_nodes(field: StaggeredGrid) -> tuple[jax.Array, jax.Array]:
        """Faces back to the node half a cell below them."""
        u_face, v_face = field.components
        return translate(u_face, 0, -0.5), translate(v_face, 1, -0.5)

    def advance(
        state: StaggeredGrid, _bounds: jax.Array
    ) -> tuple[StaggeredGrid, tuple[jax.Array, jax.Array]]:
        stepped = integrate(state, interval, substeps, nu)
        return stepped, to_nodes(stepped)

    _, (u_saved, v_saved) = jax.lax.scan(advance, to_faces(u0, v0), save_times[1:])
    return (
        jnp.concatenate([u0[None], u_saved], axis=0),
        jnp.concatenate([v0[None], v_saved], axis=0),
    )


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
