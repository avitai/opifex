"""Pseudo-spectral ETDRK4 solvers for canonical 1D periodic PDEs.

Each solver is a thin assembly over the general ETDRK4 integrator
(:mod:`opifex.physics.spectral.etdrk`) and the Fourier operators
(:mod:`opifex.physics.spectral.semilinear`): it builds the Fourier-diagonal
linear operator ``L`` and the pseudo-spectral nonlinear term ``N`` for its PDE,
rolls out the trajectory, and returns evenly spaced real-space snapshots with the
initial condition prepended (the convention of the finite-difference solvers in
:mod:`opifex.physics.solvers`). All solvers are ``jit``/``vmap``-compatible.

The three PDEs span the distinct nonlinearity/linear-operator shapes the framework
supports: viscous Burgers (diffusion + convection), Kuramoto-Sivashinsky
(anti-diffusion + hyper-diffusion + gradient-norm), and Korteweg-de Vries
(dispersion + convection).
"""

import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.spectral.etdrk import integrate_etdrk4, NonlinearFun
from opifex.physics.spectral.semilinear import (
    convection_nonlinearity,
    gradient_norm_nonlinearity,
    laplace_operator,
    third_derivative_operator,
)


def _rollout_real(
    linear_operator: jax.Array,
    nonlinear_fun: NonlinearFun,
    initial_condition: jax.Array,
    *,
    dt: float,
    num_steps: int,
    num_snapshots: int,
    num_contour_points: int,
) -> jax.Array:
    """Integrate one PDE and return ``(num_snapshots + 1, N)`` real snapshots.

    The initial condition is prepended; the remaining ``num_snapshots`` frames are
    sampled evenly (by step index) from the integrated trajectory.
    """
    num_points = initial_condition.shape[-1]
    u_hat0 = jnp.fft.rfft(initial_condition, axis=-1)
    trajectory_hat = integrate_etdrk4(
        linear_operator,
        nonlinear_fun,
        u_hat0,
        dt,
        num_steps,
        num_contour_points=num_contour_points,
    )
    # Evenly spaced snapshots whose last entry is the final state. Trajectory
    # frame ``i`` is the state after ``i + 1`` steps, so snapshot ``j`` of
    # ``num_snapshots`` lands on step index ``round(num_steps * j / k) - 1``.
    save_steps = np.arange(1, num_snapshots + 1) * num_steps / num_snapshots
    save_indices = np.round(save_steps).astype(int) - 1
    snapshots = jnp.fft.irfft(trajectory_hat[save_indices], n=num_points, axis=-1)
    return jnp.concatenate([initial_condition[None], snapshots], axis=0)


def solve_burgers_spectral(
    initial_condition: jax.Array,
    viscosity: float | jax.Array,
    *,
    domain_extent: float = 1.0,
    time_final: float = 1.0,
    num_steps: int = 250,
    num_snapshots: int = 1,
    dealias_fraction: float = 2.0 / 3.0,
    num_contour_points: int = 32,
) -> jax.Array:
    """Solve the viscous Burgers equation ``u_t + u u_x = nu u_xx`` (periodic).

    Args:
        initial_condition: Real field ``u(x, 0)``, shape ``(N,)``.
        viscosity: Kinematic viscosity ``nu``.
        domain_extent: Length of the periodic domain ``[0, L)``.
        time_final: Final integration time.
        num_steps: Number of ETDRK4 steps (accuracy/cost trade-off).
        num_snapshots: Number of saved frames after the initial condition.
        dealias_fraction: Orszag dealiasing fraction for the convection product.
        num_contour_points: Contour points for the ETDRK4 coefficients.

    Returns:
        Real snapshots of shape ``(num_snapshots + 1, N)`` including ``u(x, 0)``.
    """
    num_points = initial_condition.shape[-1]
    linear_operator = viscosity * laplace_operator(num_points, domain_extent)
    nonlinear_fun = convection_nonlinearity(
        num_points, domain_extent, scale=1.0, dealias_fraction=dealias_fraction
    )
    return _rollout_real(
        linear_operator,
        nonlinear_fun,
        initial_condition,
        dt=time_final / num_steps,
        num_steps=num_steps,
        num_snapshots=num_snapshots,
        num_contour_points=num_contour_points,
    )


def solve_kuramoto_sivashinsky_spectral(
    initial_condition: jax.Array,
    *,
    domain_extent: float = 32.0 * np.pi,
    time_final: float = 50.0,
    num_steps: int = 1000,
    num_snapshots: int = 1,
    dealias_fraction: float = 2.0 / 3.0,
    num_contour_points: int = 32,
) -> jax.Array:
    """Solve Kuramoto-Sivashinsky ``u_t + u_xx + u_xxxx + (1/2)(u_x)^2 = 0``.

    The linear part ``-u_xx - u_xxxx`` (symbol ``k^2 - k^4``) is anti-diffusive at
    low modes and hyper-diffusive at high modes; on a large domain the equation is
    spatio-temporally chaotic. Returns ``(num_snapshots + 1, N)`` real snapshots.
    """
    num_points = initial_condition.shape[-1]
    laplacian = laplace_operator(num_points, domain_extent)
    linear_operator = -laplacian - laplacian**2  # k^2 - k^4
    nonlinear_fun = gradient_norm_nonlinearity(
        num_points, domain_extent, scale=1.0, dealias_fraction=dealias_fraction
    )
    return _rollout_real(
        linear_operator,
        nonlinear_fun,
        initial_condition,
        dt=time_final / num_steps,
        num_steps=num_steps,
        num_snapshots=num_snapshots,
        num_contour_points=num_contour_points,
    )


def solve_kdv_spectral(
    initial_condition: jax.Array,
    *,
    domain_extent: float = 2.0 * np.pi,
    time_final: float = 1.0,
    num_steps: int = 1000,
    num_snapshots: int = 1,
    dealias_fraction: float = 2.0 / 3.0,
    num_contour_points: int = 32,
) -> jax.Array:
    """Solve the Korteweg-de Vries equation ``u_t + 6 u u_x + u_xxx = 0``.

    Dispersive (symbol ``i k^3``) with quadratic convection; admits solitons that
    propagate at speed proportional to their amplitude. Returns
    ``(num_snapshots + 1, N)`` real snapshots.
    """
    num_points = initial_condition.shape[-1]
    linear_operator = third_derivative_operator(num_points, domain_extent)
    nonlinear_fun = convection_nonlinearity(
        num_points, domain_extent, scale=6.0, dealias_fraction=dealias_fraction
    )
    return _rollout_real(
        linear_operator,
        nonlinear_fun,
        initial_condition,
        dt=time_final / num_steps,
        num_steps=num_steps,
        num_snapshots=num_snapshots,
        num_contour_points=num_contour_points,
    )


__all__ = [
    "solve_burgers_spectral",
    "solve_kdv_spectral",
    "solve_kuramoto_sivashinsky_spectral",
]
