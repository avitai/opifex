"""General exponential time-differencing (ETDRK4) integrator for semilinear PDEs.

Integrates ``u_t = L u + N(u)`` on a periodic torus, where ``L`` is *diagonal in
Fourier space* (a precomputed multiplier array, one entry per mode) and ``N`` is
a pseudo-spectral nonlinear term (a callable ``u_hat -> N_hat``). The fourth-order
ETDRK scheme of Cox & Matthews evaluates the exponential ``phi``-functions via the
Kassam-Trefethen contour integral so the coefficients stay accurate even where
``|L|`` is tiny (the naive ``(e^z - 1)/z`` forms cancel catastrophically there).

This module is deliberately FFT-agnostic: it performs only elementwise Fourier
arithmetic, so it works for any spatial dimensionality. The FFT conventions, the
derivative/Laplacian operators and the nonlinear terms live in
:mod:`opifex.physics.spectral.semilinear`; concrete PDE solvers live in
:mod:`opifex.physics.spectral.steppers`.

References:
    S. M. Cox, P. C. Matthews, "Exponential Time Differencing for Stiff
        Systems", J. Comput. Phys. 176, 430-455 (2002).
    A.-K. Kassam, L. N. Trefethen, "Fourth-Order Time-Stepping for Stiff
        PDEs", SIAM J. Sci. Comput. 26, 1214-1233 (2005).
    Reference JAX implementation: github.com/Ceyron/exponax (``exponax.etdrk``).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct


NonlinearFun = Callable[[jax.Array], jax.Array]


@struct.dataclass
class ETDRK4Coefficients:
    """Precomputed per-mode ETDRK4 coefficient arrays (all same shape as ``L``)."""

    exp_full: jax.Array  # exp(dt * L)
    exp_half: jax.Array  # exp(dt * L / 2)
    coef_q: jax.Array  # phi_1(dt L / 2) * dt / 2  (predictor weight)
    coef_f1: jax.Array  # final-update weight on N(u_n)
    coef_f2: jax.Array  # final-update weight on the two midpoint stages
    coef_f3: jax.Array  # final-update weight on N(c)


def etdrk4_coefficients(
    linear_operator: jax.Array,
    dt: float,
    *,
    num_contour_points: int = 32,
    contour_radius: float = 1.0,
) -> ETDRK4Coefficients:
    """Build the ETDRK4 coefficients for a Fourier-diagonal linear operator.

    The ``phi``-functions are averaged over ``num_contour_points`` equally spaced
    points on a circle of radius ``contour_radius`` centred on each ``dt * L``
    value (Kassam-Trefethen), which removes the cancellation error of the closed
    forms near ``L = 0`` (e.g. the zero Fourier mode).

    Args:
        linear_operator: Fourier-diagonal linear operator ``L`` (any shape).
        dt: Time-step size.
        num_contour_points: Number of contour points for the ``phi`` integrals.
        contour_radius: Radius of the contour circle.

    Returns:
        The precomputed :class:`ETDRK4Coefficients`.
    """
    dt_l = dt * linear_operator
    exp_full = jnp.exp(dt_l)
    exp_half = jnp.exp(dt_l / 2.0)

    roots = jnp.exp(
        2j * jnp.pi * (jnp.arange(1, num_contour_points + 1) - 0.5) / num_contour_points
    )
    # Contour points around each mode: shape (num_contour_points, *L.shape).
    contour = dt_l[None, ...] + contour_radius * roots.reshape(
        (num_contour_points,) + (1,) * dt_l.ndim
    )

    def _mean_real(values: jax.Array) -> jax.Array:
        return dt * jnp.mean(values.real, axis=0)

    coef_q = _mean_real((jnp.exp(contour / 2.0) - 1.0) / contour)
    exp_c = jnp.exp(contour)
    contour_cubed = contour**3
    coef_f1 = _mean_real(
        (-4.0 - contour + exp_c * (4.0 - 3.0 * contour + contour**2)) / contour_cubed
    )
    coef_f2 = _mean_real((2.0 + contour + exp_c * (-2.0 + contour)) / contour_cubed)
    coef_f3 = _mean_real(
        (-4.0 - 3.0 * contour - contour**2 + exp_c * (4.0 - contour)) / contour_cubed
    )
    return ETDRK4Coefficients(exp_full, exp_half, coef_q, coef_f1, coef_f2, coef_f3)


def etdrk4_step(
    u_hat: jax.Array,
    coefficients: ETDRK4Coefficients,
    nonlinear_fun: NonlinearFun,
) -> jax.Array:
    """Advance the Fourier-space state ``u_hat`` by one ETDRK4 step."""
    c = coefficients
    n_u = nonlinear_fun(u_hat)
    stage_a = c.exp_half * u_hat + c.coef_q * n_u
    n_a = nonlinear_fun(stage_a)
    stage_b = c.exp_half * u_hat + c.coef_q * n_a
    n_b = nonlinear_fun(stage_b)
    stage_c = c.exp_half * stage_a + c.coef_q * (2.0 * n_b - n_u)
    n_c = nonlinear_fun(stage_c)
    return c.exp_full * u_hat + c.coef_f1 * n_u + 2.0 * c.coef_f2 * (n_a + n_b) + c.coef_f3 * n_c


def integrate_etdrk4(
    linear_operator: jax.Array,
    nonlinear_fun: NonlinearFun,
    u_hat0: jax.Array,
    dt: float,
    num_steps: int,
    *,
    num_contour_points: int = 32,
) -> jax.Array:
    """Integrate ``u_t = L u + N(u)`` and return the full Fourier-space trajectory.

    Args:
        linear_operator: Fourier-diagonal ``L`` (broadcasts against ``u_hat0``).
        nonlinear_fun: Pseudo-spectral nonlinear term ``u_hat -> N_hat``.
        u_hat0: Initial Fourier-space state.
        dt: Time-step size.
        num_steps: Number of steps to take.
        num_contour_points: Contour points for the coefficient ``phi`` integrals.

    Returns:
        Trajectory of shape ``(num_steps, *u_hat0.shape)`` (excluding the initial
        state), one Fourier-space state per step. ``lax.scan``-based, so the call
        is ``jit``/``vmap``-compatible.
    """
    coefficients = etdrk4_coefficients(linear_operator, dt, num_contour_points=num_contour_points)

    def body(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_state = etdrk4_step(state, coefficients, nonlinear_fun)
        return next_state, next_state

    _, trajectory = jax.lax.scan(body, u_hat0, None, length=num_steps)
    return trajectory


__all__ = [
    "ETDRK4Coefficients",
    "NonlinearFun",
    "etdrk4_coefficients",
    "etdrk4_step",
    "integrate_etdrk4",
]
