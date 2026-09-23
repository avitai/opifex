"""Exact Fourier solution of linear advection-diffusion on a periodic domain.

For ``u_t + v . grad u = D lap u`` on ``[0, L)^d`` with constant ``D`` and ``v``,
every Fourier mode evolves independently::

    u_hat(k, t) = exp((-D |k|^2 - i v . k) t) u_hat(k, 0)

so the solution at any time is one forward FFT, one multiplication and one
inverse FFT -- no time stepping, no stability limit and no truncation error
beyond the representation of the initial condition. This is the exponential
propagator that exponential integrators apply to the linear part of a PDE
(Cox & Matthews 2002, J. Comput. Phys. 176, 430); for a linear PDE it is the
whole solution.

The result is the exact evolution of the band-limited trigonometric interpolant
of the initial data (Trefethen 2000, *Spectral Methods in MATLAB*, ch. 3). As in
:mod:`opifex.physics.spectral.semilinear`, the first-derivative symbol is zeroed
at the Nyquist mode of an even-length axis, which carries no well-defined sign
for a real field.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from opifex.core.spectral import wavenumber_grid


def _advection_diffusion_symbol(
    shape: tuple[int, ...],
    domain_extent: float,
    diffusivity: jax.Array,
    velocity: jax.Array,
) -> jax.Array:
    """Fourier symbol ``-D |k|^2 - i v . k`` on the ``rfftn`` layout of ``shape``."""
    wavenumbers = wavenumber_grid(shape, tuple(domain_extent / n for n in shape))
    axes = jnp.meshgrid(*wavenumbers, indexing="ij")
    squared_norm = sum(k**2 for k in axes)
    advection = jnp.zeros_like(axes[0])
    for axis, (k, n) in enumerate(zip(axes, shape, strict=True)):
        odd_symbol = k
        if n % 2 == 0:
            nyquist: list[slice | int] = [slice(None)] * len(shape)
            nyquist[axis] = n // 2
            odd_symbol = odd_symbol.at[tuple(nyquist)].set(0.0)
        advection = advection + velocity[axis] * odd_symbol
    return -diffusivity * squared_norm - 1j * advection


def solve_advection_diffusion_exact(
    initial_condition: ArrayLike,
    diffusivity: ArrayLike,
    velocity: ArrayLike,
    *,
    domain_extent: float = 1.0,
    time_final: float = 1.0,
) -> jax.Array:
    """Solve ``u_t + v . grad u = D lap u`` exactly on a periodic domain.

    Args:
        initial_condition: Real field ``u(x, 0)`` sampled at ``x_j = j L / N`` on
            each axis (1 to 3 spatial dimensions, right endpoint excluded).
        diffusivity: Diffusion coefficient ``D >= 0``.
        velocity: Advection velocity ``v``, one component per spatial axis.
        domain_extent: Period ``L`` of every axis.
        time_final: Time ``t`` at which the solution is returned.

    Returns:
        ``u(x, t)`` on the same grid, with the dtype of ``initial_condition``.

    Raises:
        ValueError: If ``velocity`` does not have one component per spatial axis.
    """
    field = jnp.asarray(initial_condition)
    velocity = jnp.asarray(velocity)
    if velocity.shape != (field.ndim,):
        raise ValueError(
            f"velocity must have shape ({field.ndim},) for a {field.ndim}-D field, "
            f"got {velocity.shape}"
        )
    symbol = _advection_diffusion_symbol(
        field.shape, domain_extent, jnp.asarray(diffusivity), velocity
    )
    field_hat = jnp.fft.rfftn(field)
    evolved = jnp.fft.irfftn(field_hat * jnp.exp(symbol * time_final), s=field.shape)
    return evolved.astype(field.dtype)
