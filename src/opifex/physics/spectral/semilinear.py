"""Fourier operators and pseudo-spectral nonlinear terms for 1D periodic PDEs.

Building blocks for :mod:`opifex.physics.spectral.steppers`: the Fourier-diagonal
linear operators (first derivative, Laplacian, third derivative) and the
pseudo-spectral nonlinear terms (convection ``-s/2 d/dx(u^2)`` and gradient-norm
``-s/2 (du/dx)^2``) consumed by the ETDRK integrator. The wavenumber convention
is reused from :mod:`opifex.core.spectral` (``wavenumber_grid`` returns the
real-FFT physical wavenumbers ``2*pi*m/L`` for ``m = 0 .. N/2``), so this module
adds no duplicate FFT-frequency logic.

All operators target the ``rfft`` spectrum (length ``N//2 + 1``); fields are real.
Odd-derivative operators zero the Nyquist mode (even ``N``), which carries no
well-defined sign for a real field, and the nonlinear terms apply Orszag's
two-thirds dealiasing rule around the quadratic products.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from opifex.core.spectral import wavenumber_grid


NonlinearFun = Callable[[jax.Array], jax.Array]


def rfft_wavenumbers(num_points: int, domain_extent: float) -> jax.Array:
    """Physical real-FFT wavenumbers ``2*pi*m/L`` for ``m = 0 .. N//2``."""
    grid = wavenumber_grid((num_points,), domain_extent / num_points)
    return jnp.asarray(grid[0] if isinstance(grid, list) else grid)


def first_derivative_operator(num_points: int, domain_extent: float) -> jax.Array:
    """Fourier symbol ``i*k`` of ``d/dx`` (Nyquist zeroed for even ``N``)."""
    derivative = 1j * rfft_wavenumbers(num_points, domain_extent)
    if num_points % 2 == 0:
        derivative = derivative.at[-1].set(0.0)
    return derivative


def laplace_operator(num_points: int, domain_extent: float) -> jax.Array:
    """Fourier symbol ``-k^2`` of the Laplacian ``d^2/dx^2``."""
    return -(rfft_wavenumbers(num_points, domain_extent) ** 2)


def third_derivative_operator(num_points: int, domain_extent: float) -> jax.Array:
    """Fourier symbol ``i*k^3`` of ``d^3/dx^3`` (Nyquist zeroed for even ``N``)."""
    wavenumbers = rfft_wavenumbers(num_points, domain_extent)
    derivative = 1j * wavenumbers**3
    if num_points % 2 == 0:
        derivative = derivative.at[-1].set(0.0)
    return derivative


def dealias_mask(num_points: int, fraction: float = 2.0 / 3.0) -> jax.Array:
    """Orszag two-thirds low-pass mask over the ``rfft`` modes (boolean array)."""
    mode_indices = jnp.arange(num_points // 2 + 1)
    cutoff = fraction * (num_points // 2)
    return mode_indices <= cutoff


def convection_nonlinearity(
    num_points: int,
    domain_extent: float,
    *,
    scale: float = 1.0,
    dealias_fraction: float = 2.0 / 3.0,
) -> NonlinearFun:
    """Conservative convection term ``N(u) = -scale/2 * d/dx(u^2)`` (dealiased).

    Used by Burgers (``scale = 1``) and KdV (``scale = 6``). The quadratic product
    is formed in physical space with two-thirds dealiasing applied before the
    inverse transform and after the forward transform.
    """
    derivative = first_derivative_operator(num_points, domain_extent)
    mask = dealias_mask(num_points, dealias_fraction)

    def nonlinear_fun(u_hat: jax.Array) -> jax.Array:
        u = jnp.fft.irfft(mask * u_hat, n=num_points, axis=-1)
        flux_hat = mask * jnp.fft.rfft(0.5 * u**2, axis=-1)
        return -scale * derivative * flux_hat

    return nonlinear_fun


def gradient_norm_nonlinearity(
    num_points: int,
    domain_extent: float,
    *,
    scale: float = 1.0,
    dealias_fraction: float = 2.0 / 3.0,
) -> NonlinearFun:
    """Gradient-norm term ``N(u) = -scale/2 * (du/dx)^2`` (dealiased).

    The Kuramoto-Sivashinsky nonlinearity. The squared gradient is formed in
    physical space with two-thirds dealiasing; the constant (zero) Fourier mode of
    the product is removed so the term injects no net mass.
    """
    derivative = first_derivative_operator(num_points, domain_extent)
    mask = dealias_mask(num_points, dealias_fraction)

    def nonlinear_fun(u_hat: jax.Array) -> jax.Array:
        u_x = jnp.fft.irfft(mask * derivative * u_hat, n=num_points, axis=-1)
        squared_hat = mask * jnp.fft.rfft(0.5 * u_x**2, axis=-1)
        return -scale * squared_hat.at[..., 0].set(0.0)

    return nonlinear_fun


__all__ = [
    "NonlinearFun",
    "convection_nonlinearity",
    "dealias_mask",
    "first_derivative_operator",
    "gradient_norm_nonlinearity",
    "laplace_operator",
    "rfft_wavenumbers",
    "third_derivative_operator",
]
