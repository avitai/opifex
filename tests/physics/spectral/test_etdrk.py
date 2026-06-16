"""Analytical tests for the general ETDRK4 integrator.

For a purely linear PDE (``N = 0``) ETDRK4 reduces to ``u_hat <- exp(dt L) u_hat``,
so it must reproduce the exact solution to machine precision. These tests pin that
exactness on diffusion (real eigenvalues) and advection (imaginary eigenvalues),
which together exercise the contour-integral coefficients away from and along the
imaginary axis. Written before the implementation was trusted: the ground truth is
the analytic propagator, not the code.
"""

import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.spectral.etdrk import integrate_etdrk4
from opifex.physics.spectral.semilinear import (
    first_derivative_operator,
    laplace_operator,
    rfft_wavenumbers,
)


def _zero_nonlinear(u_hat: jax.Array) -> jax.Array:
    return jnp.zeros_like(u_hat)


class TestETDRK4LinearExactness:
    """With no nonlinear term the integrator is the exact Fourier propagator."""

    def test_diffusion_decay_is_exact(self) -> None:
        """Heat equation: each mode decays as ``exp(-nu k^2 t)`` exactly."""
        num_points, domain, nu, t_final, steps = 64, 2.0 * np.pi, 0.05, 2.0, 40
        x = np.linspace(0.0, domain, num_points, endpoint=False)
        u0 = jnp.asarray(np.sin(x) + 0.5 * np.cos(2.0 * x))
        linear = nu * laplace_operator(num_points, domain)

        u_hat0 = jnp.fft.rfft(u0)
        trajectory = integrate_etdrk4(linear, _zero_nonlinear, u_hat0, t_final / steps, steps)
        u_final = jnp.fft.irfft(trajectory[-1], n=num_points)

        wavenumbers = np.asarray(rfft_wavenumbers(num_points, domain))
        exact = jnp.fft.irfft(u_hat0 * np.exp(-nu * wavenumbers**2 * t_final), n=num_points)
        assert float(jnp.max(jnp.abs(u_final - exact))) < 1e-11

    def test_advection_translation_is_exact(self) -> None:
        """Linear advection ``u_t = -c u_x`` translates the field exactly."""
        num_points, domain, speed, t_final, steps = 128, 2.0 * np.pi, 1.0, 1.0, 100
        x = np.linspace(0.0, domain, num_points, endpoint=False)
        u0 = jnp.asarray(np.exp(np.sin(x)))  # smooth, strictly periodic
        linear = -speed * first_derivative_operator(num_points, domain)

        u_hat0 = jnp.fft.rfft(u0)
        trajectory = integrate_etdrk4(linear, _zero_nonlinear, u_hat0, t_final / steps, steps)
        u_final = jnp.fft.irfft(trajectory[-1], n=num_points)

        wavenumbers = np.asarray(rfft_wavenumbers(num_points, domain))
        exact = jnp.fft.irfft(u_hat0 * np.exp(-1j * wavenumbers * speed * t_final), n=num_points)
        assert float(jnp.max(jnp.abs(u_final - exact))) < 1e-9

    def test_zero_mode_coefficients_are_finite(self) -> None:
        """The contour integral keeps coefficients finite where ``L = 0``.

        The zero Fourier mode has ``L = 0``; the naive ``(e^z - 1)/z`` forms are
        0/0 there. The Kassam-Trefethen contour must yield finite coefficients.
        """
        num_points, domain = 32, 2.0 * np.pi
        linear = laplace_operator(num_points, domain)  # entry 0 is exactly 0
        u_hat0 = jnp.fft.rfft(jnp.asarray(np.ones(num_points)))  # only zero mode
        trajectory = integrate_etdrk4(linear, _zero_nonlinear, u_hat0, 0.1, 5)
        assert bool(jnp.all(jnp.isfinite(trajectory)))
