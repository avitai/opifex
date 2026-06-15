"""Correctness tests for the real spherical harmonic transform (SHT).

These guards pin the JAX real-SHT (``opifex.neural.operators.fno._spherical_harmonics``)
against the orthonormalized real spherical harmonics used by NVIDIA ``torch-harmonics``
(``torch_harmonics/sht.py``, ``legendre.py``) and the analytic definition of the
spherical harmonics ``Y_l^m``.

References
----------
- Bonev et al. 2023, "Spherical Fourier Neural Operators" (arXiv:2306.03838).
- ``torch_harmonics/sht.py`` (``RealSHT`` / ``InverseRealSHT``).
- ``torch_harmonics/legendre.py`` (``legpoly`` / ``_precompute_legpoly``, ``clm``).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre

from opifex.neural.operators.fno._spherical_harmonics import SphericalHarmonicBasis


def _real_part_sh(degree: int, order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate the real part of the complex spherical harmonic ``Y_l^m`` on a grid.

    Uses the ``torch-harmonics`` orthonormal convention (``legendre.py``):
    ``Re(Y_l^m) = clm * P_l^m(cos theta) * cos(m phi)`` where ``clm`` matches
    ``torch_harmonics.legendre.clm`` and the Condon-Shortley phase is carried by
    the associated Legendre evaluation. This is the field whose ``RealSHT`` analysis
    concentrates on the ``(l, m)`` coefficient (with magnitude ``1`` for ``m == 0``
    and ``1/2`` for ``m > 0``, since the real FFT stores only non-negative orders).

    Args:
        degree: Spherical harmonic degree ``l``.
        order: Non-negative azimuthal order ``m``.
        theta: Colatitude grid of shape ``(nlat,)`` in radians.
        phi: Longitude grid of shape ``(nlon,)`` in radians.

    Returns:
        Array of shape ``(nlat, nlon)`` with the real spherical harmonic values.
    """
    from scipy.special import lpmv

    cos_theta = np.cos(theta)
    # clm matches torch_harmonics.legendre.clm (orthonormal SH normalization).
    clm = math.sqrt((2 * degree + 1) / (4 * math.pi)) * math.sqrt(
        math.factorial(degree - order) / math.factorial(degree + order)
    )
    # scipy lpmv already carries the Condon-Shortley phase (-1)^m.
    legendre = clm * lpmv(order, degree, cos_theta)  # (nlat,)
    azimuth = np.cos(order * phi)  # (nlon,)
    return legendre[:, None] * azimuth[None, :]  # (nlat, nlon)


@pytest.fixture
def gauss_grid() -> tuple[int, int, np.ndarray, np.ndarray]:
    """Provide a small Gauss-Legendre latitude / equiangular longitude grid."""
    nlat, nlon = 16, 32
    cost, _ = roots_legendre(nlat)
    # torch_harmonics flips arccos(cost) so latitudes ascend (sht.py:98).
    theta = np.flip(np.arccos(cost))
    phi = np.linspace(0.0, 2.0 * math.pi, nlon, endpoint=False)
    return nlat, nlon, theta, phi


class TestSphericalHarmonicBasis:
    """Pin the real SHT forward/inverse against analytic spherical harmonics."""

    def test_round_trip_band_limited_field(self, gauss_grid):
        """``inverse(forward(f)) approx f`` for a band-limited field on the sphere."""
        nlat, nlon, theta, phi = gauss_grid
        basis = SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=8)

        # Build a band-limited field from a few low-degree real harmonics.
        field = (
            0.7 * _real_part_sh(0, 0, theta, phi)
            + 1.3 * _real_part_sh(2, 1, theta, phi)
            - 0.5 * _real_part_sh(3, 2, theta, phi)
        )
        field_jax = jnp.asarray(field)

        coeffs = basis.forward(field_jax)
        reconstructed = basis.inverse(coeffs)

        max_abs_error = float(jnp.max(jnp.abs(reconstructed - field_jax)))
        assert max_abs_error < 1e-4, f"round-trip error too large: {max_abs_error}"

    def test_forward_of_single_harmonic_is_single_coefficient(self, gauss_grid):
        """``forward(Re Y_l^m)`` yields a single nonzero coefficient at ``(l, m)``.

        For the real-FFT SHT a ``cos(m phi)`` harmonic (``m > 0``) deposits half its
        energy at the stored ``+m`` order, so the complex coefficient magnitude is
        ``1/2`` (and ``1`` for ``m == 0``).
        """
        nlat, nlon, theta, phi = gauss_grid
        basis = SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=8)
        target_l, target_m = 3, 2

        field = jnp.asarray(_real_part_sh(target_l, target_m, theta, phi))
        coeffs = basis.forward(field)

        magnitudes = np.asarray(jnp.abs(coeffs))
        peak = magnitudes[target_l, target_m]
        assert peak == pytest.approx(0.5, abs=1e-3)

        masked = magnitudes.copy()
        masked[target_l, target_m] = 0.0
        assert float(masked.max()) < 1e-3

    def test_forward_of_zonal_harmonic_is_unit_coefficient(self, gauss_grid):
        """``forward(Y_l^0)`` yields a unit coefficient at ``(l, 0)``."""
        nlat, nlon, theta, phi = gauss_grid
        basis = SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=8)
        target_l = 4

        field = jnp.asarray(_real_part_sh(target_l, 0, theta, phi))
        coeffs = basis.forward(field)

        magnitudes = np.asarray(jnp.abs(coeffs))
        assert magnitudes[target_l, 0] == pytest.approx(1.0, abs=1e-3)
        masked = magnitudes.copy()
        masked[target_l, 0] = 0.0
        assert float(masked.max()) < 1e-3

    def test_synthesis_of_unit_coefficient_reproduces_harmonic(self, gauss_grid):
        """Synthesizing a unit coefficient at ``(l, m)`` reproduces ``2 Re(Y_l^m)``.

        The inverse of the forward analysis above: a unit complex coefficient at
        ``(l, m > 0)`` synthesizes ``2 * Re(Y_l^m)`` (the factor-two inverse of the
        ``1/2`` analysis coefficient for non-negative-only stored orders).
        """
        nlat, nlon, theta, phi = gauss_grid
        basis = SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=8)
        target_l, target_m = 4, 1

        coeffs = jnp.zeros((basis.lmax, basis.mmax), dtype=jnp.complex64)
        coeffs = coeffs.at[target_l, target_m].set(1.0 + 0.0j)
        reconstructed = basis.inverse(coeffs)

        expected = 2.0 * _real_part_sh(target_l, target_m, theta, phi)
        max_abs_error = float(jnp.max(jnp.abs(reconstructed - jnp.asarray(expected))))
        assert max_abs_error < 1e-4, f"synthesis error too large: {max_abs_error}"

    def test_forward_is_jit_grad_vmap_compatible(self, gauss_grid):
        """The forward/inverse transforms compose with jit, grad and vmap."""
        nlat, nlon, theta, phi = gauss_grid
        basis = SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=8)

        def energy(field: jax.Array) -> jax.Array:
            return jnp.sum(jnp.abs(basis.forward(field)) ** 2)

        field = jnp.asarray(_real_part_sh(2, 1, theta, phi))

        jit_value = jax.jit(energy)(field)
        assert jnp.isfinite(jit_value)

        grad_value = jax.grad(energy)(field)
        assert grad_value.shape == field.shape
        assert jnp.all(jnp.isfinite(grad_value))

        batch = jnp.stack([field, 2.0 * field, -field], axis=0)
        round_trip = jax.vmap(lambda f: basis.inverse(basis.forward(f)))(batch)
        assert round_trip.shape == batch.shape
        assert jnp.all(jnp.isfinite(round_trip))
