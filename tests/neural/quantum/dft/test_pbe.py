r"""Tests for the PBE (GGA) exchange-correlation functional.

The PBE exchange enhancement factor :math:`F_x(s)` and the gradient-correction
:math:`H(r_s, t)` of the correlation are validated against PySCF's ``libxc``
reference (``gga_x_pbe`` / ``gga_c_pbe``); PySCF lives in the optional
``[neural-dft]`` extra and is a test-time oracle only.

References
----------
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  -- the exchange factor (eq. 13-14) and the correlation ``H`` (eq. 7-8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.neural.quantum.dft.xc import (
    pbe_correlation_energy_density,
    pbe_energy_density,
    pbe_exchange_correlation_potential,
    pbe_exchange_energy_density,
)


# Test (density, |grad rho|^2) pairs spanning the typical molecular range.
_TEST_RHO = (0.05, 0.1, 0.5, 1.0, 2.0)
_TEST_SIGMA = (0.005, 0.01, 0.05, 0.2, 0.5)


def _libxc_gga_input(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Build the libxc GGA density array ``[rho, gx, gy, gz]`` with sigma on x."""
    gx = np.sqrt(sigma)
    zeros = np.zeros_like(rho)
    return np.vstack([rho, gx, zeros, zeros])


def test_pbe_exchange_reduces_to_slater_at_zero_gradient() -> None:
    r"""At :math:`\sigma=0`, :math:`F_x(0)=1` so PBE exchange equals Slater."""
    with jax.enable_x64(True):
        rho = jnp.asarray(_TEST_RHO)
        sigma = jnp.zeros_like(rho)
        pbe = pbe_exchange_energy_density(rho, sigma)
        slater = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    np.testing.assert_allclose(np.asarray(pbe), np.asarray(slater), atol=1e-12)


def test_pbe_correlation_is_negative() -> None:
    """PBE correlation energy per particle is negative for all test densities."""
    with jax.enable_x64(True):
        correlation = pbe_correlation_energy_density(
            jnp.asarray(_TEST_RHO), jnp.asarray(_TEST_SIGMA)
        )
    assert np.all(np.asarray(correlation) < 0.0)


@pytest.mark.slow
def test_pbe_exchange_matches_libxc() -> None:
    """PBE exchange energy density matches libxc ``gga_x_pbe`` to < 1e-8."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_RHO)
    sigma = np.array(_TEST_SIGMA)
    reference = libxc.eval_xc("gga_x_pbe,", _libxc_gga_input(rho, sigma), spin=0)[0]
    with jax.enable_x64(True):
        native = np.asarray(pbe_exchange_energy_density(jnp.asarray(rho), jnp.asarray(sigma)))
    np.testing.assert_allclose(native, reference, atol=1e-8)


@pytest.mark.slow
def test_pbe_correlation_matches_libxc() -> None:
    """PBE correlation energy density matches libxc ``gga_c_pbe`` to < 1e-8."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_RHO)
    sigma = np.array(_TEST_SIGMA)
    reference = libxc.eval_xc(",gga_c_pbe", _libxc_gga_input(rho, sigma), spin=0)[0]
    with jax.enable_x64(True):
        native = np.asarray(pbe_correlation_energy_density(jnp.asarray(rho), jnp.asarray(sigma)))
    np.testing.assert_allclose(native, reference, atol=1e-8)


@pytest.mark.slow
def test_pbe_energy_density_matches_libxc() -> None:
    """Total PBE energy density matches libxc ``pbe`` to < 1e-8."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_RHO)
    sigma = np.array(_TEST_SIGMA)
    reference = libxc.eval_xc("pbe", _libxc_gga_input(rho, sigma), spin=0)[0]
    with jax.enable_x64(True):
        native = np.asarray(pbe_energy_density(jnp.asarray(rho), jnp.asarray(sigma)))
    np.testing.assert_allclose(native, reference, atol=1e-8)


@pytest.mark.slow
def test_pbe_potential_matches_libxc() -> None:
    r"""AD-derived PBE potential matches libxc ``(vrho, vsigma)`` to < 1e-6.

    The GGA XC potential has two components: :math:`\partial(\rho\varepsilon)/
    \partial\rho` and :math:`\partial(\rho\varepsilon)/\partial\sigma`.
    """
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_RHO)
    sigma = np.array(_TEST_SIGMA)
    vxc = libxc.eval_xc("pbe", _libxc_gga_input(rho, sigma), spin=0)[1]
    ref_vrho, ref_vsigma = vxc[0], vxc[1]
    with jax.enable_x64(True):
        native_vrho, native_vsigma = pbe_exchange_correlation_potential(
            jnp.asarray(rho), jnp.asarray(sigma)
        )
    np.testing.assert_allclose(np.asarray(native_vrho), ref_vrho, atol=1e-6)
    np.testing.assert_allclose(np.asarray(native_vsigma), ref_vsigma, atol=1e-6)
