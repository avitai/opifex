"""Tests for the LDA exchange-correlation functional.

Slater exchange, VWN5 correlation and the AD-derived XC potential are validated
against PySCF's ``libxc`` reference (the optional ``[neural-dft]`` extra).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.neural.quantum.dft.xc import (
    lda_exchange_correlation_potential,
    slater_exchange_energy_density,
    vwn_correlation_energy_density,
)


_TEST_DENSITIES = (0.001, 0.01, 0.1, 1.0)


def test_slater_exchange_scaling() -> None:
    r"""Slater exchange obeys :math:`\varepsilon_x = -C_x \rho^{1/3}`."""
    with jax.enable_x64(True):
        rho = jnp.asarray(_TEST_DENSITIES)
        exchange = slater_exchange_energy_density(rho)
        constant = 0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)
        expected = -constant * rho ** (1.0 / 3.0)
    np.testing.assert_allclose(np.asarray(exchange), np.asarray(expected), atol=1e-12)


def test_correlation_is_negative() -> None:
    """VWN5 correlation energy per particle is negative for all densities."""
    with jax.enable_x64(True):
        correlation = vwn_correlation_energy_density(jnp.asarray(_TEST_DENSITIES))
    assert np.all(np.asarray(correlation) < 0.0)


@pytest.mark.slow
def test_slater_matches_pyscf() -> None:
    """Slater exchange matches PySCF's ``lda,`` exchange-only energy density."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_DENSITIES)
    reference = libxc.eval_xc("lda,", rho, spin=0)[0]
    with jax.enable_x64(True):
        native = np.asarray(slater_exchange_energy_density(jnp.asarray(rho)))
    np.testing.assert_allclose(native, reference, atol=1e-8)


@pytest.mark.slow
def test_vwn_matches_pyscf() -> None:
    """VWN5 correlation matches PySCF's ``,vwn`` correlation energy density."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_DENSITIES)
    reference = libxc.eval_xc(",vwn", rho, spin=0)[0]
    with jax.enable_x64(True):
        native = np.asarray(vwn_correlation_energy_density(jnp.asarray(rho)))
    np.testing.assert_allclose(native, reference, atol=1e-8)


@pytest.mark.slow
def test_potential_matches_pyscf() -> None:
    """The AD-derived XC potential matches PySCF's ``lda,vwn`` ``vxc``."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array(_TEST_DENSITIES)
    reference = libxc.eval_xc("lda,vwn", rho, spin=0)[1][0]
    with jax.enable_x64(True):
        native = np.asarray(lda_exchange_correlation_potential(jnp.asarray(rho)))
    np.testing.assert_allclose(native, reference, atol=1e-8)
