"""Tests for the Becke-partitioned molecular quadrature grid.

The grid quality is checked by integrating the converged electron density and
recovering the electron count, which is the standard quadrature sanity check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft.grid import build_molecular_grid


def _h2_system() -> MolecularSystem:
    bond = 0.74 / 0.52917721067
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
        basis_set="sto-3g",
    )


def test_grid_weights_are_positive() -> None:
    """All Becke quadrature weights are non-negative."""
    with jax.enable_x64(True):
        grid = build_molecular_grid(_h2_system())
    assert np.all(np.asarray(grid.weights) >= 0.0)


def test_grid_integrates_gaussian_density() -> None:
    r"""Integrating a normalised Gaussian density recovers unit charge.

    Uses two unit-normalised s Gaussians (one per H centre); the closed-form
    integral of each is one, so the grid must integrate the summed density to
    two within quadrature error.
    """
    with jax.enable_x64(True):
        system = _h2_system()
        grid = build_molecular_grid(system)
        centers = system.positions
        exponent = 1.2
        norm = (2.0 * exponent / jnp.pi) ** 0.75
        points = grid.points
        density = jnp.zeros(points.shape[0])
        for center in centers:
            r2 = jnp.sum((points - center[None, :]) ** 2, axis=-1)
            density = density + (norm * jnp.exp(-exponent * r2)) ** 2
        integral = float(jnp.sum(grid.weights * density))
    assert integral == pytest.approx(2.0, abs=2e-3)


def test_grid_integrates_scf_density_to_electron_count() -> None:
    """Integrating the converged H2 density recovers two electrons."""
    pytest.importorskip("scipy")
    from opifex.neural.quantum.dft.scf import SCFSolver

    with jax.enable_x64(True):
        system = _h2_system()
        result = SCFSolver(system).solve()
        basis = AtomicOrbitalBasis.from_molecular_system(system)
        grid = build_molecular_grid(system)
        ao = basis.evaluate(grid.points)
        rho = jnp.einsum("gm,mn,gn->g", ao, result.density_matrix, ao)
        electrons = float(jnp.sum(grid.weights * rho))
    assert electrons == pytest.approx(2.0, abs=1e-3)
