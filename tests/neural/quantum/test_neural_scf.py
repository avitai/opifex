"""Tests for the molecular-system-dependent default SCF path of
:class:`NeuralSCFSolver`.

The default (``hamiltonian_fn is None``) branch of ``solve_scf`` must build a
real Kohn-Sham matrix from the supplied :class:`MolecularSystem` (nuclear
charges + geometry), diagonalise it and derive both the band-structure energy
and the updated density from the occupied orbitals. The output therefore has to
genuinely depend on ``molecular_system`` -- two chemically distinct systems with
the same electron count (H2 vs He, both two electrons) must yield different SCF
results rather than the historical synthetic ``0.5 * sum(rho**2)`` energy and
``0.01 * sin(...)`` density update, neither of which referenced the molecule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.neural_scf import NeuralSCFSolver, SCFResult


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Provide FLAX NNX rngs for component initialization."""
    return nnx.Rngs(jax.random.PRNGKey(0))


@pytest.fixture
def h2_system() -> MolecularSystem:
    """Two-atom, two-electron hydrogen molecule in Bohr (Z = 1, 1)."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    atomic_numbers = jnp.array([1, 1])
    return MolecularSystem(positions=positions, atomic_numbers=atomic_numbers)


@pytest.fixture
def he_system() -> MolecularSystem:
    """Single-atom, two-electron helium atom in Bohr (Z = 2)."""
    positions = jnp.array([[0.0, 0.0, 0.0]])
    atomic_numbers = jnp.array([2])
    return MolecularSystem(positions=positions, atomic_numbers=atomic_numbers)


def _solver(rngs: nnx.Rngs, grid_size: int) -> NeuralSCFSolver:
    """Build a small linear-mixing solver for deterministic SCF tests."""
    return NeuralSCFSolver(
        max_iterations=5,
        mixing_strategy="linear",
        grid_size=grid_size,
        rngs=rngs,
    )


def test_solve_scf_depends_on_molecular_system(
    rngs: nnx.Rngs, h2_system: MolecularSystem, he_system: MolecularSystem
) -> None:
    """The default SCF path must consume ``molecular_system``.

    H2 (two protons, two electrons) and He (one Z = 2 nucleus, two electrons)
    share an electron count but differ in nuclear charge and geometry, so a
    Kohn-Sham SCF that actually builds its Hamiltonian from the molecule must
    return different total energies and different converged densities. The
    historical synthetic update ignored ``molecular_system`` entirely and would
    return identical results for both.
    """
    grid_size = 16
    initial_density = jnp.ones(grid_size)

    h2_result = _solver(rngs, grid_size).solve_scf(h2_system, initial_density)
    he_result = _solver(rngs, grid_size).solve_scf(he_system, initial_density)

    assert isinstance(h2_result, SCFResult)
    assert isinstance(he_result, SCFResult)

    # Real, finite physics (no NaNs leaking from the eigensolve).
    assert jnp.isfinite(h2_result.total_energy)
    assert jnp.isfinite(he_result.total_energy)
    assert bool(jnp.all(jnp.isfinite(h2_result.final_density)))
    assert bool(jnp.all(jnp.isfinite(he_result.final_density)))

    # The molecule actually changed the answer.
    assert not jnp.isclose(h2_result.total_energy, he_result.total_energy)
    assert not bool(jnp.allclose(h2_result.final_density, he_result.final_density))


def test_solve_scf_custom_hamiltonian_overrides_default(
    rngs: nnx.Rngs, h2_system: MolecularSystem
) -> None:
    """A supplied ``hamiltonian_fn`` still drives the scalar-energy path.

    ``NeuralDFT.compute_energy`` feeds the solver a scalar callable (the trace
    of its Kohn-Sham matrix); the default-Hamiltonian rewrite must not break
    that contract. With a constant callable the very first energy error is the
    distance from the ``+inf`` seed, so the loop runs at least one iteration and
    reports the callable's value.
    """
    grid_size = 16
    initial_density = jnp.ones(grid_size)
    constant_energy = -1.5

    result = _solver(rngs, grid_size).solve_scf(
        h2_system,
        initial_density,
        hamiltonian_fn=lambda _density: jnp.asarray(constant_energy),
    )

    assert isinstance(result, SCFResult)
    assert jnp.isclose(result.total_energy, constant_energy)
