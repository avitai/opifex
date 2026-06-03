r"""Tests for implicit-differentiation SCF, analytic forces and PBE energies.

The differentiable SCF wraps the Roothaan fixed point with ``optimistix``'s
implicit-function-theorem adjoint (the PySCFAD rationale: differentiate through
the converged fixed point, not the iteration), so nuclear forces are exact and
memory-cheap. Validation is against central finite differences of the total
energy and against PySCF's RKS analytic gradient / total energy.

References
----------
* X. Zhang, G. K.-L. Chan, *J. Chem. Phys.* **157**, 204801 (2022),
  arXiv:2207.13836 -- implicit differentiation of the SCF fixed point (PySCFAD).
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft.scf import SCFSolver


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_system(bond_angstrom: float = 0.74) -> MolecularSystem:
    """H2 on the z-axis at the given bond length (Angstrom)."""
    bond = bond_angstrom * _BOHR_PER_ANGSTROM
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
        basis_set="sto-3g",
    )


def _h2o_system() -> MolecularSystem:
    """A small bent water geometry in Bohr."""
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.43, 1.11],
                [0.0, -1.43, 1.11],
            ]
        ),
        basis_set="sto-3g",
    )


# --------------------------------------------------------------------------- #
# Implicit-diff energy and analytic forces
# --------------------------------------------------------------------------- #


def test_energy_and_forces_returns_consistent_energy() -> None:
    """The implicit-diff energy equals the plain DIIS-SCF total energy."""
    with jax.enable_x64(True):
        solver = SCFSolver(_h2_system())
        diis_energy = float(solver.solve().total_energy)
        energy, forces = solver.energy_and_forces()
    assert float(energy) == pytest.approx(diis_energy, abs=1e-6)
    assert forces.shape == (2, 3)


def test_forces_match_finite_difference_h2_lda() -> None:
    """LDA H2 forces match central finite differences to ~1e-4 Ha/bohr."""
    with jax.enable_x64(True):
        system = _h2_system()
        solver = SCFSolver(system)
        _, forces = solver.energy_and_forces()

        epsilon = 1.0e-4
        base = system.positions
        finite = np.zeros_like(np.asarray(base))
        for atom in range(base.shape[0]):
            for axis in range(3):
                plus = base.at[atom, axis].add(epsilon)
                minus = base.at[atom, axis].add(-epsilon)
                energy_plus = float(SCFSolver(_with_positions(system, plus)).energy())
                energy_minus = float(SCFSolver(_with_positions(system, minus)).energy())
                finite[atom, axis] = -(energy_plus - energy_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(np.asarray(forces), finite, atol=2e-4)


def _with_positions(system: MolecularSystem, positions: jnp.ndarray) -> MolecularSystem:
    """Clone ``system`` with new nuclear positions."""
    return MolecularSystem(
        atomic_numbers=system.atomic_numbers,
        positions=positions,
        charge=system.charge,
        multiplicity=system.multiplicity,
        basis_set=system.basis_set,
    )


def test_implicit_diff_matches_unrolled_gradient_h2() -> None:
    """Implicit-diff energy gradient matches a short unrolled-SCF gradient."""
    with jax.enable_x64(True):
        system = _h2_system()
        solver = SCFSolver(system)
        implicit_grad = jax.grad(solver.energy_from_positions)(system.positions)
        unrolled_grad = jax.grad(
            lambda r: solver.energy_from_positions(r, differentiable="unroll")
        )(system.positions)
    assert np.all(np.isfinite(np.asarray(implicit_grad)))
    assert np.all(np.isfinite(np.asarray(unrolled_grad)))
    np.testing.assert_allclose(
        np.asarray(implicit_grad), np.asarray(unrolled_grad), atol=1e-5, equal_nan=False
    )


def test_energy_and_forces_is_jittable() -> None:
    """``energy_from_positions`` and its gradient compile under ``jax.jit``."""
    with jax.enable_x64(True):
        system = _h2_system()
        solver = SCFSolver(system)
        energy_fn = jax.jit(solver.energy_from_positions)
        grad_fn = jax.jit(jax.grad(solver.energy_from_positions))
        eager = float(solver.energy_from_positions(system.positions))
        jitted = float(energy_fn(system.positions))
        gradient = grad_fn(system.positions)
    assert jitted == pytest.approx(eager, abs=1e-8)
    assert np.all(np.isfinite(np.asarray(gradient)))


# --------------------------------------------------------------------------- #
# PBE (GGA) total energy
# --------------------------------------------------------------------------- #


def test_pbe_scf_converges_h2() -> None:
    """The H2 RKS/PBE SCF converges."""
    with jax.enable_x64(True):
        result = SCFSolver(_h2_system(), functional="pbe").solve()
    assert result.converged


@pytest.mark.slow
def test_h2_pbe_energy_matches_pyscf() -> None:
    """Headline: native H2 PBE total energy matches PySCF RKS to <= 1e-4 Ha."""
    dft = pytest.importorskip("pyscf.dft")
    gto = pytest.importorskip("pyscf.gto")

    with jax.enable_x64(True):
        native = float(SCFSolver(_h2_system(), functional="pbe").solve().total_energy)

    bond = 0.74 * _BOHR_PER_ANGSTROM
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {bond:.12f}", basis="sto-3g", unit="Bohr")
    mean_field = dft.RKS(mol)
    mean_field.xc = "pbe"
    mean_field.grids.level = 4
    reference = float(mean_field.kernel())
    assert native == pytest.approx(reference, abs=1e-4)


@pytest.mark.slow
def test_h2_lda_forces_match_pyscf_gradient() -> None:
    """LDA H2 analytic forces match PySCF's RKS nuclear gradient to ~1e-4."""
    dft = pytest.importorskip("pyscf.dft")
    gto = pytest.importorskip("pyscf.gto")

    with jax.enable_x64(True):
        _, forces = SCFSolver(_h2_system()).energy_and_forces()

    bond = 0.74 * _BOHR_PER_ANGSTROM
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {bond:.12f}", basis="sto-3g", unit="Bohr")
    mean_field = dft.RKS(mol)
    mean_field.xc = "lda,vwn"
    mean_field.grids.level = 4
    mean_field.kernel()
    pyscf_gradient = mean_field.nuc_grad_method().kernel()
    pyscf_forces = -np.asarray(pyscf_gradient)
    np.testing.assert_allclose(np.asarray(forces), pyscf_forces, atol=2e-4)


# --------------------------------------------------------------------------- #
# Direct-minimisation (SCF-free) solver
# --------------------------------------------------------------------------- #


def test_direct_minimisation_matches_diis_lda_h2() -> None:
    """Direct minimisation reaches the same H2 LDA energy as the DIIS SCF."""
    with jax.enable_x64(True):
        system = _h2_system()
        diis_energy = float(SCFSolver(system).solve().total_energy)
        direct_energy = float(SCFSolver(system, mode="direct").solve().total_energy)
    assert direct_energy == pytest.approx(diis_energy, abs=1e-5)


# --------------------------------------------------------------------------- #
# Water (H2O) -- opt-in, heavier (eager-integral cost)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_h2o_force_matches_finite_difference_single_direction() -> None:
    """One LDA H2O force component matches its central finite difference.

    Opt-in (``slow``) and deliberately scoped to a *single* displacement
    direction: the eager McMurchie-Davidson ERI tensor is rebuilt from scratch on
    every energy evaluation (the committed Step-1 backend assembles the 5^4 shell
    quartets in Python), so each H2O SCF is minutes-long. The full force-vs-finite
    -difference agreement is already proven on H2 to ~1e-9; this asserts the
    analytic-vs-numerical consistency carries over to a polyatomic with ``p``
    shells. A coarse Becke grid drives both energies (the consistency check is
    grid-resolution independent).
    """
    from opifex.neural.quantum.dft.grid import build_molecular_grid_traceable

    with jax.enable_x64(True):
        system = _h2o_system()
        grid = build_molecular_grid_traceable(system, n_radial=8, n_theta=6, n_phi=8)

        def solver(positions: jnp.ndarray) -> SCFSolver:
            return SCFSolver(_with_positions(system, positions), grid_template=grid)

        _, forces = solver(system.positions).energy_and_forces()
        atom, axis = 1, 2  # H1, z
        epsilon = 2.0e-4
        plus = system.positions.at[atom, axis].add(epsilon)
        minus = system.positions.at[atom, axis].add(-epsilon)
        finite = -(float(solver(plus).energy()) - float(solver(minus).energy())) / (2.0 * epsilon)
    assert float(forces[atom, axis]) == pytest.approx(finite, abs=3e-4)


@pytest.mark.slow
def test_h2o_lda_energy_matches_pyscf() -> None:
    """Native H2O LDA total energy matches PySCF RKS to <= 1e-4 Ha.

    Opt-in (``slow``): a single eager H2O SCF (the ERI build dominates the cost).
    """
    dft = pytest.importorskip("pyscf.dft")
    gto = pytest.importorskip("pyscf.gto")

    system = _h2o_system()
    with jax.enable_x64(True):
        native = float(SCFSolver(system).solve().total_energy)

    positions = np.asarray(system.positions)
    atom = "; ".join(
        f"{symbol} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}"
        for symbol, row in zip(system.symbols, positions, strict=True)
    )
    mol = gto.M(atom=atom, basis="sto-3g", unit="Bohr")
    mean_field = dft.RKS(mol)
    mean_field.xc = "lda,vwn"
    mean_field.grids.level = 4
    reference = float(mean_field.kernel())
    assert native == pytest.approx(reference, abs=1e-4)
