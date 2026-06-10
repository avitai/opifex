"""SCF acceleration from a caller-provided initial guess.

A high-quality initial density (such as one reconstructed from a neural-network
predicted Fock) lets the Anderson/DIIS SCF reach the fixed point in fewer
iterations than the default core-Hamiltonian guess. These tests verify the new
``initial_density`` seam on the solver, the ``density_from_fock`` reconstruction
helper, and the ``measure_scf_acceleration`` reduction report -- all on water at
STO-3G (closed-shell, 7 AOs) so the suite stays fast and runs in float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft.scf import density_from_fock, SCFSolver
from opifex.neural.quantum.dft.scf_acceleration import (
    measure_scf_acceleration,
    SCFAccelerationResult,
)


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _water_system() -> MolecularSystem:
    """Water at its experimental geometry, STO-3G (10 electrons, 7 AOs)."""
    angstrom = jnp.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=angstrom * _BOHR_PER_ANGSTROM,
        basis_set="sto-3g",
    )


def _overlap(system: MolecularSystem) -> jax.Array:
    """The AO overlap matrix for ``system`` at STO-3G."""
    from opifex.core.quantum.backend import JaxGaussianBackend

    basis = AtomicOrbitalBasis.from_molecular_system(system)
    return JaxGaussianBackend(system, basis).overlap()


def test_solve_accepts_initial_density_and_keeps_fixed_point() -> None:
    """Seeding the converged density yields the same energy in fewer iterations."""
    with jax.enable_x64(True):
        solver = SCFSolver(_water_system())
        baseline = solver.solve()
        guided = solver.solve(initial_density=baseline.density_matrix)
    assert guided.converged
    assert float(guided.total_energy) == pytest.approx(float(baseline.total_energy), abs=1e-8)
    assert guided.n_iterations < baseline.n_iterations


def test_initial_guess_does_not_change_the_converged_energy() -> None:
    """An arbitrary (poor) seed converges to the same fixed-point energy."""
    with jax.enable_x64(True):
        system = _water_system()
        solver = SCFSolver(system)
        baseline = solver.solve()
        n_ao = baseline.density_matrix.shape[0]
        # A crude diagonal seed: 2 electrons on each of the n_occupied lowest AOs.
        n_occupied = int(jnp.sum(system.atomic_numbers)) // 2
        seed = jnp.diag(jnp.array([2.0] * n_occupied + [0.0] * (n_ao - n_occupied)))
        guided = solver.solve(initial_density=seed)
    assert guided.converged
    assert float(guided.total_energy) == pytest.approx(float(baseline.total_energy), abs=1e-6)


def test_density_from_fock_reconstructs_the_converged_density() -> None:
    """``density_from_fock`` on the converged Fock reproduces the SCF density."""
    with jax.enable_x64(True):
        system = _water_system()
        result = SCFSolver(system).solve()
        overlap = _overlap(system)
        n_occupied = int(jnp.sum(system.atomic_numbers)) // 2
        # Rebuild the Fock at the converged density and round-trip it to a density.
        from opifex.neural.quantum.dft._energy import build_fock
        from opifex.neural.quantum.dft.scf import Functional

        solver = SCFSolver(system)
        integrals = solver._integrals(system.positions)
        fock, _, _ = build_fock(result.density_matrix, integrals, Functional.LDA, None)
        reconstructed = density_from_fock(fock, overlap, n_occupied)
        residual = float(jnp.max(jnp.abs(reconstructed - result.density_matrix)))
    assert residual < 1e-7


def test_measure_scf_acceleration_reports_iteration_reduction() -> None:
    """The reduction report shows the guided solve taking strictly fewer steps."""
    with jax.enable_x64(True):
        solver = SCFSolver(_water_system())
        converged = solver.solve()
        report = measure_scf_acceleration(solver, converged.density_matrix)
    assert isinstance(report, SCFAccelerationResult)
    assert report.guided_iterations < report.baseline_iterations
    assert report.iteration_reduction == report.baseline_iterations - report.guided_iterations
    assert report.converged
    assert float(report.energy_hartree) == pytest.approx(float(converged.total_energy), abs=1e-8)
