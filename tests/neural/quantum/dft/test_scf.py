"""Tests for the restricted Kohn-Sham (RKS) LDA SCF solver.

The headline validation is that the native total energy of H2 at STO-3G with the
LDA (Slater + VWN5) functional matches ``pyscf.dft.RKS`` with ``xc='lda,vwn'`` to
better than 1e-4 Hartree. PySCF lives in the optional ``[neural-dft]`` extra and
is used here as a test-time oracle only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft.scf import SCFResult, SCFSolver


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_system() -> MolecularSystem:
    """H2 at 0.74 Angstrom on the z-axis."""
    bond = 0.74 * _BOHR_PER_ANGSTROM
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
        basis_set="sto-3g",
    )


def test_scf_converges_h2() -> None:
    """The H2 RKS/LDA SCF reaches the convergence tolerance."""
    with jax.enable_x64(True):
        result = SCFSolver(_h2_system()).solve()
    assert isinstance(result, SCFResult)
    assert result.converged
    assert result.n_iterations < 100


def test_scf_density_is_idempotent_in_overlap_metric() -> None:
    """The closed-shell density satisfies ``D S D = 2 D`` (idempotency)."""
    from opifex.core.quantum.backend import JaxGaussianBackend
    from opifex.core.quantum.basis import AtomicOrbitalBasis

    with jax.enable_x64(True):
        system = _h2_system()
        result = SCFSolver(system).solve()
        basis = AtomicOrbitalBasis.from_molecular_system(system)
        overlap = JaxGaussianBackend(system, basis).overlap()
        density = result.density_matrix
        residual = float(jnp.max(jnp.abs(density @ overlap @ density - 2.0 * density)))
    assert residual < 1e-6


def test_open_shell_rejected() -> None:
    """A multiplicity != 1 system is rejected (closed-shell only)."""
    triplet = MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        multiplicity=3,
        basis_set="sto-3g",
    )
    with pytest.raises(ValueError, match="closed-shell"):
        SCFSolver(triplet)


@pytest.mark.slow
def test_h2_lda_energy_matches_pyscf_rks() -> None:
    """Headline: native H2 LDA total energy matches PySCF RKS to <= 1e-4 Ha."""
    dft = pytest.importorskip("pyscf.dft")
    gto = pytest.importorskip("pyscf.gto")

    with jax.enable_x64(True):
        native_energy = float(SCFSolver(_h2_system()).solve().total_energy)

    bond = 0.74 * _BOHR_PER_ANGSTROM
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {bond:.12f}", basis="sto-3g", unit="Bohr")
    mean_field = dft.RKS(mol)
    mean_field.xc = "lda,vwn"
    mean_field.grids.level = 3
    reference_energy = float(mean_field.kernel())

    assert native_energy == pytest.approx(reference_energy, abs=1e-4)


@pytest.mark.slow
def test_single_fock_build_is_jittable() -> None:
    """A single Fock build compiles under ``jax.jit`` (jit smoke)."""
    from opifex.core.quantum.backend import JaxGaussianBackend
    from opifex.core.quantum.basis import AtomicOrbitalBasis

    with jax.enable_x64(True):
        system = _h2_system()
        basis = AtomicOrbitalBasis.from_molecular_system(system)
        solver = SCFSolver(system, basis)
        backend = JaxGaussianBackend(system, basis)
        core = backend.core_hamiltonian()
        eri = backend.electron_repulsion()
        overlap = backend.overlap()
        grid_data = solver._grid_data()
        n_ao = basis.n_atomic_orbitals
        density = jnp.eye(n_ao)

        def fock_only(matrix: jnp.ndarray) -> jnp.ndarray:
            return solver.build_fock(matrix, core, eri, grid_data)[0]

        eager = fock_only(density)
        jitted = jax.jit(fock_only)(density)
        max_diff = float(jnp.max(jnp.abs(eager - jitted)))
        # The Fock matrix must be symmetric.
        symmetry = float(jnp.max(jnp.abs(eager - eager.T)))
        _ = overlap
    assert max_diff < 1e-10
    assert symmetry < 1e-10
