"""Tests for the def2-SVP basis (d-shells) and its spherical integrals.

AO counts and overlap/kinetic integrals are validated against PySCF's
``gto.M(..., basis='def2-svp')`` in both the Cartesian and spherical
representations, exercising the new ``l = 2`` Cartesian components in
:mod:`opifex.core.quantum.basis` and the Cartesian->spherical transform in
:mod:`opifex.core.quantum._spherical`. PySCF is a test-time oracle only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from opifex.core.quantum._flat_harness import one_electron_matrices
from opifex.core.quantum._spherical import build_block_transform
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import ANGSTROM_TO_BOHR, MolecularSystem


# Geometries in Angstrom (atomic number, x, y, z); converted to Bohr on build.
_WATER_ANGSTROM: tuple[tuple[int, float, float, float], ...] = (
    (8, 0.0, 0.0, 0.0),
    (1, 0.0, 0.0, 0.957),
    (1, 0.927, 0.0, -0.240),
)
_METHANE_ANGSTROM: tuple[tuple[int, float, float, float], ...] = (
    (6, 0.0, 0.0, 0.0),
    (1, 0.629, 0.629, 0.629),
    (1, -0.629, -0.629, 0.629),
    (1, -0.629, 0.629, -0.629),
    (1, 0.629, -0.629, -0.629),
)


def _system(geometry: tuple[tuple[int, float, float, float], ...]) -> MolecularSystem:
    """Build a :class:`MolecularSystem` (positions in Bohr) from an Angstrom geometry."""
    atomic_numbers = jnp.asarray([atom[0] for atom in geometry])
    positions = jnp.asarray([[atom[1], atom[2], atom[3]] for atom in geometry]) * ANGSTROM_TO_BOHR
    return MolecularSystem(
        atomic_numbers=atomic_numbers,
        positions=positions,
        basis_set="def2-svp",
    )


def _pyscf_atom_string(geometry: tuple[tuple[int, float, float, float], ...]) -> str:
    """PySCF ``atom`` string (Angstrom) for the same geometry."""
    symbols = {1: "H", 6: "C", 8: "O"}
    return "; ".join(f"{symbols[z]} {x} {y} {zc}" for z, x, y, zc in geometry)


def _angular_momenta(basis: AtomicOrbitalBasis) -> tuple[int, ...]:
    """Angular momentum of every shell in AO order."""
    return tuple(shell.angular_momentum for shell in basis.shells)


@pytest.mark.parametrize(
    ("geometry", "expected_cart", "expected_sph"),
    [(_WATER_ANGSTROM, 25, 24), (_METHANE_ANGSTROM, 35, 34)],
)
def test_ao_counts_match_pyscf(
    geometry: tuple[tuple[int, float, float, float], ...],
    expected_cart: int,
    expected_sph: int,
) -> None:
    """Cartesian and spherical def2-SVP AO counts match PySCF for H2O / CH4."""
    gto = pytest.importorskip("pyscf.gto")
    atom = _pyscf_atom_string(geometry)
    assert gto.M(atom=atom, basis="def2-svp", cart=True).nao_nr() == expected_cart
    assert gto.M(atom=atom, basis="def2-svp", cart=False).nao_nr() == expected_sph

    basis = AtomicOrbitalBasis.from_molecular_system(_system(geometry), "def2-svp")
    assert basis.n_atomic_orbitals == expected_cart  # opifex builds Cartesian AOs.

    transform = build_block_transform(_angular_momenta(basis))
    assert transform.shape == (expected_cart, expected_sph)


def test_def2svp_includes_d_shells() -> None:
    """Heavy atoms gain an ``l = 2`` polarisation shell; H stays s/p only."""
    basis = AtomicOrbitalBasis.from_molecular_system(_system(_WATER_ANGSTROM), "def2-svp")
    oxygen_shells = [shell.angular_momentum for shell in basis.shells if shell.atom_index == 0]
    assert oxygen_shells == [0, 0, 0, 1, 1, 2]  # 3s, 2p, 1d
    hydrogen_shells = [shell.angular_momentum for shell in basis.shells if shell.atom_index == 1]
    assert hydrogen_shells == [0, 0, 1]  # 2s, 1p -- no d-shell on H.


def _spherical_one_electron(
    geometry: tuple[tuple[int, float, float, float], ...],
) -> tuple[Array, Array]:
    """Return opifex spherical ``(overlap, kinetic)`` for ``geometry`` (def2-SVP)."""
    system = _system(geometry)
    basis = AtomicOrbitalBasis.from_molecular_system(system, "def2-svp")
    flat = basis.flat_primitives()
    overlap_cart, kinetic_cart, _ = one_electron_matrices(
        flat, system.positions, system.atomic_numbers.astype(jnp.float64)
    )
    transform = build_block_transform(_angular_momenta(basis))
    overlap_sph = transform.T @ overlap_cart @ transform
    kinetic_sph = transform.T @ kinetic_cart @ transform
    return overlap_sph, kinetic_sph


@pytest.mark.parametrize("geometry", [_WATER_ANGSTROM, _METHANE_ANGSTROM])
def test_spherical_overlap_matches_pyscf(
    geometry: tuple[tuple[int, float, float, float], ...],
) -> None:
    """Spherical def2-SVP overlap (with d-shells) matches PySCF to ~1e-8."""
    gto = pytest.importorskip("pyscf.gto")
    molecule = gto.M(atom=_pyscf_atom_string(geometry), basis="def2-svp", cart=False)
    reference = np.asarray(molecule.intor("int1e_ovlp"))
    with jax.enable_x64(True):
        overlap_sph, _ = _spherical_one_electron(geometry)
    np.testing.assert_allclose(np.asarray(overlap_sph), reference, atol=1e-8)


@pytest.mark.parametrize("geometry", [_WATER_ANGSTROM, _METHANE_ANGSTROM])
def test_spherical_kinetic_matches_pyscf(
    geometry: tuple[tuple[int, float, float, float], ...],
) -> None:
    """Spherical def2-SVP kinetic energy (with d-shells) matches PySCF to ~1e-8."""
    gto = pytest.importorskip("pyscf.gto")
    molecule = gto.M(atom=_pyscf_atom_string(geometry), basis="def2-svp", cart=False)
    reference = np.asarray(molecule.intor("int1e_kin"))
    with jax.enable_x64(True):
        _, kinetic_sph = _spherical_one_electron(geometry)
    np.testing.assert_allclose(np.asarray(kinetic_sph), reference, atol=1e-8)


def test_d_shell_overlap_block_is_identity() -> None:
    """A single isolated d-shell has unit spherical self-overlap (5x5 identity)."""
    gto = pytest.importorskip("pyscf.gto")
    reference = np.asarray(gto.M(atom="O 0 0 0", basis="def2-svp", cart=False).intor("int1e_ovlp"))
    with jax.enable_x64(True):
        overlap_sph, _ = _spherical_one_electron(_WATER_ANGSTROM)
    # Oxygen's spherical d-block is the last 5 AOs of its 14-AO block (first atom).
    oxygen_d = np.asarray(overlap_sph)[9:14, 9:14]
    np.testing.assert_allclose(oxygen_d, reference[9:14, 9:14], atol=1e-8)
    np.testing.assert_allclose(np.diag(oxygen_d), np.ones(5), atol=1e-8)
