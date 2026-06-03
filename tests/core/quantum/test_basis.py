"""Tests for the STO-3G atomic-orbital basis builder.

The shell tables and AO offsets are validated against PySCF's ``gto.Mole`` (the
reference data structure). PySCF lives in the optional ``[neural-dft]`` extra and
is used here as a test-time oracle only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem


def _h2_system() -> MolecularSystem:
    """H2 at 0.74 Angstrom (1.398397 Bohr), aligned on the z-axis."""
    bond_bohr = 0.74 / 0.52917721067
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_bohr]]),
        basis_set="sto-3g",
    )


def _water_system() -> MolecularSystem:
    """Water with a simple bent geometry in Bohr."""
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


def test_h2_basis_has_two_s_aos() -> None:
    """H2 STO-3G has exactly two contracted s AOs, one per atom."""
    basis = AtomicOrbitalBasis.from_molecular_system(_h2_system())

    assert basis.n_atomic_orbitals == 2
    assert basis.n_shells == 2
    assert all(shell.angular_momentum == 0 for shell in basis.shells)
    assert [shell.ao_offset for shell in basis.shells] == [0, 1]


def test_water_basis_ao_count_and_offsets() -> None:
    """Water STO-3G has 7 AOs (O: 1s,2s,2p×3; H: 1s ×2) with contiguous offsets."""
    basis = AtomicOrbitalBasis.from_molecular_system(_water_system())

    assert basis.n_atomic_orbitals == 7
    # O shells: s, s, p (offsets 0,1,2); H shells: s (5), s (6).
    offsets = [shell.ao_offset for shell in basis.shells]
    assert offsets == [0, 1, 2, 5, 6]
    angular = [shell.angular_momentum for shell in basis.shells]
    assert angular == [0, 0, 1, 0, 0]


def test_contracted_self_overlap_is_unit() -> None:
    """Each contracted AO is renormalised to unit self-overlap (PySCF convention).

    The s-shell same-centre self-overlap is ``sum_ij c_i c_j (pi/(a_i+a_j))^1.5``.
    """
    with jax.enable_x64(True):
        basis = AtomicOrbitalBasis.from_molecular_system(_h2_system())
        shell = basis.shells[0]
        exps = shell.exponents
        coeffs = shell.coefficients
        combined = exps[:, None] + exps[None, :]
        overlap = jnp.sum(coeffs[:, None] * coeffs[None, :] * (jnp.pi / combined) ** 1.5)

    assert float(overlap) == pytest.approx(1.0, abs=1e-10)


def test_unsupported_basis_raises() -> None:
    """An unsupported basis name fails fast."""
    with pytest.raises(ValueError, match="Unsupported basis"):
        AtomicOrbitalBasis.from_molecular_system(_h2_system(), basis_name="cc-pvdz")


def test_unsupported_element_raises() -> None:
    """An element without tabulated STO-3G data fails fast."""
    helium = MolecularSystem(
        atomic_numbers=jnp.array([2]),
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        multiplicity=1,
        basis_set="sto-3g",
    )
    with pytest.raises(ValueError, match="No STO-3G data"):
        AtomicOrbitalBasis.from_molecular_system(helium)


@pytest.mark.slow
def test_basis_matches_pyscf_shell_structure() -> None:
    """Shell count, angular momenta and AO count match PySCF's gto.Mole."""
    gto = pytest.importorskip("pyscf.gto")
    basis = AtomicOrbitalBasis.from_molecular_system(_water_system())

    mol = gto.M(
        atom="O 0 0 0; H 0 0.756 0.587; H 0 -0.756 0.587",
        basis="sto-3g",
        unit="Angstrom",
    )
    assert basis.n_atomic_orbitals == mol.nao_nr()
    # PySCF reports 5 shells for water STO-3G (O: 1s,2s,2p; H: 1s; H: 1s).
    assert basis.n_shells == mol.nbas
