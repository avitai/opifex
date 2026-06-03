"""Tests for the native McMurchie-Davidson Gaussian-integral backend.

The native ``S``, ``T``, ``V``, ``ERI`` and nuclear-repulsion quantities are
validated against PySCF (the optional ``[neural-dft]`` extra) as an oracle. The
Boys function is checked against its closed-form / series limits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum.backend import boys_function, JaxGaussianBackend, QCBackend
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_bohr() -> float:
    """H2 bond length in Bohr (0.74 Angstrom)."""
    return 0.74 * _BOHR_PER_ANGSTROM


def _h2_system() -> MolecularSystem:
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, _h2_bohr()]]),
        basis_set="sto-3g",
    )


def _water_system() -> MolecularSystem:
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
        basis_set="sto-3g",
    )


def _backend(system: MolecularSystem) -> JaxGaussianBackend:
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    return JaxGaussianBackend(system, basis)


def test_backend_satisfies_protocol() -> None:
    """The native backend implements the :class:`QCBackend` protocol."""
    assert isinstance(_backend(_h2_system()), QCBackend)


def test_boys_zero_argument_limit() -> None:
    """``F_n(0) = 1/(2n+1)`` (the removable-singularity limit)."""
    with jax.enable_x64(True):
        for order in range(5):
            value = float(boys_function(order, jnp.asarray(0.0)))
            assert value == pytest.approx(1.0 / (2 * order + 1), abs=1e-12)


def test_boys_zeroth_order_matches_erf() -> None:
    r"""``F_0(x) = (1/2)\sqrt{\pi/x}\,\mathrm{erf}(\sqrt{x})``."""
    with jax.enable_x64(True):
        x = jnp.asarray(2.5)
        expected = 0.5 * jnp.sqrt(jnp.pi / x) * jax.scipy.special.erf(jnp.sqrt(x))
        assert float(boys_function(0, x)) == pytest.approx(float(expected), abs=1e-12)


def test_nuclear_repulsion_h2() -> None:
    """Nuclear repulsion of H2 is ``1/R`` in atomic units."""
    with jax.enable_x64(True):
        energy = float(_backend(_h2_system()).nuclear_repulsion())
    assert energy == pytest.approx(1.0 / _h2_bohr(), abs=1e-10)


def test_overlap_diagonal_is_unit() -> None:
    """Each AO is normalised, so the overlap diagonal is one."""
    with jax.enable_x64(True):
        overlap = _backend(_h2_system()).overlap()
    np.testing.assert_allclose(np.diag(np.asarray(overlap)), 1.0, atol=1e-10)


def test_core_hamiltonian_is_kinetic_plus_nuclear() -> None:
    """``h_core`` equals ``T + V``."""
    with jax.enable_x64(True):
        backend = _backend(_h2_system())
        core = np.asarray(backend.core_hamiltonian())
        kinetic = np.asarray(backend.kinetic())
        nuclear = np.asarray(backend.nuclear_attraction())
    np.testing.assert_allclose(core, kinetic + nuclear, atol=1e-12)


@pytest.mark.slow
def test_integrals_match_pyscf_h2() -> None:
    """H2 STO-3G S, T, V, ERI and E_nn match PySCF to ~1e-8."""
    gto = pytest.importorskip("pyscf.gto")
    with jax.enable_x64(True):
        backend = _backend(_h2_system())
        overlap = np.asarray(backend.overlap())
        kinetic = np.asarray(backend.kinetic())
        nuclear = np.asarray(backend.nuclear_attraction())
        eri = np.asarray(backend.electron_repulsion())
        e_nn = float(backend.nuclear_repulsion())

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {_h2_bohr():.12f}",
        basis="sto-3g",
        unit="Bohr",
    )
    assert np.max(np.abs(overlap - mol.intor("int1e_ovlp"))) < 1e-8
    assert np.max(np.abs(kinetic - mol.intor("int1e_kin"))) < 1e-8
    assert np.max(np.abs(nuclear - mol.intor("int1e_nuc"))) < 1e-8
    assert np.max(np.abs(eri - mol.intor("int2e"))) < 1e-8
    assert e_nn == pytest.approx(float(mol.energy_nuc()), abs=1e-8)


@pytest.mark.slow
def test_one_electron_integrals_match_pyscf_water() -> None:
    """Water STO-3G S, T, V match PySCF (cart=True) to ~1e-8.

    Exercises the p-shell angular-momentum recurrences and the Cartesian AO
    ordering, which must agree with PySCF's Cartesian convention.
    """
    gto = pytest.importorskip("pyscf.gto")
    with jax.enable_x64(True):
        backend = _backend(_water_system())
        overlap = np.asarray(backend.overlap())
        kinetic = np.asarray(backend.kinetic())
        nuclear = np.asarray(backend.nuclear_attraction())

    mol = gto.M(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        basis="sto-3g",
        unit="Bohr",
        cart=True,
    )
    assert np.max(np.abs(overlap - mol.intor("int1e_ovlp"))) < 1e-8
    assert np.max(np.abs(kinetic - mol.intor("int1e_kin"))) < 1e-8
    assert np.max(np.abs(nuclear - mol.intor("int1e_nuc"))) < 1e-8


@pytest.mark.slow
def test_eri_matches_pyscf_water() -> None:
    """Water STO-3G ERI matches PySCF (cart=True) to ~1e-8.

    Validates the p-shell two-electron quartets. The eager build is heavy (the
    native engine evaluates every primitive quartet without screening), so this
    is a deliberately slow, opt-in check; the H2 ERI test already exercises the
    full ERI code path quickly.
    """
    gto = pytest.importorskip("pyscf.gto")
    with jax.enable_x64(True):
        eri = np.asarray(_backend(_water_system()).electron_repulsion())

    mol = gto.M(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        basis="sto-3g",
        unit="Bohr",
        cart=True,
    )
    assert np.max(np.abs(eri - mol.intor("int2e"))) < 1e-8


@pytest.mark.slow
def test_overlap_builder_is_jittable() -> None:
    """The overlap builder compiles under ``jax.jit``."""
    with jax.enable_x64(True):
        backend = _backend(_h2_system())
        eager = np.asarray(backend.overlap())
        jitted = np.asarray(jax.jit(backend.overlap)())
    np.testing.assert_allclose(eager, jitted, atol=1e-12)
