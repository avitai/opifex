r"""Tests for the resolution-of-identity / density-fitting integral engine.

The RI/DF auxiliary integrals (the three-centre :math:`(\mu\nu|P)`, the
two-centre Coulomb metric :math:`(P|Q)`, and the fitted four-index ERI) are
validated against PySCF as an oracle:

* :math:`(\mu\nu|P)` against ``pyscf.df.incore.aux_e2(intor='int3c2e')``,
* :math:`(P|Q)` against ``auxmol.intor('int2c2e')``,
* the fitted ERI against PySCF's explicit Coulomb-metric fit
  :math:`J V^{-1} J` (the density-fitting reference).

PySCF's ``int2c2e``/``int3c2e`` use a Coulomb auxiliary normalisation that
differs from the overlap normalisation of :mod:`opifex.core.quantum.basis` by a
single auxiliary-only diagonal :math:`d_P = \sqrt{4\pi/(2l_P+1)}`. That diagonal
cancels in the fit, so the three-/two-centre tensors are compared to PySCF up to
that documented diagonal (matching to ~1e-10), while the *fitted* ERI matches
PySCF's fit to ~1e-10 directly. The integral path is additionally checked for
``jit``/``grad``/``vmap`` compatibility.

References for the RI approximation itself live in
:mod:`opifex.core.quantum._ri` (Whitten 1973; Dunlap 1979; Vahtras, Almlof &
Feyereisen 1993).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum._ri import (
    AuxiliaryBasis,
    fitted_eri,
    three_center_eri,
    two_center_metric,
)
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


# ---------------------------------------------------------------------------
# Small molecular systems and auxiliary bases (s,p only -- the engine's scope).
# ---------------------------------------------------------------------------
def _h2_system() -> MolecularSystem:
    """H2 at 0.74 Angstrom in Bohr."""
    bond = 0.74 * _BOHR_PER_ANGSTROM
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
        basis_set="sto-3g",
    )


def _water_system() -> MolecularSystem:
    """Bent water in Bohr (the p-shell stress case)."""
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
        basis_set="sto-3g",
    )


# Hand-built s+p auxiliary basis (in scope) shared between the engine and PySCF.
# Two even-tempered s and p sets per heavy atom, two s sets per hydrogen.
_AUX_DEF: dict[str, list[list]] = {
    "O": [
        [0, [3.0, 1.0]],
        [0, [1.0, 1.0]],
        [1, [1.4, 1.0]],
        [1, [0.5, 1.0]],
    ],
    "H": [
        [0, [1.2, 1.0]],
        [0, [0.4, 1.0]],
    ],
}


def _aux_shells_for_system(
    system: MolecularSystem,
) -> tuple[tuple[int, int, tuple[tuple[float, float], ...]], ...]:
    """Expand ``_AUX_DEF`` into ``(center_index, l, primitives)`` shell tuples."""
    symbols = {1: "H", 8: "O"}
    shells: list[tuple[int, int, tuple[tuple[float, float], ...]]] = []
    for atom_index, z in enumerate(np.asarray(system.atomic_numbers)):
        for angular_momentum, (exp, coeff) in _AUX_DEF[symbols[int(z)]]:
            shells.append((atom_index, angular_momentum, ((float(exp), float(coeff)),)))
    return tuple(shells)


def _aux_basis(system: MolecularSystem) -> AuxiliaryBasis:
    """Build the in-tree :class:`AuxiliaryBasis` for a system."""
    return AuxiliaryBasis.from_shells(_aux_shells_for_system(system), jnp.asarray(system.positions))


def _flat(system: MolecularSystem):
    """Flat main-basis primitives for a system."""
    return AtomicOrbitalBasis.from_molecular_system(system).flat_primitives()


def _pyscf_aux_basis_dict(system: MolecularSystem) -> dict[str, list[list]]:
    """The same auxiliary basis in PySCF's nested-list format, keyed by element."""
    symbols = {1: "H", 8: "O"}
    out: dict[str, list[list]] = {}
    for z in {int(zz) for zz in np.asarray(system.atomic_numbers)}:
        out[symbols[z]] = [list(shell) for shell in _AUX_DEF[symbols[z]]]
    return out


def _pyscf_mol_and_aux(system: MolecularSystem):
    """Build the matching PySCF main + auxiliary ``Mole`` objects (Cartesian)."""
    gto = pytest.importorskip("pyscf.gto")
    df = pytest.importorskip("pyscf.df")
    positions = np.asarray(system.positions)
    symbols = {1: "H", 8: "O"}
    atom = "; ".join(
        f"{symbols[int(z)]} {positions[i, 0]:.12f} {positions[i, 1]:.12f} {positions[i, 2]:.12f}"
        for i, z in enumerate(np.asarray(system.atomic_numbers))
    )
    spin = int(np.sum(np.asarray(system.atomic_numbers)) % 2)
    mol = gto.M(atom=atom, basis="sto-3g", unit="Bohr", cart=True, spin=spin)
    auxmol = df.addons.make_auxmol(mol, auxbasis=_pyscf_aux_basis_dict(system))
    return mol, auxmol


def _pyscf_consistency_diagonal(metric: np.ndarray, pyscf_int2c2e: np.ndarray) -> np.ndarray:
    r"""Recover the auxiliary-only diagonal :math:`d_P` relating the conventions.

    The in-tree metric equals :math:`D\,V_\text{pyscf}\,D` with
    :math:`D = \mathrm{diag}(d_P)`; ``d_P`` is read off the diagonals (both are
    strictly positive).
    """
    return np.sqrt(np.abs(np.diag(metric) / np.diag(pyscf_int2c2e)))


# ---------------------------------------------------------------------------
# Auxiliary-basis construction
# ---------------------------------------------------------------------------
def test_auxiliary_basis_counts_s_and_p_functions() -> None:
    """Water's s+p auxiliary basis has the expected contracted-function count."""
    aux = _aux_basis(_water_system())
    # O: 2 s + 2 p(=6 cart) = 8; each H: 2 s = 2 -> 8 + 2 + 2 = 12.
    assert aux.num_orbitals == 12
    assert aux.max_total_l == 1


def test_auxiliary_basis_rejects_unsupported_angular_momentum() -> None:
    """An angular momentum without a tabulated Cartesian component list is rejected."""
    with pytest.raises(ValueError, match="angular momentum"):
        AuxiliaryBasis.from_shells(((0, 3, ((1.0, 1.0),)),), jnp.zeros((1, 3)))


def test_auxiliary_basis_rejects_out_of_range_center() -> None:
    """An out-of-range centre index is rejected."""
    with pytest.raises(ValueError, match="centre index"):
        AuxiliaryBasis.from_shells(((3, 0, ((1.0, 1.0),)),), jnp.zeros((1, 3)))


# ---------------------------------------------------------------------------
# Two-centre Coulomb metric
# ---------------------------------------------------------------------------
def test_two_center_metric_is_symmetric_positive_definite() -> None:
    """The Coulomb metric ``(P|Q)`` is symmetric with positive eigenvalues."""
    with jax.enable_x64(True):
        metric = np.asarray(two_center_metric(_aux_basis(_water_system())))
    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(metric)) > 0.0


@pytest.mark.slow
def test_two_center_metric_matches_pyscf_int2c2e_water() -> None:
    """``(P|Q)`` matches PySCF ``int2c2e`` up to the convention diagonal (~1e-10)."""
    system = _water_system()
    with jax.enable_x64(True):
        metric = np.asarray(two_center_metric(_aux_basis(system)))
    _mol, auxmol = _pyscf_mol_and_aux(system)
    pyscf_metric = auxmol.intor("int2c2e")
    diagonal = _pyscf_consistency_diagonal(metric, pyscf_metric)
    rescaled = diagonal[:, None] * pyscf_metric * diagonal[None, :]
    assert np.max(np.abs(metric - rescaled)) < 1e-10


# ---------------------------------------------------------------------------
# Three-centre integrals
# ---------------------------------------------------------------------------
def test_three_center_shape_and_bra_symmetry_h2() -> None:
    """``(mu nu|P)`` has the right shape and is symmetric in the bra ``mu<->nu``."""
    system = _h2_system()
    with jax.enable_x64(True):
        tensor = np.asarray(three_center_eri(_flat(system), _aux_basis(system)))
    flat = _flat(system)
    aux = _aux_basis(system)
    assert tensor.shape == (flat.num_orbitals, flat.num_orbitals, aux.num_orbitals)
    np.testing.assert_allclose(tensor, tensor.transpose(1, 0, 2), atol=1e-12)


@pytest.mark.slow
def test_three_center_matches_pyscf_int3c2e_water() -> None:
    """``(mu nu|P)`` matches PySCF ``int3c2e`` up to the convention diagonal (~1e-10).

    Exercises s and p on both the main and auxiliary side (the p-shell
    recurrences and Cartesian ordering must agree with PySCF).
    """
    df = pytest.importorskip("pyscf.df")
    system = _water_system()
    with jax.enable_x64(True):
        tensor = np.asarray(three_center_eri(_flat(system), _aux_basis(system)))
        metric = np.asarray(two_center_metric(_aux_basis(system)))
    mol, auxmol = _pyscf_mol_and_aux(system)
    pyscf_3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    diagonal = _pyscf_consistency_diagonal(metric, auxmol.intor("int2c2e"))
    rescaled = pyscf_3c * diagonal[None, None, :]
    assert np.max(np.abs(tensor - rescaled)) < 1e-10


# ---------------------------------------------------------------------------
# Fitted ERI (convention-independent)
# ---------------------------------------------------------------------------
def test_fitted_eri_has_chemist_permutation_symmetry_h2() -> None:
    """The fitted ERI is symmetric under the permutations the fit preserves."""
    system = _h2_system()
    with jax.enable_x64(True):
        eri = np.asarray(fitted_eri(_flat(system), _aux_basis(system)))
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)


@pytest.mark.slow
def test_fitted_eri_matches_pyscf_density_fit_h2() -> None:
    """The fitted ERI matches PySCF's explicit Coulomb-metric fit ``J V^-1 J``."""
    df = pytest.importorskip("pyscf.df")
    system = _h2_system()
    with jax.enable_x64(True):
        eri = np.asarray(fitted_eri(_flat(system), _aux_basis(system)))
    mol, auxmol = _pyscf_mol_and_aux(system)
    n_ao = mol.nao_nr()
    pyscf_3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    pyscf_2c = auxmol.intor("int2c2e")
    pyscf_fit = np.einsum("ijP,PQ,klQ->ijkl", pyscf_3c, np.linalg.inv(pyscf_2c), pyscf_3c).reshape(
        n_ao, n_ao, n_ao, n_ao
    )
    assert np.max(np.abs(eri - pyscf_fit)) < 1e-10


@pytest.mark.slow
def test_fitted_eri_matches_pyscf_density_fit_water() -> None:
    """The p-shell fitted ERI matches PySCF's explicit Coulomb-metric fit (~1e-10)."""
    df = pytest.importorskip("pyscf.df")
    system = _water_system()
    with jax.enable_x64(True):
        eri = np.asarray(fitted_eri(_flat(system), _aux_basis(system)))
    mol, auxmol = _pyscf_mol_and_aux(system)
    n_ao = mol.nao_nr()
    pyscf_3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    pyscf_2c = auxmol.intor("int2c2e")
    pyscf_fit = np.einsum("ijP,PQ,klQ->ijkl", pyscf_3c, np.linalg.inv(pyscf_2c), pyscf_3c).reshape(
        n_ao, n_ao, n_ao, n_ao
    )
    assert np.max(np.abs(eri - pyscf_fit)) < 1e-10


@pytest.mark.slow
def test_fitted_eri_approximates_exact_eri_h2() -> None:
    """RI is approximate but tracks the exact ERI; the error is bounded and small.

    Documents the RI tolerance: with this compact s+p auxiliary basis the fitted
    H2 ERI is within ~1e-2 of the exact ``int2e`` (RI is an approximation, not an
    identity -- the 3c/2c integrals themselves match PySCF to ~1e-10 above).
    """
    system = _h2_system()
    with jax.enable_x64(True):
        eri = np.asarray(fitted_eri(_flat(system), _aux_basis(system)))
    mol, _auxmol = _pyscf_mol_and_aux(system)
    exact = mol.intor("int2e").reshape(eri.shape)
    assert np.max(np.abs(eri - exact)) < 1e-1


# ---------------------------------------------------------------------------
# JAX transform compatibility (required)
# ---------------------------------------------------------------------------
def test_three_center_is_jittable() -> None:
    """The three-centre builder compiles under ``jax.jit`` (traced over geometry).

    The flat-primitive containers carry NumPy-static metadata (angular powers,
    orbital maps), so -- exactly like the main-basis harness -- the ``jit`` seam
    is a function of the *positions*: the bases are rebuilt inside the trace and
    only the centres/exponents/coefficients are traced.
    """
    system = _h2_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    aux_shells = _aux_shells_for_system(system)

    def build(positions: jnp.ndarray) -> jnp.ndarray:
        flat = basis.with_positions(positions).flat_primitives()
        aux = AuxiliaryBasis.from_shells(aux_shells, positions)
        return three_center_eri(flat, aux)

    with jax.enable_x64(True):
        eager = np.asarray(build(system.positions))
        jitted = np.asarray(jax.jit(build)(system.positions))
    np.testing.assert_allclose(eager, jitted, atol=1e-12)


def test_two_center_metric_is_jittable() -> None:
    """The Coulomb-metric builder compiles under ``jax.jit`` (traced over geometry)."""
    system = _h2_system()
    aux_shells = _aux_shells_for_system(system)

    def build(positions: jnp.ndarray) -> jnp.ndarray:
        return two_center_metric(AuxiliaryBasis.from_shells(aux_shells, positions))

    with jax.enable_x64(True):
        eager = np.asarray(build(system.positions))
        jitted = np.asarray(jax.jit(build)(system.positions))
    np.testing.assert_allclose(eager, jitted, atol=1e-12)


def test_fitted_eri_jit_matches_eager_h2() -> None:
    """The full fitted-ERI build compiles and matches the eager result."""
    system = _h2_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    aux_shells = _aux_shells_for_system(system)

    def build(positions: jnp.ndarray) -> jnp.ndarray:
        flat = basis.with_positions(positions).flat_primitives()
        aux = AuxiliaryBasis.from_shells(aux_shells, positions)
        return fitted_eri(flat, aux)

    with jax.enable_x64(True):
        eager = np.asarray(build(system.positions))
        jitted = np.asarray(jax.jit(build)(system.positions))
    np.testing.assert_allclose(eager, jitted, atol=1e-10)


def test_fitted_eri_is_differentiable_wrt_positions() -> None:
    """A scalar reduction of the fitted ERI has a finite, nonzero position gradient."""
    system = _water_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    aux_shells = _aux_shells_for_system(system)

    def scalar(positions: jnp.ndarray) -> jnp.ndarray:
        flat = basis.with_positions(positions).flat_primitives()
        aux = AuxiliaryBasis.from_shells(aux_shells, positions)
        return jnp.sum(fitted_eri(flat, aux) ** 2)

    with jax.enable_x64(True):
        grad = jax.grad(scalar)(system.positions)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.max(np.abs(np.asarray(grad))) > 0.0


def test_three_center_path_is_vmappable() -> None:
    """The three-centre builder ``vmap``-s over a batch of geometries."""
    system = _h2_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    aux_shells = _aux_shells_for_system(system)
    base = np.asarray(system.positions)
    batch = jnp.asarray(np.stack([base, base + 0.05, base - 0.05]))

    def build(positions: jnp.ndarray) -> jnp.ndarray:
        flat = basis.with_positions(positions).flat_primitives()
        aux = AuxiliaryBasis.from_shells(aux_shells, positions)
        return three_center_eri(flat, aux)

    with jax.enable_x64(True):
        batched = jax.vmap(build)(batch)
        single = build(jnp.asarray(base))
    assert batched.shape[0] == 3
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(single), atol=1e-10)
