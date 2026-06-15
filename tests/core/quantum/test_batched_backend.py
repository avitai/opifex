"""Tests for the batched (flat-primitive) Gaussian-integral engine.

The batched McMurchie-Davidson backend assembles every AO integral tensor with a
single ``vmap`` over a flat primitive pytree followed by ``segment_sum``
contraction (the MESS batching pattern). These tests check that

* the flat-primitive representation is internally consistent,
* the batched ``S/T/V/ERI`` reproduce the per-primitive MMD math (the validated
  eager kernels) to ~1e-10,
* the full H2O integral build *compiles* under ``jax.jit`` (the eager
  shell-loop build did not), and
* the build stays differentiable with respect to the nuclear positions.

PySCF cross-checks live in ``test_backend.py`` (the slow oracle suite); here the
eager per-primitive kernels are the fast reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum.backend import (
    boys_function,
    JaxGaussianBackend,
)
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_system() -> MolecularSystem:
    bond = 0.74 * _BOHR_PER_ANGSTROM
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
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


# ---------------------------------------------------------------------------
# Flat-primitive representation
# ---------------------------------------------------------------------------


def test_flat_primitives_orbital_index_covers_all_aos() -> None:
    """Every AO appears in the primitive->orbital map; counts match the basis."""
    basis = AtomicOrbitalBasis.from_molecular_system(_water_system())
    flat = basis.flat_primitives()

    assert flat.orbital_index.min() == 0
    assert flat.orbital_index.max() == basis.n_atomic_orbitals - 1
    assert len(np.unique(flat.orbital_index)) == basis.n_atomic_orbitals
    # Water STO-3G: O has 1s,2s (3 prim each) + 2p (3 comps x 3 prim = 9 prim per
    # AO is wrong -- each Cartesian p AO has 3 primitives); H 1s (3 prim each).
    # O: 1s(3) + 2s(3) + px(3)+py(3)+pz(3) = 15; H: 3 each -> total 15 + 6 = 21.
    assert flat.alpha.shape[0] == 21
    assert flat.center.shape == (21, 3)
    assert flat.lmn.shape == (21, 3)


def test_flat_primitives_lmn_is_static_numpy() -> None:
    """Angular-momentum powers are NumPy-static integer arrays (jit-safe)."""
    flat = AtomicOrbitalBasis.from_molecular_system(_water_system()).flat_primitives()
    assert isinstance(flat.lmn, np.ndarray)
    assert isinstance(flat.orbital_index, np.ndarray)
    assert np.issubdtype(flat.lmn.dtype, np.integer)


# ---------------------------------------------------------------------------
# Boys function (jnp.select branches)
# ---------------------------------------------------------------------------


def test_boys_matches_reference_across_regimes() -> None:
    """Boys ``F_n(x)`` matches a high-accuracy series across t in {0, small, large}."""
    with jax.enable_x64(True):
        for order in (0, 1, 2, 3, 4):
            for x in (0.0, 1e-6, 0.5, 12.0, 60.0, 200.0):
                got = float(boys_function(order, jnp.asarray(x)))
                ref = _boys_reference(order, x)
                assert got == pytest.approx(ref, rel=1e-9, abs=1e-12)


def _boys_reference(n: int, x: float) -> float:
    """Reference Boys ``F_n(x)`` via the stable ascending series / analytic limit."""
    import math

    if x == 0.0:
        return 1.0 / (2 * n + 1)
    a = n + 0.5
    total = 1.0 / a
    term = 1.0 / a
    k = 0
    while k < 1000:
        k += 1
        term *= x / (a + k)
        total += term
        if term < 1e-18 * total:
            break
    return 0.5 * math.exp(-x) * total


# ---------------------------------------------------------------------------
# Batched-vs-eager equivalence (the engine's own correctness contract)
# ---------------------------------------------------------------------------


def test_overlap_h2_is_symmetric_unit_diagonal() -> None:
    """Batched overlap is symmetric with unit diagonal."""
    with jax.enable_x64(True):
        s = np.asarray(_backend(_h2_system()).overlap())
    np.testing.assert_allclose(s, s.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(s), 1.0, atol=1e-10)


def test_h2_one_electron_matches_eager_kernels() -> None:
    """Batched S/T/V on H2 match the per-primitive eager reference to ~1e-10."""
    from opifex.core.quantum._eager_reference import eager_one_electron

    system = _h2_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    with jax.enable_x64(True):
        backend = JaxGaussianBackend(system, basis)
        s = np.asarray(backend.overlap())
        t = np.asarray(backend.kinetic())
        v = np.asarray(backend.nuclear_attraction())
        s_ref, t_ref, v_ref = eager_one_electron(system, basis)
    np.testing.assert_allclose(s, np.asarray(s_ref), atol=1e-10)
    np.testing.assert_allclose(t, np.asarray(t_ref), atol=1e-10)
    np.testing.assert_allclose(v, np.asarray(v_ref), atol=1e-10)


def test_h2_eri_matches_eager_kernels() -> None:
    """Batched ERI on H2 matches the per-primitive eager reference to ~1e-10.

    H2 (16 AO quartets) exercises the full ERI batching path against the eager
    primitive kernels cheaply; the p-shell quartets are covered by the slow
    water S/T/V eager check below and by the PySCF water-ERI oracle in
    ``test_backend.py``.
    """
    from opifex.core.quantum._eager_reference import eager_eri

    system = _h2_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    with jax.enable_x64(True):
        eri = np.asarray(JaxGaussianBackend(system, basis).electron_repulsion())
        eri_ref = np.asarray(eager_eri(system, basis))
    np.testing.assert_allclose(eri, eri_ref, atol=1e-10)


@pytest.mark.slow
def test_water_one_electron_matches_eager_kernels() -> None:
    """Batched S/T/V on water (p-shells) match the eager reference to ~1e-10.

    Exercises the p-shell angular-momentum recurrences in the batched harness;
    the eager AO-by-AO reference is slow (~10s) so this is opt-in.
    """
    from opifex.core.quantum._eager_reference import eager_one_electron

    system = _water_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)
    with jax.enable_x64(True):
        backend = JaxGaussianBackend(system, basis)
        s = np.asarray(backend.overlap())
        t = np.asarray(backend.kinetic())
        v = np.asarray(backend.nuclear_attraction())
        s_ref, t_ref, v_ref = eager_one_electron(system, basis)
    np.testing.assert_allclose(s, np.asarray(s_ref), atol=1e-10)
    np.testing.assert_allclose(t, np.asarray(t_ref), atol=1e-10)
    np.testing.assert_allclose(v, np.asarray(v_ref), atol=1e-10)


_D_SHELL = ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))


def test_d_shell_primitive_kernels_match_eager_reference() -> None:
    """The fixed-size kernels reproduce the eager MMD math up to ``d`` shells.

    STO-3G only tabulates ``s``/``p`` shells, but the flat kernels are sized by a
    static ``max_l`` and must generalise. This checks ``max_l = 2`` overlap,
    kinetic and ERI primitives against the validated eager per-primitive kernels
    (the same recurrences) to machine precision.
    """
    from opifex.core.quantum import _eager_reference as eager, _flat_mmd as flat

    with jax.enable_x64(True):
        center_a = jnp.array([0.0, 0.0, 0.0])
        center_b = jnp.array([-0.3, 0.2, -0.5])
        center_c = jnp.array([0.2, 0.1, -0.3])
        center_d = jnp.array([0.4, -0.1, 0.2])
        rab = center_a - center_b
        rcd = center_c - center_d
        exp_a, exp_b, exp_c, exp_d = (
            jnp.asarray(1.3),
            jnp.asarray(0.7),
            jnp.asarray(0.9),
            jnp.asarray(1.1),
        )
        max_l = 2
        worst_s = worst_t = worst_e = 0.0
        for la in (*_D_SHELL, (0, 0, 0), (1, 0, 0)):
            for lb in (*_D_SHELL, (0, 0, 0)):
                s_e, t_e = eager._primitive_overlap_kinetic(la, lb, rab, exp_a, exp_b)
                s_f = flat.overlap_primitive(
                    jnp.array(la), jnp.array(lb), center_a, center_b, exp_a, exp_b, max_l
                )
                t_f = flat.kinetic_primitive(
                    jnp.array(la), jnp.array(lb), center_a, center_b, exp_a, exp_b, max_l
                )
                worst_s = max(worst_s, abs(float(s_e) - float(s_f)))
                worst_t = max(worst_t, abs(float(t_e) - float(t_f)))
        for la in ((2, 0, 0), (1, 1, 0)):
            for lc in ((0, 0, 2), (0, 0, 0)):
                center_p = (exp_a * center_a + exp_b * center_b) / (exp_a + exp_b)
                center_q = (exp_c * center_c + exp_d * center_d) / (exp_c + exp_d)
                lb, ld = (0, 1, 0), (1, 0, 0)
                max_total = sum(la) + sum(lb) + sum(lc) + sum(ld)
                v_e = eager._primitive_eri(
                    la,
                    lb,
                    lc,
                    ld,
                    rab,
                    rcd,
                    exp_a,
                    exp_b,
                    exp_c,
                    exp_d,
                    center_p,
                    center_q,
                    max_total,
                )
                v_f = flat.eri_primitive(
                    jnp.array(la),
                    jnp.array(lb),
                    jnp.array(lc),
                    jnp.array(ld),
                    center_a,
                    center_b,
                    center_c,
                    center_d,
                    exp_a,
                    exp_b,
                    exp_c,
                    exp_d,
                    max_l,
                )
                worst_e = max(worst_e, abs(float(v_e) - float(v_f)))
    assert worst_s < 1e-12
    assert worst_t < 1e-12
    assert worst_e < 1e-12


def test_eri_has_eightfold_permutation_symmetry() -> None:
    """The batched ERI tensor obeys the 8-fold chemist-notation symmetry."""
    with jax.enable_x64(True):
        eri = np.asarray(_backend(_water_system()).electron_repulsion())
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)


# ---------------------------------------------------------------------------
# JIT + differentiability (the headline requirement)
# ---------------------------------------------------------------------------


def test_full_water_integral_build_jit_compiles() -> None:
    """The full H2O S/T/V/ERI build compiles and runs under ``jax.jit``."""
    system = _water_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)

    def build(positions: jnp.ndarray) -> tuple:
        moved = basis.with_positions(positions)
        moved_system = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set="sto-3g",
        )
        backend = JaxGaussianBackend(moved_system, moved)
        return (
            backend.overlap(),
            backend.core_hamiltonian(),
            backend.electron_repulsion(),
            backend.nuclear_repulsion(),
        )

    with jax.enable_x64(True):
        jitted = jax.jit(build)
        s, _h, eri, enn = jitted(system.positions)
        s_eager, _h_eager, eri_eager, enn_eager = build(system.positions)
    np.testing.assert_allclose(np.asarray(s), np.asarray(s_eager), atol=1e-10)
    np.testing.assert_allclose(np.asarray(eri), np.asarray(eri_eager), atol=1e-10)
    assert float(enn) == pytest.approx(float(enn_eager), abs=1e-10)


def test_eri_is_differentiable_wrt_positions() -> None:
    """A scalar reduction of the ERI tensor has a finite gradient w.r.t. positions."""
    system = _water_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)

    def scalar(positions: jnp.ndarray) -> jnp.ndarray:
        moved = basis.with_positions(positions)
        moved_system = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set="sto-3g",
        )
        return jnp.sum(JaxGaussianBackend(moved_system, moved).electron_repulsion() ** 2)

    with jax.enable_x64(True):
        grad = jax.grad(scalar)(system.positions)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.max(np.abs(np.asarray(grad))) > 0.0


def test_nuclear_repulsion_force_matches_finite_difference() -> None:
    """Analytic dE_nn/dR matches a central finite difference."""
    system = _water_system()
    basis = AtomicOrbitalBasis.from_molecular_system(system)

    def e_nn(positions: jnp.ndarray) -> jnp.ndarray:
        moved_system = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set="sto-3g",
        )
        return JaxGaussianBackend(moved_system, basis.with_positions(positions)).nuclear_repulsion()

    with jax.enable_x64(True):
        analytic = np.asarray(jax.grad(e_nn)(system.positions))
        eps = 1e-5
        numeric = np.zeros_like(analytic)
        base = np.asarray(system.positions)
        for i in range(base.shape[0]):
            for j in range(3):
                plus = base.copy()
                plus[i, j] += eps
                minus = base.copy()
                minus[i, j] -= eps
                numeric[i, j] = (
                    float(e_nn(jnp.asarray(plus))) - float(e_nn(jnp.asarray(minus)))
                ) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)
