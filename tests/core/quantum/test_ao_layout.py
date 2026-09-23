r"""Tests for the def2-SVP per-element orbital layout and validity masks.

The full second-row AO block is 14-dimensional, laid out as the irrep
``3x0e + 2x1e + 1x2e`` (3 s-shells, 2 p-shells, 1 d-shell; ``3 + 2*3 + 5 = 14``).
QHNet (Yu et al. 2023, arXiv:2306.04922) predicts a
dense ``(14, 14)`` block per atom/edge and masks out the AO slots an element does
not populate: H keeps ``[0, 1, 3, 4, 5]`` (2 s + 1 p), C/N/O/F keep all 14.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum.ao_layout import (
    ao_permutation,
    atom_orbital_counts,
    BACK2PYSCF,
    block_validity_mask,
    FULL_ORBITALS,
    native_ao_count,
    ORBITAL_MASK,
    PYSCF_DEF2SVP,
)


def test_full_orbitals_is_fourteen() -> None:
    """The def2-SVP second-row full AO count is 14."""
    assert FULL_ORBITALS == 14


def test_hydrogen_orbital_mask_indices() -> None:
    """H keeps the def2-SVP 2 s + 1 p slots ``[0, 1, 3, 4, 5]``."""
    assert ORBITAL_MASK[1] == (0, 1, 3, 4, 5)


def test_heavy_elements_keep_all_orbitals() -> None:
    """C/N/O/F populate every one of the 14 AO slots."""
    for atomic_number in (6, 7, 8, 9):
        assert ORBITAL_MASK[atomic_number] == tuple(range(FULL_ORBITALS))


def test_atom_orbital_counts() -> None:
    """Per-atom AO count: 5 for H, 14 for C/N/O/F."""
    atomic_numbers = jnp.asarray([1, 6, 7, 8, 9])
    counts = atom_orbital_counts(atomic_numbers)
    np.testing.assert_array_equal(np.asarray(counts), np.asarray([5, 14, 14, 14, 14]))


def test_block_validity_mask_shape() -> None:
    """A per-atom validity mask has shape ``(..., 14, 14)``."""
    atomic_numbers = jnp.asarray([1, 6])
    mask = block_validity_mask(atomic_numbers)
    assert mask.shape == (2, 14, 14)
    assert mask.dtype == jnp.bool_


def test_hydrogen_hydrogen_pair_is_five_by_five() -> None:
    """An H-H block keeps a 5x5 valid region (outer product of the H masks)."""
    mask = block_validity_mask(jnp.asarray(1), jnp.asarray(1))
    assert mask.shape == (14, 14)
    assert int(jnp.sum(mask)) == 25
    valid = (0, 1, 3, 4, 5)
    for row in range(14):
        for col in range(14):
            expected = row in valid and col in valid
            assert bool(mask[row, col]) == expected


def test_carbon_hydrogen_pair_is_fourteen_by_five() -> None:
    """A C-H block keeps the 14x5 region: all carbon rows x the 5 H columns."""
    mask = block_validity_mask(jnp.asarray(6), jnp.asarray(1))
    assert int(jnp.sum(mask)) == 14 * 5
    valid_cols = (0, 1, 3, 4, 5)
    np.testing.assert_array_equal(
        np.asarray(jnp.all(mask, axis=0)),
        np.asarray(jnp.asarray([col in valid_cols for col in range(14)])),
    )
    # Every carbon row is valid.
    assert bool(jnp.all(mask[:, 0]))


def test_block_validity_mask_is_outer_product() -> None:
    """The pair mask is the outer product of the row and column atom masks."""
    row_numbers = jnp.asarray([1, 6])
    col_numbers = jnp.asarray([6, 1])
    pair_mask = block_validity_mask(row_numbers, col_numbers)
    row_self = block_validity_mask(row_numbers)
    col_self = block_validity_mask(col_numbers)
    # Diagonal of the self masks gives the per-atom 1-D AO validity.
    row_valid = jnp.diagonal(row_self, axis1=-2, axis2=-1)
    col_valid = jnp.diagonal(col_self, axis1=-2, axis2=-1)
    expected = row_valid[:, :, None] & col_valid[:, None, :]
    np.testing.assert_array_equal(np.asarray(pair_mask), np.asarray(expected))


def test_block_validity_mask_is_vmap_safe() -> None:
    """``block_validity_mask`` composes under ``jax.vmap`` (jit/vmap clean)."""
    atomic_numbers = jnp.asarray([1, 6, 8])
    batched = jax.vmap(block_validity_mask)(atomic_numbers)
    direct = block_validity_mask(atomic_numbers)
    np.testing.assert_array_equal(np.asarray(batched), np.asarray(direct))


def test_block_validity_mask_is_jit_safe() -> None:
    """``block_validity_mask`` traces cleanly under ``jax.jit``."""
    jitted = jax.jit(block_validity_mask)
    mask = jitted(jnp.asarray([1, 9]))
    assert mask.shape == (2, 14, 14)


# The AO permutations and counts below are the published QH9 conventions (Yu et al. 2023,
# arXiv:2306.09549), pinned as they were produced before the layout moved to one module.
_WATER = (8, 1, 1)
_WATER_QH9_TO_PYSCF = (
    0,
    1,
    2,
    4,
    5,
    3,
    7,
    8,
    6,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    18,
    16,
    19,
    20,
    22,
    23,
    21,
)
_WATER_BACK2PYSCF = (
    0,
    1,
    2,
    5,
    3,
    4,
    8,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    18,
    16,
    17,
    19,
    20,
    23,
    21,
    22,
)
_EVERY_ELEMENT = (6, 7, 8, 9, 1)


class TestAoPermutation:
    """One construction serves both reorderings, differing only in the component order."""

    def test_qh9_to_pyscf_def2svp_on_water(self) -> None:
        indices, signs = ao_permutation(np.asarray(_WATER), PYSCF_DEF2SVP)
        assert tuple(indices.tolist()) == _WATER_QH9_TO_PYSCF
        assert set(signs.tolist()) == {1}

    def test_spherical_to_pyscf_internal_on_water(self) -> None:
        indices, _ = ao_permutation(np.asarray(_WATER), BACK2PYSCF)
        assert tuple(indices.tolist()) == _WATER_BACK2PYSCF

    @pytest.mark.parametrize("components", ["PYSCF_DEF2SVP", "BACK2PYSCF"])
    def test_every_element_gives_a_permutation(self, components: str) -> None:
        order = {"PYSCF_DEF2SVP": PYSCF_DEF2SVP, "BACK2PYSCF": BACK2PYSCF}[components]
        indices, signs = ao_permutation(np.asarray(_EVERY_ELEMENT), order)
        assert sorted(indices.tolist()) == list(range(native_ao_count(np.asarray(_EVERY_ELEMENT))))
        assert indices.dtype == np.int64
        assert signs.shape == indices.shape

    def test_the_two_orderings_are_inverse_permutations(self) -> None:
        forward, _ = ao_permutation(np.asarray(_EVERY_ELEMENT), PYSCF_DEF2SVP)
        back, _ = ao_permutation(np.asarray(_EVERY_ELEMENT), BACK2PYSCF)
        np.testing.assert_array_equal(forward[back], np.arange(forward.size))

    def test_an_element_outside_the_basis_is_refused(self) -> None:
        with pytest.raises(KeyError):
            ao_permutation(np.asarray([16]), PYSCF_DEF2SVP)


class TestNativeAoCount:
    """Five AOs per hydrogen, fourteen per C, N, O or F; nothing else is in the basis."""

    @pytest.mark.parametrize(("atoms", "count"), [((1,), 5), (_WATER, 24), (_EVERY_ELEMENT, 61)])
    def test_counts(self, atoms: tuple[int, ...], count: int) -> None:
        assert native_ao_count(np.asarray(atoms)) == count

    def test_it_agrees_with_the_traced_count(self) -> None:
        atoms = np.asarray(_EVERY_ELEMENT)
        assert native_ao_count(atoms) == int(jnp.sum(atom_orbital_counts(jnp.asarray(atoms))))

    @pytest.mark.parametrize("atomic_number", [2, 16])
    def test_an_element_outside_the_basis_is_refused(self, atomic_number: int) -> None:
        with pytest.raises(KeyError):
            native_ao_count(np.asarray([atomic_number]))


class TestElementsOutsideTheBasis:
    """An element the layout does not describe populates nothing, traced or not.

    The lookup table has one row per atomic number up to fluorine. A plain gather clamps an
    out-of-range index to the last row, so sulphur used to read fluorine's fourteen slots,
    and a negative index wraps to it.
    """

    @pytest.mark.parametrize("atomic_number", [2, 10, 16, -1])
    def test_no_orbitals(self, atomic_number: int) -> None:
        assert int(atom_orbital_counts(jnp.asarray(atomic_number))) == 0

    @pytest.mark.parametrize("atomic_number", [10, 16, -1])
    def test_an_empty_block_even_beside_a_heavy_atom(self, atomic_number: int) -> None:
        mask = block_validity_mask(jnp.asarray(atomic_number), jnp.asarray(9))
        assert not bool(mask.any())

    def test_under_jit_and_vmap(self) -> None:
        atoms = jnp.asarray([1, 9, 10, 16, -1, 0])
        expected = [5, 14, 0, 0, 0, 0]
        np.testing.assert_array_equal(np.asarray(jax.jit(atom_orbital_counts)(atoms)), expected)
        np.testing.assert_array_equal(np.asarray(jax.vmap(atom_orbital_counts)(atoms)), expected)


def test_every_shell_string_fills_its_element_mask() -> None:
    """The shell composition and the slot mask describe the same element."""
    sizes = {"s": 1, "p": 3, "d": 5}
    assert set(PYSCF_DEF2SVP.shells) == set(ORBITAL_MASK)
    for atomic_number, shells in PYSCF_DEF2SVP.shells.items():
        assert sum(sizes[shell] for shell in shells) == len(ORBITAL_MASK[atomic_number])
