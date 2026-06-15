r"""Tests for the def2-SVP per-element orbital layout and validity masks.

The full second-row AO block is 14-dimensional, laid out as the irrep
``3x0e + 2x1e + 1x2e`` (3 s-shells, 2 p-shells, 1 d-shell; ``3 + 2*3 + 5 = 14``).
QHNet (Yu et al. 2023, arXiv:2306.04922; reference
``divelab/AIRS`` ``OpenDFT/QHBench/QH9/datasets.py`` ``orbital_mask``) predicts a
dense ``(14, 14)`` block per atom/edge and masks out the AO slots an element does
not populate: H/He keep ``[0, 1, 3, 4, 5]`` (2 s + 1 p), C/N/O/F keep all 14.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.equivariant import Irreps
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    atom_orbital_counts,
    BLOCK_IRREPS,
    block_validity_mask,
    FULL_ORBITALS,
    ORBITAL_MASK,
)


def test_full_orbitals_is_fourteen() -> None:
    """The def2-SVP second-row full AO count is 14."""
    assert FULL_ORBITALS == 14


def test_block_irreps_layout() -> None:
    """``BLOCK_IRREPS`` is ``3x0e + 2x1e + 1x2e`` with total dimension 14."""
    assert Irreps("3x0e + 2x1e + 1x2e") == BLOCK_IRREPS
    assert BLOCK_IRREPS.dim == FULL_ORBITALS


def test_hydrogen_orbital_mask_indices() -> None:
    """H keeps the 2 s + 1 p slots ``[0, 1, 3, 4, 5]`` (QH9 ``idx_1s_2s_2p``)."""
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
