# ruff: noqa: UP037
r"""def2-SVP per-element orbital layout and AO validity masks (QHNet block form).

A heterogeneous-batchable Hamiltonian predictor emits a *fixed* dense block per
atom (diagonal Fock block) and per directed edge (off-diagonal block), then masks
out the AO slots an element does not populate. For the full second-row def2-SVP
basis the block is 14-dimensional, laid out as the irrep

.. math::  \mathrm{BLOCK\_IRREPS} = 3\!\times\!0e + 2\!\times\!1e + 1\!\times\!2e,

i.e. 3 s-shells, 2 p-shells and 1 d-shell (``3 + 2*3 + 5 = 14``). The per-element
AO mask follows QHNet's ``orbital_mask`` (Yu et al. 2023, "QHNet",
arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/datasets.py``): hydrogen/helium keep the 2 s + 1 p slots
``[0, 1, 3, 4, 5]`` while C/N/O/F populate all 14. A directed-edge (pair) block's
validity mask is the outer product of its row element's and column element's
per-atom masks.

The masks are derived from a static ``(max_Z+1, 14)`` boolean lookup table built
from :data:`ORBITAL_MASK`, so :func:`block_validity_mask` and
:func:`atom_orbital_counts` are pure index gathers -- fully ``jit``/``vmap`` clean
(the table is a compile-time-constant array; the static metadata lives in plain
Python tuples).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int  # noqa: TC002

from opifex.neural.equivariant import Irreps


FULL_ORBITALS: int = 14
"""def2-SVP full second-row AO count (``3 s + 2 p + 1 d`` = ``3 + 6 + 5``)."""

BLOCK_IRREPS: Irreps = Irreps("3x0e + 2x1e + 1x2e")
r"""The 14-dim row/col representation of a Fock block (``3x0e + 2x1e + 1x2e``)."""

_HYDROGEN_AO_INDICES: tuple[int, ...] = (0, 1, 3, 4, 5)
"""QH9 ``idx_1s_2s_2p``: the 2 s + 1 p slots H/He populate in the 14-slot block."""

ORBITAL_MASK: dict[int, tuple[int, ...]] = {
    1: _HYDROGEN_AO_INDICES,
    6: tuple(range(FULL_ORBITALS)),
    7: tuple(range(FULL_ORBITALS)),
    8: tuple(range(FULL_ORBITALS)),
    9: tuple(range(FULL_ORBITALS)),
}
"""Atomic number -> valid AO indices into the 14-slot irrep-ordered block."""


def _build_validity_table() -> np.ndarray:
    """Return the static ``(max_Z + 1, 14)`` boolean per-element AO validity table.

    Row ``Z`` is the 1-D mask of which of the 14 AO slots element ``Z`` populates;
    unlisted atomic numbers (and the ``Z = 0`` padding row) are all-``False``.
    """
    max_z = max(ORBITAL_MASK)
    table = np.zeros((max_z + 1, FULL_ORBITALS), dtype=bool)
    for atomic_number, indices in ORBITAL_MASK.items():
        table[atomic_number, list(indices)] = True
    return table


_VALIDITY_TABLE: np.ndarray = _build_validity_table()
"""Compile-time-constant per-element AO validity lookup (``numpy``, module-level)."""


def _atom_orbital_validity(atomic_numbers: Int[Array, "..."]) -> Bool[Array, "... 14"]:
    """Gather the per-atom 1-D ``(..., 14)`` AO validity mask for each atom.

    Args:
        atomic_numbers: Integer atomic numbers ``Z`` (any leading shape).

    Returns:
        Boolean mask of shape ``atomic_numbers.shape + (14,)``.
    """
    table = jnp.asarray(_VALIDITY_TABLE)
    return table[atomic_numbers]


def atom_orbital_counts(atomic_numbers: Int[Array, "..."]) -> Int[Array, "..."]:
    """Return the number of populated AOs per atom (5 for H/He, 14 for C/N/O/F).

    Args:
        atomic_numbers: Integer atomic numbers ``Z`` (any leading shape).

    Returns:
        Integer AO counts of shape ``atomic_numbers.shape``.
    """
    return jnp.sum(_atom_orbital_validity(atomic_numbers), axis=-1)


def block_validity_mask(
    row_atomic_numbers: Int[Array, "..."],
    col_atomic_numbers: Int[Array, "..."] | None = None,
) -> Bool[Array, "... 14 14"]:
    r"""Return the ``(..., 14, 14)`` AO validity mask of an atom or directed pair.

    For a single atom (``col_atomic_numbers is None``) the mask is the outer
    product of the atom's per-AO validity with itself (the diagonal Fock block).
    For a directed edge it is the outer product of the **row** element's mask and
    the **column** element's mask -- ``mask[i, j] = row_valid[i] & col_valid[j]``
    -- matching QHNet's per-pair ``matrix_block_mask`` (reference
    ``OpenDFT/QHBench/QH9/datasets.py``).

    Args:
        row_atomic_numbers: Atomic numbers ``Z`` of the row (receiver) atoms.
        col_atomic_numbers: Atomic numbers ``Z`` of the column (sender) atoms; if
            ``None``, the row atoms are reused (diagonal block).

    Returns:
        Boolean mask of shape ``row_atomic_numbers.shape + (14, 14)``.
    """
    if col_atomic_numbers is None:
        col_atomic_numbers = row_atomic_numbers
    row_valid = _atom_orbital_validity(row_atomic_numbers)
    col_valid = _atom_orbital_validity(col_atomic_numbers)
    return row_valid[..., :, None] & col_valid[..., None, :]


__all__ = [
    "BLOCK_IRREPS",
    "FULL_ORBITALS",
    "ORBITAL_MASK",
    "atom_orbital_counts",
    "block_validity_mask",
]
