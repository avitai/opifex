r"""QH9-Stable per-molecule Fock block cut (the operator equivalence reference).

The :class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
emits a fixed ``(14, 14)`` diagonal Fock block per atom and a ``(14, 14)``
off-diagonal block per directed within-molecule edge. This module produces the
matching *targets* on the host in NumPy: it cuts each molecule's spherical
def2-SVP Fock matrix into those blocks (QHNet ``cut_matrix``; Yu et al. 2023,
"QHNet"/"QH9", arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/datasets.py``). :func:`cut_fock_to_blocks` and its inverse
:func:`reconstruct_fock_from_blocks` are the *record of equivalence* the device
block-cut operator (:mod:`opifex.data.sources.qh9_fock_operators`) is tested
against -- the live training path runs the cut on the GPU; this NumPy cut is the
ground-truth reference and the round-trip check.

Reuse (DRY)
-----------
The 14-slot per-element AO layout and validity masks come from
:mod:`opifex.neural.quantum.hamiltonian._orbital_layout`
(:data:`~...FULL_ORBITALS`, :data:`~...ORBITAL_MASK`,
:func:`~...block_validity_mask`). Because the per-atom AO order of the spherical
Fock matches the 14-slot block layout (``3 s + 2 p + 1 d`` = ``BLOCK_IRREPS``),
cutting atom ``i``'s contiguous AO range into its element's ``ORBITAL_MASK[Z_i]``
slot indices is exact (verified by the round-trip test to ``1e-10``).

The block cut (per molecule)
----------------------------
For a complete *directed* molecular graph (every ordered pair ``(i, j)`` with
``i != j``):

* ``diagonal_blocks[a]`` -- the ``(14, 14)`` block with ``H[block_a, block_a]``
  scattered into rows/cols ``ORBITAL_MASK[Z_a]``; ``diagonal_mask[a] =
  block_validity_mask(Z_a)``.
* ``off_diagonal_blocks[e]`` for edge ``e`` with ``edge_index[:, e] = (row, col)``
  -- the ``(14, 14)`` block with ``H[block_row, block_col]`` at
  ``ORBITAL_MASK[Z_row] x ORBITAL_MASK[Z_col]``; ``off_diagonal_mask[e] =
  block_validity_mask(Z_row, Z_col)``.

Edge indexing matches QHNet ``cut_matrix`` (``edge_index_full.append([idx_j,
idx_i])`` = ``[dst, src]``) and the predictor's
:meth:`~...block_predictor.BlockHamiltonianPredictor.assemble_matrix`: row 0 is
the *receiver* (block row), row 1 the *sender* (block column). The complete graph
carries both ``(i, j)`` and ``(j, i)``, so the predictor's assembly
symmetrisation ``H = H~ + H~^T`` reproduces QHNet's off-diagonal law.

Bounded scope
-------------
Edges are the *complete directed graph* (no radius cutoff): QH9 molecules are
small (``n_atoms`` 3..29) and QHNet's reference ``cut_matrix`` itself uses the
complete graph, so a cutoff would drop real off-diagonal Fock blocks the loss
must supervise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence  # noqa: TC003

import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.neural.quantum.hamiltonian._orbital_layout import (
    block_validity_mask,
    FULL_ORBITALS,
    ORBITAL_MASK,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Per-molecule block cut (QHNet cut_matrix, NumPy host-side)
# =============================================================================


def _atom_ao_offsets(
    atomic_numbers: Int[NDArray[np.integer], " n_atoms"],
) -> NDArray[np.int64]:
    """Return the ``(n_atoms + 1,)`` cumulative AO offsets of each atom's block.

    Atom ``a`` occupies AO rows/cols ``[offsets[a], offsets[a + 1])`` of the
    spherical def2-SVP Fock matrix; the block size is ``len(ORBITAL_MASK[Z_a])``
    (5 for H/He, 14 for C/N/O/F), matching the QH9-native per-element AO count.
    """
    counts = np.asarray([len(ORBITAL_MASK[int(z)]) for z in atomic_numbers], dtype=np.int64)
    offsets = np.zeros(len(atomic_numbers) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _complete_directed_edges(n_atoms: int) -> NDArray[np.int64]:
    """Return the complete directed ``(2, n_atoms*(n_atoms-1))`` edge index.

    Row 0 is the *receiver* (block row), row 1 the *sender* (block column),
    matching QHNet ``cut_matrix``'s ``edge_index_full.append([idx_j, idx_i])``
    (``[dst, src]``) and the predictor's ``assemble_matrix`` convention. Ordering
    is sender-major over ``idx_i`` then ``idx_j`` (the reference loop order).
    """
    edges: list[tuple[int, int]] = []
    for idx_i in range(n_atoms):  # sender / column (outer loop, QHNet src)
        for idx_j in range(n_atoms):  # receiver / row (inner loop, QHNet dst)
            if idx_i != idx_j:
                edges.append((idx_j, idx_i))  # [receiver, sender] = [row, col]
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _scatter_block(
    full_block: NDArray[np.float64],
    sub_matrix: NDArray[np.float64],
    row_indices: Sequence[int],
    col_indices: Sequence[int],
) -> None:
    """Scatter ``sub_matrix`` into ``full_block`` at ``row_indices x col_indices``.

    Mutates ``full_block`` in place (it is a freshly allocated zero block owned by
    the caller), mirroring the QHNet ``cut_matrix`` masked assignment
    ``matrix_block[mask_j][:, mask_i] = extracted_matrix``.
    """
    full_block[np.ix_(list(row_indices), list(col_indices))] = sub_matrix


def cut_fock_to_blocks(
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"],
    fock: Float[NDArray[np.float64], "n_ao n_ao"],
) -> tuple[
    Float[NDArray[np.float64], "n_atoms 14 14"],
    Bool[NDArray[np.bool_], "n_atoms 14 14"],
    Float[NDArray[np.float64], "n_edges 14 14"],
    Bool[NDArray[np.bool_], "n_edges 14 14"],
    Int[NDArray[np.int64], "2 n_edges"],
]:
    r"""Cut a spherical def2-SVP Fock matrix into QHNet per-atom/per-edge blocks.

    Faithful NumPy reimplementation of QHNet's ``cut_matrix`` (reference
    ``OpenDFT/QHBench/QH9/datasets.py``): atom ``a``'s contiguous AO range of the
    spherical Fock is scattered into a ``(14, 14)`` block at its element's
    :data:`~...ORBITAL_MASK` slot indices, with the validity mask from
    :func:`~...block_validity_mask`. The off-diagonal edge set is the complete
    *directed* graph; ``edge_index[:, e] = (receiver, sender)`` so the block holds
    ``H[block_receiver, block_sender]``.

    Args:
        atomic_numbers: Nuclear charges ``(n_atoms,)`` (QH9 elements H/C/N/O/F).
        fock: Spherical def2-SVP Fock matrix ``(n_ao, n_ao)`` in opifex ordering
            (the target produced by
            :func:`~opifex.data.sources.qh9_source.read_qh9_sqlite`).

    Returns:
        ``(diagonal_blocks, diagonal_mask, off_diagonal_blocks,
        off_diagonal_mask, edge_index)`` with shapes ``(A, 14, 14)``,
        ``(A, 14, 14)``, ``(E, 14, 14)``, ``(E, 14, 14)`` and ``(2, E)`` where
        ``E = A * (A - 1)``.
    """
    charges = np.asarray(atomic_numbers).astype(np.int64)
    matrix = np.asarray(fock, dtype=np.float64)
    n_atoms = int(charges.shape[0])
    offsets = _atom_ao_offsets(charges)
    slot_indices = [ORBITAL_MASK[int(z)] for z in charges]

    diagonal_blocks = np.zeros((n_atoms, FULL_ORBITALS, FULL_ORBITALS), dtype=np.float64)
    for atom in range(n_atoms):
        start, stop = int(offsets[atom]), int(offsets[atom + 1])
        _scatter_block(
            diagonal_blocks[atom],
            matrix[start:stop, start:stop],
            slot_indices[atom],
            slot_indices[atom],
        )
    diagonal_mask = np.asarray(block_validity_mask(jnp.asarray(charges)))

    edge_index = _complete_directed_edges(n_atoms)
    n_edges = int(edge_index.shape[1])
    off_diagonal_blocks = np.zeros((n_edges, FULL_ORBITALS, FULL_ORBITALS), dtype=np.float64)
    for edge in range(n_edges):
        row = int(edge_index[0, edge])  # receiver
        col = int(edge_index[1, edge])  # sender
        row_start, row_stop = int(offsets[row]), int(offsets[row + 1])
        col_start, col_stop = int(offsets[col]), int(offsets[col + 1])
        _scatter_block(
            off_diagonal_blocks[edge],
            matrix[row_start:row_stop, col_start:col_stop],
            slot_indices[row],
            slot_indices[col],
        )
    if n_edges:
        off_diagonal_mask = np.asarray(
            block_validity_mask(
                jnp.asarray(charges[edge_index[0]]),
                jnp.asarray(charges[edge_index[1]]),
            )
        )
    else:
        off_diagonal_mask = np.zeros((0, FULL_ORBITALS, FULL_ORBITALS), dtype=bool)

    return diagonal_blocks, diagonal_mask, off_diagonal_blocks, off_diagonal_mask, edge_index


def reconstruct_fock_from_blocks(
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"],
    diagonal_blocks: Float[NDArray[np.float64], "n_atoms 14 14"],
    diagonal_mask: Bool[NDArray[np.bool_], "n_atoms 14 14"],
    off_diagonal_blocks: Float[NDArray[np.float64], "n_edges 14 14"],
    off_diagonal_mask: Bool[NDArray[np.bool_], "n_edges 14 14"],
    edge_index: Int[NDArray[np.int64], "2 n_edges"],
) -> Float[NDArray[np.float64], "n_ao n_ao"]:
    r"""Scatter masked blocks back into the dense spherical Fock matrix.

    Inverse of :func:`cut_fock_to_blocks` (no symmetrisation: each directed edge
    is written exactly once at ``H[block_receiver, block_sender]``, so a complete
    directed graph reconstructs the full matrix). Used to verify the cut + mask +
    ordering round-trip to the original Fock (``1e-10``).

    Args:
        atomic_numbers: Nuclear charges ``(n_atoms,)``.
        diagonal_blocks: Per-atom blocks ``(A, 14, 14)``.
        diagonal_mask: Per-atom validity masks ``(A, 14, 14)``.
        off_diagonal_blocks: Per-edge blocks ``(E, 14, 14)``.
        off_diagonal_mask: Per-edge validity masks ``(E, 14, 14)``.
        edge_index: Directed ``(2, E)`` ``(receiver, sender)`` index.

    Returns:
        The reconstructed dense ``(n_ao, n_ao)`` spherical Fock matrix.
    """
    charges = np.asarray(atomic_numbers).astype(np.int64)
    offsets = _atom_ao_offsets(charges)
    n_ao = int(offsets[-1])
    matrix = np.zeros((n_ao, n_ao), dtype=np.float64)
    slot_indices = [ORBITAL_MASK[int(z)] for z in charges]

    for atom in range(int(charges.shape[0])):
        start, stop = int(offsets[atom]), int(offsets[atom + 1])
        block = diagonal_blocks[atom] * diagonal_mask[atom]
        matrix[start:stop, start:stop] = block[np.ix_(slot_indices[atom], slot_indices[atom])]

    for edge in range(int(edge_index.shape[1])):
        row = int(edge_index[0, edge])
        col = int(edge_index[1, edge])
        row_start, row_stop = int(offsets[row]), int(offsets[row + 1])
        col_start, col_stop = int(offsets[col]), int(offsets[col + 1])
        block = off_diagonal_blocks[edge] * off_diagonal_mask[edge]
        matrix[row_start:row_stop, col_start:col_stop] = block[
            np.ix_(slot_indices[row], slot_indices[col])
        ]
    return matrix


__all__ = [
    "cut_fock_to_blocks",
    "reconstruct_fock_from_blocks",
]
