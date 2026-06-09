r"""QH9-Stable block-form data path for the heterogeneous-batch Fock predictor.

The :class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
consumes a *flat concatenated* heterogeneous batch -- many molecules of differing
composition packed into one ``(atomic_numbers, positions, edge_index)`` graph --
and emits a fixed ``(14, 14)`` diagonal Fock block per atom and a ``(14, 14)``
off-diagonal block per directed within-molecule edge. This module produces the
matching *targets*: it cuts each molecule's spherical def2-SVP Fock matrix into
those blocks (QHNet ``cut_matrix``; Yu et al. 2023, "QHNet"/"QH9",
arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/datasets.py``) and collates a set of molecules into one
flat, padded concatenation with the batch/segment structure and masks the loss
needs.

Reuse (DRY)
-----------
The SQLite decode (:func:`~opifex.data.sources.qh9_source.read_qh9_sqlite`), the
PySCF def2-SVP convention transform
(:func:`~opifex.data.sources.qh9_source.matrix_transform_def2svp`, applied inside
the decode) and the reference random split
(:func:`~opifex.data.sources.qh9_source.qh9_random_split`) are reused verbatim --
this module never re-implements them. The 14-slot per-element AO layout and
validity masks come from
:mod:`opifex.neural.quantum.hamiltonian._orbital_layout`
(:data:`~...FULL_ORBITALS`, :data:`~...ORBITAL_MASK`,
:func:`~...block_validity_mask`, :func:`~...atom_orbital_counts`). Because the
per-atom AO order of the spherical Fock matches the 14-slot block layout
(``3 s + 2 p + 1 d`` = ``BLOCK_IRREPS``), cutting atom ``i``'s contiguous AO range
into its element's ``ORBITAL_MASK[Z_i]`` slot indices is exact (verified by the
round-trip test to ``1e-10``).

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

Padded-concat collation (the heterogeneous batch)
-------------------------------------------------
:func:`collate_block_batch` packs ``B`` molecules of different sizes into one
flat batch padded to a fixed ``(max_atoms, max_edges)`` so a given shape compiles
once and is reused. Padded atoms carry ``Z = 0`` (the predictor's embedding table
has a row for it) and ``node_pad_mask = False``; padded edges point at a single
reserved in-range padded atom (NOT ``-1`` -- the predictor gathers node features
with the raw edge index, so a ``-1`` would gather-wrap into real data) and carry
``edge_pad_mask = False``. The emitted dict keys/shapes are documented on
:func:`collate_block_batch`.

Bounded scope
-------------
Edges are the *complete directed graph* (no radius cutoff): QH9 molecules are
small (``n_atoms`` 3..29) and QHNet's reference ``cut_matrix`` itself uses the
complete graph, so a cutoff would drop real off-diagonal Fock blocks the loss
must supervise. Batches pad to a *configurable* ``(max_atoms, max_edges)`` rather
than bucketing by composition: the block form is composition-agnostic (one
``(max_atoms, max_edges)`` shape compiles once for *any* mixture), so a single
padded shape -- not many per-signature buckets -- is the natural unit here. No
download happens at import time; an existing ``QH9Stable.db`` is read.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.data.sources.qh9_source import (
    load_qh9_data,
    QH9Example,
)
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    block_validity_mask,
    FULL_ORBITALS,
    ORBITAL_MASK,
)


logger = logging.getLogger(__name__)

# Default location of the (externally downloaded) QH9-Stable database, mirroring
# :mod:`opifex.data.sources.qh9_source`.
_DEFAULT_DATABASE: Path = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")

_PADDING_ATOMIC_NUMBER: int = 0
"""Atomic number assigned to padded atoms (the embedding table's Z=0 row)."""


@dataclass(frozen=True)
class BlockBatchConfig:
    """Immutable configuration for the QH9 block-form padded-concat loader.

    Attributes:
        max_atoms: Fixed total atom count a collated batch is padded to. Must be
            at least the sum of atom counts of the molecules in any batch.
        max_edges: Fixed total directed-edge count a collated batch is padded to.
            Must be at least the sum of ``n_atoms * (n_atoms - 1)`` over a batch.
        batch_size: Number of molecules concatenated into one batch.
        seed: Seed reproducing the QH9-Stable random split (fixed at 43 in the
            reference; exposed for completeness).
    """

    max_atoms: int
    max_edges: int
    batch_size: int = 8
    seed: int = 43

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on bad values."""
        if self.max_atoms < 1:
            raise ValueError(f"max_atoms must be >= 1, got {self.max_atoms}.")
        if self.max_edges < 0:
            raise ValueError(f"max_edges must be >= 0, got {self.max_edges}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")


@dataclass(frozen=True)
class QH9BlockLoaders:
    """Train/val/test iterables of block-form padded-concat batch dicts.

    Attributes:
        train: Training-split batches (one dict per :func:`collate_block_batch`).
        val: Validation-split batches.
        test: Test-split batches.
        units: Mapping of quantity name to its physical unit string.
    """

    train: tuple[dict[str, Array], ...]
    val: tuple[dict[str, Array], ...]
    test: tuple[dict[str, Array], ...]
    units: dict[str, str]


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


# =============================================================================
# Padded-concat collation (ragged molecules -> one flat padded batch)
# =============================================================================


@dataclass(frozen=True)
class _CutExample:
    """A molecule's cut blocks plus its flat positions/atoms (collation input)."""

    atomic_numbers: NDArray[np.int64]
    positions: NDArray[np.float64]
    diagonal_blocks: NDArray[np.float64]
    diagonal_mask: NDArray[np.bool_]
    off_diagonal_blocks: NDArray[np.float64]
    off_diagonal_mask: NDArray[np.bool_]
    edge_index: NDArray[np.int64]


def _cut_example(example: QH9Example) -> _CutExample:
    """Cut one decoded :class:`QH9Example` into its block-form arrays."""
    diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(
        example.atomic_numbers, example.fock
    )
    positions = np.asarray(example.system.positions, dtype=np.float64).reshape(example.n_atoms, 3)
    return _CutExample(
        atomic_numbers=np.asarray(example.atomic_numbers).astype(np.int64),
        positions=positions,
        diagonal_blocks=diag,
        diagonal_mask=diag_mask,
        off_diagonal_blocks=off,
        off_diagonal_mask=off_mask,
        edge_index=edge_index,
    )


def _empty_batch_arrays(config: BlockBatchConfig) -> dict[str, NDArray]:
    """Allocate the zero-initialised padded batch arrays for one collation."""
    block_shape = (FULL_ORBITALS, FULL_ORBITALS)
    return {
        "atomic_numbers": np.full(config.max_atoms, _PADDING_ATOMIC_NUMBER, dtype=np.int64),
        "positions": np.zeros((config.max_atoms, 3), dtype=np.float64),
        "edge_index": np.zeros((2, config.max_edges), dtype=np.int64),
        "node_batch": np.full(config.max_atoms, -1, dtype=np.int64),
        "edge_batch": np.full(config.max_edges, -1, dtype=np.int64),
        "diagonal_blocks": np.zeros((config.max_atoms, *block_shape), dtype=np.float64),
        "diagonal_mask": np.zeros((config.max_atoms, *block_shape), dtype=bool),
        "off_diagonal_blocks": np.zeros((config.max_edges, *block_shape), dtype=np.float64),
        "off_diagonal_mask": np.zeros((config.max_edges, *block_shape), dtype=bool),
        "node_pad_mask": np.zeros(config.max_atoms, dtype=bool),
        "edge_pad_mask": np.zeros(config.max_edges, dtype=bool),
    }


def collate_block_batch(
    examples: Sequence[QH9Example],
    config: BlockBatchConfig,
) -> dict[str, Array]:
    r"""Collate molecules into one flat batch padded to ``(max_atoms, max_edges)``.

    Each molecule is cut to block form (:func:`cut_fock_to_blocks`) and packed
    into a single flat concatenation with its within-molecule edges offset by the
    running atom count (so an edge never crosses molecules). Padded atoms carry
    ``Z = 0`` and ``node_pad_mask = False``; padded edges point at the first
    padded atom slot (a safe in-range index, never ``-1``) and carry
    ``edge_pad_mask = False`` (see the module docstring's padding contract).

    Args:
        examples: Molecules to concatenate (their total atom/edge counts must fit
            ``config.max_atoms`` / ``config.max_edges``).
        config: Padded-batch shape and split configuration.

    Returns:
        A dict of JAX arrays:
        ``{"atomic_numbers" (A_max,), "positions" (A_max, 3),
        "edge_index" (2, E_max), "node_batch" (A_max,), "edge_batch" (E_max,),
        "diagonal_blocks" (A_max, 14, 14), "diagonal_mask" (A_max, 14, 14),
        "off_diagonal_blocks" (E_max, 14, 14),
        "off_diagonal_mask" (E_max, 14, 14), "node_pad_mask" (A_max,),
        "edge_pad_mask" (E_max,)}``.

    Raises:
        ValueError: If the molecules' total atoms or edges exceed the configured
            ``max_atoms`` / ``max_edges``.
    """
    cut = [_cut_example(example) for example in examples]
    total_atoms = sum(item.atomic_numbers.shape[0] for item in cut)
    total_edges = sum(item.edge_index.shape[1] for item in cut)
    if total_atoms > config.max_atoms:
        raise ValueError(f"batch has {total_atoms} atoms > max_atoms {config.max_atoms}.")
    if total_edges > config.max_edges:
        raise ValueError(f"batch has {total_edges} edges > max_edges {config.max_edges}.")

    batch = _empty_batch_arrays(config)
    # The reserved padded atom that all padded edges point at (in-range, masked).
    padded_atom = total_atoms if total_atoms < config.max_atoms else 0

    atom_cursor = 0
    edge_cursor = 0
    for mol_id, item in enumerate(cut):
        n_atoms = item.atomic_numbers.shape[0]
        n_edges = item.edge_index.shape[1]
        atom_slice = slice(atom_cursor, atom_cursor + n_atoms)
        edge_slice = slice(edge_cursor, edge_cursor + n_edges)

        batch["atomic_numbers"][atom_slice] = item.atomic_numbers
        batch["positions"][atom_slice] = item.positions
        batch["node_batch"][atom_slice] = mol_id
        batch["node_pad_mask"][atom_slice] = True
        batch["diagonal_blocks"][atom_slice] = item.diagonal_blocks
        batch["diagonal_mask"][atom_slice] = item.diagonal_mask

        if n_edges:
            batch["edge_index"][:, edge_slice] = item.edge_index + atom_cursor
            batch["edge_batch"][edge_slice] = mol_id
            batch["edge_pad_mask"][edge_slice] = True
            batch["off_diagonal_blocks"][edge_slice] = item.off_diagonal_blocks
            batch["off_diagonal_mask"][edge_slice] = item.off_diagonal_mask

        atom_cursor += n_atoms
        edge_cursor += n_edges

    # Point every padded edge at the reserved padded atom (safe in-range gather).
    if edge_cursor < config.max_edges:
        batch["edge_index"][:, edge_cursor:] = padded_atom

    return {key: jnp.asarray(value) for key, value in batch.items()}


# =============================================================================
# Loader factory (split -> batched padded-concat iterables)
# =============================================================================


def _batched_split(
    examples: Sequence[QH9Example],
    config: BlockBatchConfig,
) -> tuple[dict[str, Array], ...]:
    """Group a split's molecules into ``batch_size`` chunks and collate each."""
    batches: list[dict[str, Array]] = []
    for start in range(0, len(examples), config.batch_size):
        chunk = examples[start : start + config.batch_size]
        if chunk:
            batches.append(collate_block_batch(chunk, config))
    return tuple(batches)


def create_qh9_block_loader(
    *,
    config: BlockBatchConfig,
    db_path: Path | None = None,
    limit: int | None = None,
) -> QH9BlockLoaders:
    """Create train/val/test block-form padded-concat loaders for QH9-Stable.

    Reads an existing ``QH9Stable.db`` (no download) via the reused
    :func:`~opifex.data.sources.qh9_source.load_qh9_data` (decode + convention
    transform + reference random split), then cuts every molecule to QHNet block
    form and groups each split into ``config.batch_size`` molecules per padded
    concatenation (:func:`collate_block_batch`).

    Args:
        config: Padded-batch shape and split configuration.
        db_path: Explicit database path, overriding the default
            ``/mnt/ssd2/Data/qh9/raw/QH9Stable.db`` (the hook network-free tests
            use).
        limit: Optional cap on decoded rows (quick smoke-tests).

    Returns:
        A :class:`QH9BlockLoaders` bundle of train/val/test batch-dict tuples.

    Raises:
        FileNotFoundError: If the resolved database path does not exist.
    """
    resolved_path = db_path if db_path is not None else _DEFAULT_DATABASE
    data = load_qh9_data(resolved_path, seed=config.seed, limit=limit)
    examples = data.examples
    train = tuple(examples[i] for i in data.train_indices)
    val = tuple(examples[i] for i in data.val_indices)
    test = tuple(examples[i] for i in data.test_indices)
    logger.info(
        "QH9 block loader: %d train / %d val / %d test molecules",
        len(train),
        len(val),
        len(test),
    )
    return QH9BlockLoaders(
        train=_batched_split(train, config),
        val=_batched_split(val, config),
        test=_batched_split(test, config),
        units={"positions": "Bohr", "fock": "Hartree"},
    )


def iterate_block_batches(
    loaders: QH9BlockLoaders,
) -> Iterator[dict[str, Array]]:
    """Yield every batch across train/val/test splits (convenience helper)."""
    yield from loaders.train
    yield from loaders.val
    yield from loaders.test


__all__ = [
    "BlockBatchConfig",
    "QH9BlockLoaders",
    "collate_block_batch",
    "create_qh9_block_loader",
    "cut_fock_to_blocks",
    "iterate_block_batches",
    "reconstruct_fock_from_blocks",
]
