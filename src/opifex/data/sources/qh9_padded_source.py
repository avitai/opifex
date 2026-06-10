r"""Out-of-core per-molecule padded QH9-Stable source feeding the GPU Fock operators.

:class:`QH9PaddedSource` is the host-IO half of the GPU-resident QH9 block path.
It reads the QH9-Stable SQLite database with the validated decode/split of
:mod:`opifex.data.sources.qh9_source` and, for **each molecule**, emits a single
element padded to a fixed shape -- but it does **not** decode the Fock to
spherical ordering or cut it into blocks. Those two steps are deferred to the GPU
via the canonical datarax operators in
:mod:`opifex.data.sources.qh9_fock_operators`, so the host side does only integer
index preparation (cheap) and the heavy gather/decode runs fused inside the
jitted train step.

Out-of-core (memory = one batch, not the whole dataset)
-------------------------------------------------------
The source holds only its split's ``id`` list (the QH9-Stable primary key; ~1 MB
for 130k ints) and the database path -- never the decoded Focks. Each
:meth:`get_batch_at` reads ``size`` molecules *by id* with an indexed
``SELECT * FROM data WHERE id = ?`` over a read-only connection, decodes them with
the reused :func:`~opifex.data.sources.qh9_source._decode_row` and pads each into
the fixed-shape element with :func:`_pad_molecule`, so resident memory is one
batch's Focks at a time. Padding to a fixed ``(max_ao, max_ao)`` per molecule
would cost ~1.3 MB/molecule x 130k ~ 170 GB if materialised up front (an OOM on a
full run); reading lazily per batch keeps the host footprint to a single batch.

Per-molecule element (padded to ``max_atoms`` / ``max_edges`` / ``max_ao = 14 *
max_atoms``)
--------------------------------------------------------------------------------
* ``native_fock`` ``(max_ao, max_ao)`` -- the QH9-native Fock, zero-padded.
* ``decode_perm`` ``(max_ao,)`` / ``decode_sign`` ``(max_ao,)`` -- the
  native->spherical AO permutation and signs
  (:func:`~opifex.data.sources.qh9_source.def2svp_decode_indices`); padding AOs
  point at index ``0`` with sign ``1``.
* ``atom_ao_start`` ``(max_atoms,)`` -- per-atom spherical AO block start.
* ``atom_slot_indices`` ``(max_atoms, 14)`` -- inverse ``ORBITAL_MASK``: the
  within-atom source AO offset for each of the 14 block slots (invalid slots map
  to ``0`` and are masked out downstream).
* ``atomic_numbers`` ``(max_atoms,)`` (padded atoms carry ``Z = 0``),
  ``positions`` ``(max_atoms, 3)`` (Bohr), ``edge_index`` ``(2, max_edges)``
  ``(receiver, sender)`` (the complete directed graph; padded edges point at
  atom ``0``), ``node_pad_mask`` ``(max_atoms,)`` and ``edge_pad_mask``
  ``(max_edges,)``.

The fixed per-molecule shape lets :meth:`get_batch_at` stack ``size`` molecules
into a leading batch axis with no ragged collation, and the operators then vmap
over that axis Batch-free.

Bounded scope
-------------
Edges are the complete *directed* graph (no radius cutoff), matching the QHNet
reference cut and the existing :mod:`opifex.data.sources.qh9_blocks` path. The
spherical AO count per atom equals the native AO count (both ``5`` for H/He and
``14`` for C/N/O/F -- the decode only reorders within an atom), so a single
``max_ao = 14 * max_atoms`` bound covers both layouts. No download happens at
import time; an existing ``QH9Stable.db`` is read.
"""

from __future__ import annotations

import logging
import resource
import sqlite3
from collections.abc import Iterator  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from flax import nnx
from jaxtyping import Array  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.data.sources.qh9_blocks import _atom_ao_offsets, _complete_directed_edges
from opifex.data.sources.qh9_source import (
    _decode_row,
    def2svp_decode_indices,
    qh9_random_split,
    QH9Example,
)
from opifex.neural.quantum.hamiltonian._orbital_layout import FULL_ORBITALS, ORBITAL_MASK


logger = logging.getLogger(__name__)

_DEFAULT_DATABASE: Path = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")

_PADDING_ATOMIC_NUMBER: int = 0
"""Atomic number assigned to padded atoms (the embedding table's Z=0 row)."""


@dataclass(frozen=True)
class QH9PaddedConfig(StructuralConfig):
    """Immutable configuration for :class:`QH9PaddedSource`.

    Attributes:
        max_atoms: Fixed per-molecule atom count each element is padded to. Must
            be at least the largest molecule's atom count (QH9-Stable: 29).
        max_edges: Fixed per-molecule directed-edge count. Must be at least
            ``max_atoms * (max_atoms - 1)``.
        seed: Seed reproducing the QH9-Stable reference random split (43) and the
            per-epoch shuffle permutation.
        shuffle: Whether the source iterates molecules in shuffled (per-epoch
            permuted) order.
    """

    max_atoms: int = 32
    max_edges: int = 992
    seed: int = 43
    shuffle: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration, failing fast on bad values."""
        super().__post_init__()
        if self.max_atoms < 1:
            raise ValueError(f"max_atoms must be >= 1, got {self.max_atoms}.")
        if self.max_edges < self.max_atoms * (self.max_atoms - 1):
            raise ValueError(
                f"max_edges {self.max_edges} < max_atoms*(max_atoms-1) "
                f"{self.max_atoms * (self.max_atoms - 1)} (complete directed graph)."
            )

    @property
    def max_ao(self) -> int:
        """Fixed per-molecule spherical/native AO count (``14 * max_atoms``)."""
        return FULL_ORBITALS * self.max_atoms


def _inverse_orbital_mask(atomic_number: int) -> NDArray[np.int64]:
    """Return the ``(14,)`` slot -> within-atom AO offset map for one element.

    For element ``Z`` with valid AO slots ``ORBITAL_MASK[Z]`` (ascending), the
    ``k``-th valid slot holds the ``k``-th packed AO, so the inverse map sends
    that slot to within-atom offset ``k``. Invalid slots map to ``0`` (their
    gathered value is masked out by ``block_validity_mask`` downstream).
    """
    slot_to_offset = np.zeros(FULL_ORBITALS, dtype=np.int64)
    for packed_offset, slot in enumerate(ORBITAL_MASK[int(atomic_number)]):
        slot_to_offset[slot] = packed_offset
    return slot_to_offset


def _pad_molecule(example: QH9Example, config: QH9PaddedConfig) -> dict[str, NDArray]:
    """Build one molecule's fixed-shape padded element (host-side index prep).

    Reuses the validated native-AO offsets
    (:func:`~opifex.data.sources.qh9_blocks._atom_ao_offsets`), the complete
    directed edge set
    (:func:`~opifex.data.sources.qh9_blocks._complete_directed_edges`) and the
    decode permutation/signs
    (:func:`~opifex.data.sources.qh9_source.def2svp_decode_indices`). No Fock
    decode or block cut happens here -- only integer prep and zero-padding.
    """
    charges = np.asarray(example.atomic_numbers).astype(np.int64)
    n_atoms = int(charges.shape[0])
    if n_atoms > config.max_atoms:
        raise ValueError(f"molecule has {n_atoms} atoms > max_atoms {config.max_atoms}.")
    offsets = _atom_ao_offsets(charges)
    n_ao = int(offsets[-1])
    native = np.asarray(example.native_fock, dtype=np.float64)

    max_ao = config.max_ao
    native_padded = np.zeros((max_ao, max_ao), dtype=np.float64)
    native_padded[:n_ao, :n_ao] = native
    indices, signs = def2svp_decode_indices(charges.astype(np.int32))
    decode_perm = np.zeros(max_ao, dtype=np.int64)
    decode_perm[:n_ao] = indices
    decode_sign = np.ones(max_ao, dtype=np.int64)
    decode_sign[:n_ao] = signs

    atom_ao_start = np.zeros(config.max_atoms, dtype=np.int64)
    atom_ao_start[:n_atoms] = offsets[:n_atoms]
    atom_slots = np.zeros((config.max_atoms, FULL_ORBITALS), dtype=np.int64)
    for atom in range(n_atoms):
        atom_slots[atom] = _inverse_orbital_mask(int(charges[atom]))

    atomic_numbers = np.full(config.max_atoms, _PADDING_ATOMIC_NUMBER, dtype=np.int64)
    atomic_numbers[:n_atoms] = charges
    positions = np.zeros((config.max_atoms, 3), dtype=np.float64)
    positions[:n_atoms] = np.asarray(example.system.positions, dtype=np.float64).reshape(n_atoms, 3)

    edges = _complete_directed_edges(n_atoms)
    n_edges = int(edges.shape[1])
    if n_edges > config.max_edges:
        raise ValueError(f"molecule has {n_edges} edges > max_edges {config.max_edges}.")
    edge_index = np.zeros((2, config.max_edges), dtype=np.int64)
    edge_index[:, :n_edges] = edges

    node_pad_mask = np.zeros(config.max_atoms, dtype=bool)
    node_pad_mask[:n_atoms] = True
    edge_pad_mask = np.zeros(config.max_edges, dtype=bool)
    edge_pad_mask[:n_edges] = True

    return {
        "native_fock": native_padded,
        "decode_perm": decode_perm,
        "decode_sign": decode_sign,
        "atom_ao_start": atom_ao_start,
        "atom_slot_indices": atom_slots,
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "edge_index": edge_index,
        "node_pad_mask": node_pad_mask,
        "edge_pad_mask": edge_pad_mask,
    }


def _stack_padded(padded: list[dict[str, NDArray]]) -> dict[str, Array]:
    """Stack per-molecule padded elements into a leading-axis dict of JAX arrays."""
    if not padded:
        raise ValueError("QH9PaddedSource requires at least one molecule per batch.")
    return {key: jnp.asarray(np.stack([item[key] for item in padded], axis=0)) for key in padded[0]}


# =============================================================================
# Lazy id-indexed SQLite read (one molecule at a time)
# =============================================================================


def _read_split_ids(db_path: Path, *, limit: int | None = None) -> tuple[int, ...]:
    """Read only the ``id`` column from the QH9-Stable ``data`` table.

    The cheap pre-pass: it touches no ``Z``/``pos``/``Ham`` blob, so the full
    130k id list costs ~1 MB regardless of Fock size.

    Args:
        db_path: Path to ``QH9Stable.db``.
        limit: Optional cap on the number of ids returned (ascending ``id``).

    Returns:
        The molecule ids in ascending order.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"QH9-Stable database not found: {db_path}")
    query = "SELECT id FROM data ORDER BY id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        ids = [int(row[0]) for row in connection.execute(query)]
    logger.info("QH9 padded source: read %d molecule ids from %s", len(ids), db_path)
    return tuple(ids)


def _read_one_example(connection: sqlite3.Connection, molecule_id: int) -> QH9Example:
    """Lazily read + decode a single molecule by id (one indexed row).

    Issues ``SELECT * FROM data WHERE id = ?`` (``id`` is the primary key, so the
    read is index-seeked) and decodes the row with the reused
    :func:`~opifex.data.sources.qh9_source._decode_row`, keeping exactly one
    molecule's Fock resident.

    Args:
        connection: An open read-only SQLite connection to ``QH9Stable.db``.
        molecule_id: The ``id`` primary key of the molecule to read.

    Returns:
        The decoded molecule with its native + spherical Fock.

    Raises:
        KeyError: If no row with ``molecule_id`` exists.
    """
    cursor = connection.execute("SELECT * FROM data WHERE id = ?", (molecule_id,))
    row = cursor.fetchone()
    if row is None:
        raise KeyError(f"QH9-Stable molecule id {molecule_id} not found.")
    molecule_identifier, num_nodes, atoms_blob, pos_blob, ham_blob = row
    return _decode_row(molecule_identifier, num_nodes, atoms_blob, pos_blob, ham_blob)


class QH9PaddedSource(DataSourceModule):
    """Out-of-core per-molecule padded QH9-Stable source (lazy host IO).

    A concrete :class:`datarax.core.data_source.DataSourceModule` that holds only
    its split's ``id`` list and the database path -- never the decoded Focks.
    :meth:`__len__` is the molecule count; :meth:`__getitem__` reads + pads one
    molecule by id; :meth:`get_batch_at` reads + pads ``size`` molecules and stacks
    them into a leading batch axis. With ``shuffle`` the source maps logical
    positions through a per-epoch seeded permutation of its id list (advance the
    epoch with :meth:`next_epoch`), giving perfect per-epoch coverage at ~1 MB
    cost. The Fock spherical decode and block cut are deferred to the GPU operators
    in :mod:`opifex.data.sources.qh9_fock_operators`.

    Args:
        config: Padding/split configuration.
        db_path: Path to ``QH9Stable.db`` (read-only).
        split_ids: The molecule ids this source streams (cheap; ~1 MB for 130k).
        split_name: Optional split label (``"train"``/``"val"``/``"test"``).
        rngs: Optional NNX RNGs (unused; the per-epoch shuffle is seeded by
            ``config.seed`` for reproducibility independent of NNX state).
        name: Optional module name.
    """

    config: QH9PaddedConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: QH9PaddedConfig,
        *,
        db_path: Path,
        split_ids: tuple[int, ...],
        split_name: str | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise the lazy source from a split's id list (no Fock read)."""
        super().__init__(config, rngs=rngs, name=name or "QH9PaddedSource")
        self._db_path = nnx.static(db_path)
        self.split_ids = nnx.static(tuple(int(i) for i in split_ids))
        self.length = nnx.static(len(self.split_ids))
        self.dataset_name = "QH9Stable"
        self.split_name = split_name
        self.epoch = nnx.Variable(0)
        self._is_random_order = config.shuffle

    def __len__(self) -> int:
        """Number of molecules in this split."""
        return int(self.length)

    @property
    def is_random_order(self) -> bool:
        """Whether iteration order is randomized (per-epoch permuted)."""
        return self._is_random_order

    def next_epoch(self) -> None:
        """Advance the epoch counter so the next shuffle permutation differs."""
        self.epoch.value = int(self.epoch.value) + 1

    def _epoch_order(self) -> NDArray[np.int64]:
        """Return this epoch's id order: seeded permutation if shuffling, else ascending.

        The seed advances with the epoch so each pass differs while staying
        reproducible from ``config.seed`` -- perfect per-epoch coverage (every
        molecule exactly once, no drops/dupes), strictly better than a fixed
        shuffle buffer.
        """
        ids = np.asarray(self.split_ids, dtype=np.int64)
        if not self._is_random_order:
            return ids
        rng = np.random.default_rng(self.config.seed + int(self.epoch.value))
        return ids[rng.permutation(int(self.length))]

    def _ids_for_positions(self, positions: NDArray[np.int64]) -> NDArray[np.int64]:
        """Map logical positions (wrapped) to molecule ids under the epoch order."""
        order = self._epoch_order()
        return order[positions % int(self.length)]

    def _read_padded(self, molecule_ids: NDArray[np.int64]) -> list[dict[str, NDArray]]:
        """Read + pad the given molecule ids over a fresh read-only connection."""
        with sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True) as connection:
            return [
                _pad_molecule(_read_one_example(connection, int(mid)), self.config)
                for mid in molecule_ids
            ]

    def __getitem__(self, index: int) -> dict[str, Array]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Read + pad one molecule by logical index (lazy single-row read)."""
        molecule_id = self._ids_for_positions(np.asarray([int(index)], dtype=np.int64))[0]
        padded = self._read_padded(np.asarray([molecule_id], dtype=np.int64))[0]
        return {key: jnp.asarray(value) for key, value in padded.items()}

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, Array]:
        """Lazily read ``size`` molecules from ``start`` and stack them.

        Reads ``size`` molecules *by id* (one indexed row each) under the current
        epoch's order, pads each into the fixed-shape element and stacks them into
        a leading batch axis. Resident memory is one batch's Focks. ``start`` is a
        concrete Python position (the host-side reader is not JAX-traced); the
        wrap-around (``% length``) fills the final partial batch.

        Args:
            start: Starting logical position (concrete int).
            size: Number of molecules to read and stack.
            key: Unused; the per-epoch shuffle is seeded by ``config.seed`` and
                advanced via :meth:`next_epoch`.

        Returns:
            A padded batch dict with a leading axis of ``size``.
        """
        del key
        positions = int(start) + np.arange(size, dtype=np.int64)
        molecule_ids = self._ids_for_positions(positions)
        return _stack_padded(self._read_padded(molecule_ids))

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Per-element shape/dtype spec (one padded molecule, no leading axis)."""
        max_ao = self.config.max_ao
        max_atoms = self.config.max_atoms
        max_edges = self.config.max_edges
        f64 = jnp.float64
        i64 = jnp.int64
        return {
            "native_fock": jax.ShapeDtypeStruct((max_ao, max_ao), f64),
            "decode_perm": jax.ShapeDtypeStruct((max_ao,), i64),
            "decode_sign": jax.ShapeDtypeStruct((max_ao,), i64),
            "atom_ao_start": jax.ShapeDtypeStruct((max_atoms,), i64),
            "atom_slot_indices": jax.ShapeDtypeStruct((max_atoms, FULL_ORBITALS), i64),
            "atomic_numbers": jax.ShapeDtypeStruct((max_atoms,), i64),
            "positions": jax.ShapeDtypeStruct((max_atoms, 3), f64),
            "edge_index": jax.ShapeDtypeStruct((2, max_edges), i64),
            "node_pad_mask": jax.ShapeDtypeStruct((max_atoms,), jnp.bool_),
            "edge_pad_mask": jax.ShapeDtypeStruct((max_edges,), jnp.bool_),
        }


@dataclass(frozen=True)
class QH9PaddedSplits:
    """Train/val/test :class:`QH9PaddedSource` instances plus physical units.

    Attributes:
        train: Training-split source.
        val: Validation-split source.
        test: Test-split source.
        units: Mapping of quantity name to its physical unit string.
    """

    train: QH9PaddedSource
    val: QH9PaddedSource
    test: QH9PaddedSource
    units: dict[str, str]


def _with_shuffle(config: QH9PaddedConfig, shuffle: bool) -> QH9PaddedConfig:
    """Return a copy of ``config`` with ``shuffle`` overridden (val/test stay ordered)."""
    return QH9PaddedConfig(
        max_atoms=config.max_atoms,
        max_edges=config.max_edges,
        seed=config.seed,
        shuffle=shuffle,
        cacheable=config.cacheable,
    )


def create_qh9_padded_sources(
    *,
    config: QH9PaddedConfig,
    db_path: Path | None = None,
    limit: int | None = None,
    rngs: nnx.Rngs | None = None,
) -> QH9PaddedSplits:
    """Create out-of-core per-molecule padded train/val/test sources for QH9-Stable.

    Reads only the ``id`` column of an existing ``QH9Stable.db`` (no download, no
    Fock decode) to partition the reference
    :func:`~opifex.data.sources.qh9_source.qh9_random_split` (``0.8 / 0.1 / 0.1``,
    seed 43), then builds a lazy :class:`QH9PaddedSource` per split. The train
    split is shuffled per ``config.shuffle``; val/test stream in ascending-id
    order. No Fock is resident until a batch is read.

    Args:
        config: Padding/split configuration.
        db_path: Explicit database path, overriding the default.
        limit: Optional cap on molecules considered (ascending ``id``; quick
            smoke-tests).
        rngs: Optional RNGs forwarded to each source.

    Returns:
        A :class:`QH9PaddedSplits` bundle.

    Raises:
        FileNotFoundError: If the resolved database path does not exist.
    """
    resolved_path = db_path if db_path is not None else _DEFAULT_DATABASE
    ids = _read_split_ids(resolved_path, limit=limit)
    train_idx, val_idx, test_idx = qh9_random_split(len(ids), seed=config.seed)
    id_array = np.asarray(ids, dtype=np.int64)

    def _source(indices: NDArray[np.int64], split_name: str, *, shuffle: bool) -> QH9PaddedSource:
        split_ids = tuple(int(i) for i in np.sort(id_array[indices]))
        split_config = config if shuffle == config.shuffle else _with_shuffle(config, shuffle)
        return QH9PaddedSource(
            split_config,
            db_path=resolved_path,
            split_ids=split_ids,
            split_name=split_name,
            rngs=rngs,
        )

    logger.info(
        "QH9 padded source: %d train / %d val / %d test molecules",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return QH9PaddedSplits(
        train=_source(train_idx, "train", shuffle=config.shuffle),
        val=_source(val_idx, "val", shuffle=False),
        test=_source(test_idx, "test", shuffle=False),
        units={"positions": "Bohr", "fock": "Hartree"},
    )


def iterate_padded_batches(source: QH9PaddedSource, size: int) -> Iterator[dict[str, Array]]:
    """Yield consecutive ``size``-molecule padded batches over one epoch.

    Advances the source's epoch counter once (so a shuffled source re-permutes its
    id order each pass), then drives :meth:`QH9PaddedSource.get_batch_at` with a
    Python position counter covering every molecule once (the final partial batch
    wraps to fill ``size`` -- the wrapped molecules carry valid masks, so a
    downstream masked loss ignores them by tracking the real count).

    Args:
        source: The padded source to iterate.
        size: Number of molecules per batch.

    Yields:
        Padded batch dicts with a leading axis of ``size``.
    """
    source.next_epoch()
    for start in range(0, len(source), size):
        yield source.get_batch_at(start, size)


def read_padded_source_rss(
    *,
    config: QH9PaddedConfig,
    db_path: Path,
    batch_size: int,
    max_batches: int,
) -> tuple[int, float]:
    """Stream up to ``max_batches`` real padded batches and report molecules + peak RSS (MB).

    A real-data smoke driver (not a unit test): drives the lazy train source over
    the *full* split and tracks the host resident set size, proving resident memory
    stays bounded while the full 130k split is iterated.

    Args:
        config: Padded-source configuration (full split; ``limit=None``).
        db_path: Path to the real ``QH9Stable.db``.
        batch_size: Number of molecules per padded batch.
        max_batches: Number of batches to read before stopping.

    Returns:
        ``(molecules_seen, peak_rss_mb)``.
    """
    splits = create_qh9_padded_sources(config=config, db_path=db_path)
    molecules_seen = 0
    peak_rss_mb = 0.0
    for batch_index, batch in enumerate(iterate_padded_batches(splits.train, batch_size)):
        molecules_seen += int(np.asarray(batch["node_pad_mask"]).sum())
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = max(peak_rss_mb, rss_kb / 1024.0)
        if batch_index + 1 >= max_batches:
            break
    return molecules_seen, peak_rss_mb


__all__ = [
    "QH9PaddedConfig",
    "QH9PaddedSource",
    "QH9PaddedSplits",
    "create_qh9_padded_sources",
    "iterate_padded_batches",
    "read_padded_source_rss",
]
