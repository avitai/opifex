r"""STREAMING block-form QH9 data path for out-of-core Fock-block training.

The eager block loader
(:func:`~opifex.data.sources.qh9_blocks.create_qh9_block_loader`) decodes *every*
molecule's full def2-SVP Fock matrix into host RAM and materialises *every* padded
block batch up front -- about 9 GB of Focks plus 17 GB+ of padded batches for the
full QH9-Stable set (130 831 molecules; Yu et al. 2023, "QH9", arXiv:2306.04922),
which OOMs a 60 GB host. This module is the **streaming** alternative: it reads one
molecule at a time *by id* from the SQLite database, decodes + cuts it to QHNet
block form on the fly, accumulates ``batch_size`` cut molecules, and yields one
padded-concat batch dict -- so resident memory is one batch plus the split id lists
(~1 MB for 130k ints), never the whole dataset. The emitted batch-dict schema is
*byte-for-byte identical* to the eager loader's, so the training driver is
unchanged when it switches eager -> streaming.

Streaming mechanism (datarax ``StreamingSourceBase`` + lazy id-indexed SQLite read)
-----------------------------------------------------------------------------------
:class:`QH9BlockStreamSource` is a concrete
:class:`datarax.sources._source_base.StreamingSourceBase` (the streaming base whose
``__iter__`` resets the backend iterator and bumps the epoch counter and whose
``__next__`` yields the next per-record dict; mirrored from
:class:`datarax.sources.tfds_source.TFDSStreamingSource`). Each *record* is one
molecule's cut block arrays (no leading batch axis). Per epoch the source optionally
shuffles its split's id list with a seeded permutation, then streams each molecule
with a single indexed ``SELECT * FROM data WHERE id = ?`` over a read-only
(``file:...?mode=ro``) connection -- ``id`` is the QH9-Stable primary key, so the
read is index-seeked, one row resident at a time. The thin
:class:`QH9BlockStreamLoader` batching iterable groups ``batch_size`` records and
applies :func:`~opifex.data.sources.qh9_blocks.collate_block_batch` to produce the
padded batch dict.

Reuse (DRY)
-----------
This module re-implements *nothing* of the cut/collate/decode/split. It composes:

* :func:`~opifex.data.sources.qh9_source._decode_row` -- decode one ``(id, N, Z,
  pos, Ham)`` row into a :class:`~opifex.data.sources.qh9_source.QH9Example`
  (def2-SVP convention transform + Bohr positions);
* :func:`~opifex.data.sources.qh9_source.qh9_random_split` -- the reference
  ``0.8 / 0.1 / 0.1`` random split (seed 43) used to partition the id list;
* :func:`~opifex.data.sources.qh9_blocks._cut_example` /
  :func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks` -- the per-molecule
  QHNet block cut;
* :func:`~opifex.data.sources.qh9_blocks.collate_block_batch` and
  :class:`~opifex.data.sources.qh9_blocks.BlockBatchConfig` -- the padded-concat
  collation and its shape/validation contract.

Bounded scope
-------------
*id-permutation, not a shuffle buffer.* Because the cheap ``id`` column is read in
full up front, a seeded permutation of the split's id list gives perfect per-epoch
coverage (every molecule exactly once, no drops/dupes) at ~1 MB cost -- strictly
better than a TF-style fixed shuffle buffer, which only approximates a shuffle and
risks repeats/drops at buffer boundaries. *Per-id SELECT, not a chunked cursor.* One
indexed ``WHERE id = ?`` per molecule keeps exactly one row resident and is simplest;
the primary-key index makes it O(log n) per row. No download happens at import time;
an existing ``QH9Stable.db`` is read.
"""

from __future__ import annotations

import logging
import resource
import sqlite3
from collections.abc import Iterator  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.sources._source_base import StreamingSourceBase
from flax import nnx
from jaxtyping import Array  # noqa: TC002
from numpy.typing import NDArray

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.data.sources.qh9_blocks import (
    _cut_example,
    BlockBatchConfig,
    collate_block_batch,
    reconstruct_fock_from_blocks,
)
from opifex.data.sources.qh9_source import (
    _decode_row,
    qh9_random_split,
    QH9Example,
)


logger = logging.getLogger(__name__)

# Default location of the (externally downloaded) QH9-Stable database, mirroring
# :mod:`opifex.data.sources.qh9_blocks`.
_DEFAULT_DATABASE: Path = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")

# One streamed molecule's cut block arrays, kept as host NumPy arrays (the
# streaming contract: nothing is moved to device until the batch collate).
_RecordDict = dict[str, NDArray]


@dataclass(frozen=True)
class QH9BlockStreamConfig(StructuralConfig):
    """Streaming-loader configuration (a datarax :class:`StructuralConfig`).

    Inherits the datarax module-config contract (``cacheable``/``stochastic``/...)
    required by :class:`StreamingSourceBase`, and carries the block-form
    padded-concat shape/batch fields mirroring
    :class:`~opifex.data.sources.qh9_blocks.BlockBatchConfig` (exposed verbatim via
    :attr:`block_config` for the reused collate).

    Attributes:
        max_atoms: Fixed total atom count a collated batch is padded to.
        max_edges: Fixed total directed-edge count a collated batch is padded to.
        batch_size: Number of molecules concatenated into one padded batch.
        seed: Seed reproducing the QH9-Stable random split (43) and the per-epoch
            stream-shuffle permutation.
        shuffle: If ``True``, the split's id list is permuted with a seeded
            per-epoch permutation before streaming (perfect-coverage shuffle).
            ``False`` streams in ascending-id (split) order -- the natural choice
            for the val/test splits.
    """

    max_atoms: int = 1
    max_edges: int = 0
    batch_size: int = 8
    seed: int = 43
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Validate the streaming + block-shape configuration, failing fast."""
        super().__post_init__()
        # Reuse the eager block config's validation (DRY) -- raises on bad shapes.
        BlockBatchConfig(
            max_atoms=self.max_atoms,
            max_edges=self.max_edges,
            batch_size=self.batch_size,
            seed=self.seed,
        )

    @property
    def block_config(self) -> BlockBatchConfig:
        """The eager-loader :class:`BlockBatchConfig` for the reused collate."""
        return BlockBatchConfig(
            max_atoms=self.max_atoms,
            max_edges=self.max_edges,
            batch_size=self.batch_size,
            seed=self.seed,
        )


@dataclass(frozen=True)
class QH9BlockStreamLoaders:
    """Train/val/test streaming iterables of block-form padded-concat batch dicts.

    Each split is a :class:`QH9BlockStreamLoader` -- a re-iterable that, on every
    ``iter()``, streams its molecules by id and yields padded batch dicts whose
    schema matches the eager :class:`~opifex.data.sources.qh9_blocks.QH9BlockLoaders`.

    Attributes:
        train: Training-split streaming loader (shuffled per ``QH9BlockStreamConfig``).
        val: Validation-split streaming loader (sequential).
        test: Test-split streaming loader (sequential).
        units: Mapping of quantity name to its physical unit string.
    """

    train: QH9BlockStreamLoader
    val: QH9BlockStreamLoader
    test: QH9BlockStreamLoader
    units: dict[str, str]


# =============================================================================
# Lazy id-indexed SQLite read (one molecule at a time)
# =============================================================================


def _read_split_ids(db_path: Path, *, limit: int | None = None) -> tuple[int, ...]:
    """Read only the ``id`` column from the QH9-Stable ``data`` table.

    This is the cheap pre-pass: it touches no ``Z``/``pos``/``Ham`` blob, so the
    full 130k id list costs ~1 MB regardless of Fock size.

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
    logger.info("QH9 stream: read %d molecule ids from %s", len(ids), db_path)
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
        The decoded molecule with its def2-SVP spherical Fock target.

    Raises:
        KeyError: If no row with ``molecule_id`` exists.
    """
    cursor = connection.execute("SELECT * FROM data WHERE id = ?", (molecule_id,))
    row = cursor.fetchone()
    if row is None:
        raise KeyError(f"QH9-Stable molecule id {molecule_id} not found.")
    molecule_identifier, num_nodes, atoms_blob, pos_blob, ham_blob = row
    return _decode_row(molecule_identifier, num_nodes, atoms_blob, pos_blob, ham_blob)


def _example_to_record(example: QH9Example) -> _RecordDict:
    """Cut one decoded molecule to block form and pack it as a per-record dict."""
    cut = _cut_example(example)
    return {
        "atomic_numbers": cut.atomic_numbers,
        "positions": cut.positions,
        "diagonal_blocks": cut.diagonal_blocks,
        "diagonal_mask": cut.diagonal_mask,
        "off_diagonal_blocks": cut.off_diagonal_blocks,
        "off_diagonal_mask": cut.off_diagonal_mask,
        "edge_index": cut.edge_index,
    }


# =============================================================================
# Per-record streaming source (datarax StreamingSourceBase)
# =============================================================================


class QH9BlockStreamSource(StreamingSourceBase):
    """Per-record streaming source over one QH9 split, yielding cut molecules.

    A concrete :class:`datarax.sources._source_base.StreamingSourceBase`: it holds
    only the cheap split id list (never the decoded examples) and, per epoch,
    streams each molecule with an indexed ``SELECT * FROM data WHERE id = ?`` and
    yields its cut block arrays as one record dict. Memory is one molecule at a
    time. ``__iter__`` permutes the id list (when ``shuffle``), bumps the epoch and
    opens a fresh read-only cursor; ``__next__`` reads + cuts the next id.

    Attributes:
        split_ids: The molecule ids of this split (the only per-split state held).
    """

    config: QH9BlockStreamConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        config: QH9BlockStreamConfig,
        db_path: Path,
        split_ids: tuple[int, ...],
        split_name: str | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise the per-record stream from a split's id list.

        Args:
            config: Streaming-loader configuration (shape, batch, seed, shuffle).
            db_path: Path to ``QH9Stable.db`` (read-only).
            split_ids: The molecule ids this source streams (cheap; ~1 MB for 130k).
            split_name: Optional split label (``"train"``/``"val"``/``"test"``).
            rngs: Optional NNX RNGs (unused; the per-epoch shuffle is seeded by
                ``config.seed`` for reproducibility independent of NNX state).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self._db_path = nnx.static(db_path)
        self.split_ids = nnx.static(tuple(int(i) for i in split_ids))
        self.dataset_name = "QH9Stable"
        self.split_name = split_name
        self._length = nnx.static(len(self.split_ids))
        self.length = self._length
        self._is_random_order = config.shuffle
        self._dataset_info = nnx.static({"n_molecules": self._length})
        self.epoch = nnx.Variable(0)
        self._iterator: Iterator[_RecordDict] | None = None

    def __len__(self) -> int:
        """Number of molecules in this split."""
        return self._length

    def _epoch_order(self) -> tuple[int, ...]:
        """Return this epoch's id order: seeded permutation if shuffling, else ascending."""
        if not self._is_random_order:
            return self.split_ids
        # Per-epoch permutation: seed advances with the epoch so each pass differs
        # while staying reproducible from ``config.seed``.
        epoch = int(self.epoch.value)
        rng = np.random.default_rng(self.config.seed + epoch)
        permuted = np.asarray(self.split_ids)[rng.permutation(self._length)]
        return tuple(int(i) for i in permuted)

    def _stream_records(self) -> Iterator[_RecordDict]:
        """Yield one cut-molecule record per id over a fresh read-only cursor."""
        order = self._epoch_order()
        with sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True) as connection:
            for molecule_id in order:
                yield _example_to_record(_read_one_example(connection, molecule_id))

    def __iter__(self) -> Iterator[_RecordDict]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Begin an epoch: bump the counter and open a fresh record stream."""
        self.epoch.value = int(self.epoch.value) + 1
        self._iterator = self._stream_records()
        return self

    def __next__(self) -> _RecordDict:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the next molecule's cut record dict.

        Raises:
            StopIteration: When the split's ids are exhausted.
        """
        iterator = self._iterator
        if iterator is None:
            iterator = self._stream_records()
            self._iterator = iterator
        return next(iterator)


# =============================================================================
# Batching iterable (per-record stream -> padded-concat batch dicts)
# =============================================================================


@dataclass(frozen=True)
class QH9BlockStreamLoader:
    """Re-iterable streaming loader: groups ``batch_size`` records and collates.

    Iterating yields padded-concat batch dicts identical in schema to the eager
    :func:`~opifex.data.sources.qh9_blocks.collate_block_batch` output. A fresh
    :class:`QH9BlockStreamSource` epoch is started on every ``iter()`` so the loader
    can be consumed once per training epoch (re-shuffling when ``shuffle=True``).

    Attributes:
        source: The per-record streaming source for this split.
        config: The streaming-loader configuration (provides ``batch_size`` and the
            collate shape contract).
    """

    source: QH9BlockStreamSource
    config: QH9BlockStreamConfig

    @property
    def split_ids(self) -> tuple[int, ...]:
        """The molecule ids of this split (ascending)."""
        return self.source.split_ids

    def __len__(self) -> int:
        """Number of padded batches per epoch (ceil of molecules / batch_size)."""
        n = len(self.source)
        return (n + self.config.batch_size - 1) // self.config.batch_size

    def __iter__(self) -> Iterator[dict[str, Array]]:
        """Stream molecules, accumulate ``batch_size``, and yield padded batches.

        Memory at any instant is one batch's worth of decoded/cut molecules plus
        the id list; the trailing partial batch (if any) is collated and yielded.
        """
        block_config = self.config.block_config
        chunk: list[QH9Example] = []
        for record in self.source:
            chunk.append(_record_to_example(record))
            if len(chunk) == self.config.batch_size:
                yield collate_block_batch(chunk, block_config)
                chunk = []
        if chunk:
            yield collate_block_batch(chunk, block_config)


def _record_to_example(record: _RecordDict) -> QH9Example:
    """Rebuild a minimal :class:`QH9Example` so the reused collate can re-cut it.

    :func:`~opifex.data.sources.qh9_blocks.collate_block_batch` consumes
    :class:`QH9Example` and re-cuts the Fock internally; to feed it from a streamed
    *cut* record without re-implementing the collate, the record's blocks are
    scattered back into a dense Fock via
    :func:`~opifex.data.sources.qh9_blocks.reconstruct_fock_from_blocks` and wrapped
    as a :class:`QH9Example`. This keeps a single source of truth for the
    padded-concat packing (DRY) at the cost of one re-cut per batch.
    """
    atomic_numbers = np.asarray(record["atomic_numbers"], dtype=np.int32)
    fock = reconstruct_fock_from_blocks(
        atomic_numbers,
        np.asarray(record["diagonal_blocks"], dtype=np.float64),
        np.asarray(record["diagonal_mask"], dtype=bool),
        np.asarray(record["off_diagonal_blocks"], dtype=np.float64),
        np.asarray(record["off_diagonal_mask"], dtype=bool),
        np.asarray(record["edge_index"], dtype=np.int64),
    )
    positions = np.asarray(record["positions"], dtype=np.float64)
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(atomic_numbers, dtype=jnp.int32),
        positions=jnp.asarray(positions, dtype=jnp.float64),
        charge=0,
        multiplicity=1,
        basis_set="def2-svp",
    )
    return QH9Example(
        molecule_id=-1,
        system=system,
        fock=fock,
        atomic_numbers=atomic_numbers,
    )


# =============================================================================
# Loader factory
# =============================================================================


def create_qh9_block_stream_loader(
    *,
    config: QH9BlockStreamConfig,
    db_path: Path | None = None,
    limit: int | None = None,
) -> QH9BlockStreamLoaders:
    """Create streaming train/val/test block-form loaders for QH9-Stable.

    Reads only the ``id`` column to partition the reference
    :func:`~opifex.data.sources.qh9_source.qh9_random_split` (``0.8 / 0.1 / 0.1``,
    seed 43), then wraps each split in a :class:`QH9BlockStreamLoader` that streams
    its molecules by id one at a time and collates ``config.batch_size`` of them per
    padded batch. No Fock is decoded until iteration; resident memory is one batch
    plus the id lists, so the full 130 831-molecule set trains without OOM.

    Args:
        config: Streaming-loader configuration (shape, batch, seed, shuffle).
        db_path: Explicit database path, overriding the default
            ``/mnt/ssd2/Data/qh9/raw/QH9Stable.db``.
        limit: Optional cap on the number of molecules considered (ascending ``id``;
            for quick smoke-tests). ``None`` uses the full dataset.

    Returns:
        A :class:`QH9BlockStreamLoaders` bundle of streaming train/val/test loaders.

    Raises:
        FileNotFoundError: If the resolved database path does not exist.
    """
    resolved_path = db_path if db_path is not None else _DEFAULT_DATABASE
    ids = _read_split_ids(resolved_path, limit=limit)
    train_idx, val_idx, test_idx = qh9_random_split(len(ids), seed=config.seed)
    id_array = np.asarray(ids)

    def _loader(indices: np.ndarray, split_name: str, *, shuffle: bool) -> QH9BlockStreamLoader:
        split_ids = tuple(int(i) for i in np.sort(id_array[indices]))
        split_config = config if shuffle == config.shuffle else _with_shuffle(config, shuffle)
        source = QH9BlockStreamSource(
            config=split_config,
            db_path=resolved_path,
            split_ids=split_ids,
            split_name=split_name,
        )
        return QH9BlockStreamLoader(source=source, config=split_config)

    logger.info(
        "QH9 block stream loader: %d train / %d val / %d test molecules",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return QH9BlockStreamLoaders(
        train=_loader(train_idx, "train", shuffle=config.shuffle),
        val=_loader(val_idx, "val", shuffle=False),
        test=_loader(test_idx, "test", shuffle=False),
        units={"positions": "Bohr", "fock": "Hartree"},
    )


def _with_shuffle(config: QH9BlockStreamConfig, shuffle: bool) -> QH9BlockStreamConfig:
    """Return a copy of ``config`` with ``shuffle`` overridden (val/test stay ordered)."""
    return QH9BlockStreamConfig(
        max_atoms=config.max_atoms,
        max_edges=config.max_edges,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=shuffle,
        cacheable=config.cacheable,
    )


def stream_block_batches(
    loader: QH9BlockStreamLoader,
) -> Iterator[dict[str, Array]]:
    """Yield every padded batch of one streaming split (convenience helper)."""
    yield from loader


def read_real_data_stream_rss(
    *,
    config: QH9BlockStreamConfig,
    db_path: Path,
    max_batches: int,
) -> tuple[int, float]:
    """Stream up to ``max_batches`` real batches and report molecules + peak RSS (MB).

    A real-data smoke driver (not a unit test): drives the streaming train loader
    over the *full* split and tracks the host resident set size, proving resident
    memory stays bounded while the full 130k is iterated.

    Args:
        config: Streaming-loader configuration (full split; ``limit=None``).
        db_path: Path to the real ``QH9Stable.db``.
        max_batches: Number of streamed batches to iterate before stopping.

    Returns:
        ``(molecules_seen, peak_rss_mb)``.
    """
    loaders = create_qh9_block_stream_loader(config=config, db_path=db_path)
    molecules_seen = 0
    peak_rss_mb = 0.0
    for batch_index, batch in enumerate(loaders.train):
        molecules_seen += int(np.asarray(batch["node_pad_mask"]).sum())
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = max(peak_rss_mb, rss_kb / 1024.0)
        if batch_index + 1 >= max_batches:
            break
    return molecules_seen, peak_rss_mb


__all__ = [
    "QH9BlockStreamConfig",
    "QH9BlockStreamLoader",
    "QH9BlockStreamLoaders",
    "QH9BlockStreamSource",
    "create_qh9_block_stream_loader",
    "read_real_data_stream_rss",
    "stream_block_batches",
]
