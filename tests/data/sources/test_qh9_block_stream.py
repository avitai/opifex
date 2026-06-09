r"""Tests for the STREAMING block-form QH9 data path.

The streaming loader (:mod:`opifex.data.sources.qh9_block_stream`) reads one
molecule at a time *by id* from the QH9-Stable SQLite database, decodes + cuts it
to QHNet block form on the fly (reusing
:func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks` and
:func:`~opifex.data.sources.qh9_blocks.collate_block_batch`), and accumulates
``batch_size`` molecules into one padded-concat batch dict -- so host memory is
one batch, never the whole dataset. This is the out-of-core path that makes the
full 130 831-molecule QH9 trainable (the eager
:func:`~opifex.data.sources.qh9_blocks.create_qh9_block_loader` materialises every
molecule's Fock and every padded batch up front and OOMs at 130k).

These tests build a TINY SYNTHETIC sqlite fixture (NOT real QH9 data -- random
*symmetric* Fock blobs of the correct QH9-native def2-SVP size, mirroring the real
``(id, N, Z, pos, Ham)`` row schema, exactly as
:mod:`tests.data.sources.test_qh9_blocks` does) and assert:

* (a) the streamed batch dicts have the SAME keys/shapes as the eager loader;
* (b) the union of streamed molecules over one epoch equals the split's id set
  (no drops/dupes), and the order shuffles with the seed;
* (c) a streamed batch's blocks match the eager loader's for the same molecules
  (correctness of the cut + collate on the streamed path);
* (d) MEMORY: the source holds only the id lists + at most one batch worth of
  decoded examples -- it never materialises every example at once (lazy read).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from opifex.data.sources.qh9_block_stream import (
    create_qh9_block_stream_loader,
    QH9BlockStreamConfig,
    QH9BlockStreamSource,
)
from opifex.data.sources.qh9_blocks import (
    BlockBatchConfig,
    collate_block_batch,
    create_qh9_block_loader,
)
from opifex.data.sources.qh9_source import read_qh9_sqlite
from opifex.neural.quantum.hamiltonian._orbital_layout import FULL_ORBITALS


_REAL_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")

_BATCH_KEYS = frozenset(
    {
        "atomic_numbers",
        "positions",
        "edge_index",
        "node_batch",
        "edge_batch",
        "diagonal_blocks",
        "diagonal_mask",
        "off_diagonal_blocks",
        "off_diagonal_mask",
        "node_pad_mask",
        "edge_pad_mask",
    }
)

# QH9 element compositions exercised below (varied sizes -> ragged batches).
_SPECS: tuple[tuple[np.ndarray, int], ...] = (
    (np.array([8, 1, 1], dtype=np.int32), 1),  # H2O (3 atoms)
    (np.array([1, 6, 7, 8], dtype=np.int32), 2),  # HCNO (4 atoms)
    (np.array([1, 1], dtype=np.int32), 3),  # H2 (2 atoms)
    (np.array([8, 1, 1], dtype=np.int32), 4),  # H2O (3 atoms)
    (np.array([6, 6, 1, 1, 1, 1], dtype=np.int32), 5),  # C2H4-ish (6 atoms)
    (np.array([7, 1, 1, 1], dtype=np.int32), 6),  # NH3 (4 atoms)
    (np.array([9, 1], dtype=np.int32), 7),  # HF (2 atoms)
    (np.array([6, 7, 8, 9, 1], dtype=np.int32), 8),  # mixed (5 atoms)
)


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count for an atom array (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _random_symmetric(n: int, seed: int) -> np.ndarray:
    """Random symmetric float64 matrix (a stand-in QH9-native Fock blob)."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n)).astype(np.float64)
    return matrix + matrix.T


def _make_row(molecule_id: int, atoms: np.ndarray, seed: int) -> tuple:
    """Build one synthetic QH9 row ``(id, N, Z, pos, Ham)`` for ``atoms``."""
    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
    return (
        molecule_id,
        len(atoms),
        atoms.tobytes(),
        positions.tobytes(),
        _random_symmetric(_native_ao(atoms), seed=seed).tobytes(),
    )


@pytest.fixture
def synthetic_qh9_db(tmp_path: Path) -> Path:
    """Write a tiny SYNTHETIC QH9-Stable sqlite db (8 varied molecules) to tmp_path.

    NOT real QH9 data -- random symmetric Fock blobs of the correct QH9-native
    def2-SVP size, mirroring the reference ``(id, N, Z, pos, Ham)`` row schema with
    ``id`` as the indexed primary key (so the streaming per-id ``SELECT`` is fast).
    """
    db_path = tmp_path / "QH9Stable.db"
    rows = [_make_row(index, atoms, seed) for index, (atoms, seed) in enumerate(_SPECS)]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE data (id INTEGER PRIMARY KEY, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)"
        )
        connection.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?)", rows)
        connection.commit()
    return db_path


def _max_shapes() -> tuple[int, int]:
    """Padded ``(max_atoms, max_edges)`` large enough for any batch of the fixture."""
    total_atoms = sum(len(atoms) for atoms, _ in _SPECS)
    total_edges = sum(len(atoms) * (len(atoms) - 1) for atoms, _ in _SPECS)
    return total_atoms, total_edges


# =============================================================================
# (a) Batch-dict schema parity with the eager loader
# =============================================================================


def test_stream_batch_dict_keys_and_shapes_match_eager(synthetic_qh9_db: Path) -> None:
    """A streamed batch dict has the SAME keys + fixed padded shapes as eager."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=3)
    loaders = create_qh9_block_stream_loader(config=config, db_path=synthetic_qh9_db)

    batch = next(iter(loaders.train))
    assert set(batch) == set(_BATCH_KEYS)
    assert batch["atomic_numbers"].shape == (max_atoms,)
    assert batch["positions"].shape == (max_atoms, 3)
    assert batch["edge_index"].shape == (2, max_edges)
    assert batch["diagonal_blocks"].shape == (max_atoms, FULL_ORBITALS, FULL_ORBITALS)
    assert batch["off_diagonal_blocks"].shape == (max_edges, FULL_ORBITALS, FULL_ORBITALS)


# =============================================================================
# (b) Coverage: union of streamed molecules == split id set; order shuffles
# =============================================================================


def _streamed_molecule_ids(loaders_split, db_path: Path) -> list[int]:
    """Recover the source-row ids of every streamed molecule, in stream order.

    Identifies a molecule by its (n_atoms, atomic-number tuple, position hash),
    matching it back to the decoded fixture rows (the synthetic compositions are
    distinct enough, and positions are unique per id, so the map is 1:1).
    """
    decoded = read_qh9_sqlite(db_path)
    fingerprint_to_id = {
        (
            ex.n_atoms,
            tuple(int(z) for z in ex.atomic_numbers),
            float(np.asarray(ex.system.positions).sum()),
        ): ex.molecule_id
        for ex in decoded
    }
    ids: list[int] = []
    for batch in loaders_split:
        node_batch = np.asarray(batch["node_batch"])
        node_pad = np.asarray(batch["node_pad_mask"])
        atoms = np.asarray(batch["atomic_numbers"])
        positions = np.asarray(batch["positions"])
        for mol_id in sorted({int(m) for m in node_batch[node_pad]}):
            sel = (node_batch == mol_id) & node_pad
            fingerprint = (
                int(sel.sum()),
                tuple(int(z) for z in atoms[sel]),
                float(positions[sel].sum()),
            )
            ids.append(fingerprint_to_id[fingerprint])
    return ids


def test_stream_covers_train_split_exactly_once(synthetic_qh9_db: Path) -> None:
    """One epoch streams every train-split molecule exactly once (no drops/dupes)."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(
        max_atoms=max_atoms, max_edges=max_edges, batch_size=2, shuffle=False
    )
    loaders = create_qh9_block_stream_loader(config=config, db_path=synthetic_qh9_db)
    streamed = _streamed_molecule_ids(loaders.train, synthetic_qh9_db)
    assert sorted(streamed) == sorted(loaders.train.split_ids)
    assert len(streamed) == len(set(streamed))


def test_stream_order_shuffles_with_seed(synthetic_qh9_db: Path) -> None:
    """Shuffled streaming reorders the SAME id set vs. the ascending (unshuffled) order.

    The split seed is held fixed (so the train partition is identical); only
    ``shuffle`` flips, isolating the per-epoch stream permutation from the split.
    """
    max_atoms, max_edges = _max_shapes()

    shuffled = create_qh9_block_stream_loader(
        config=QH9BlockStreamConfig(
            max_atoms=max_atoms, max_edges=max_edges, batch_size=1, seed=43, shuffle=True
        ),
        db_path=synthetic_qh9_db,
    )
    ordered = create_qh9_block_stream_loader(
        config=QH9BlockStreamConfig(
            max_atoms=max_atoms, max_edges=max_edges, batch_size=1, seed=43, shuffle=False
        ),
        db_path=synthetic_qh9_db,
    )
    order_shuffled = _streamed_molecule_ids(shuffled.train, synthetic_qh9_db)
    order_ascending = _streamed_molecule_ids(ordered.train, synthetic_qh9_db)

    assert sorted(order_shuffled) == sorted(order_ascending)  # same coverage
    assert order_ascending == sorted(order_ascending)  # unshuffled == ascending id
    assert order_shuffled != order_ascending  # shuffle reorders


def test_stream_two_epochs_reshuffle(synthetic_qh9_db: Path) -> None:
    """Iterating the same train split twice reshuffles (per-epoch permutation)."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(
        max_atoms=max_atoms, max_edges=max_edges, batch_size=1, shuffle=True, seed=7
    )
    loaders = create_qh9_block_stream_loader(config=config, db_path=synthetic_qh9_db)
    epoch_one = _streamed_molecule_ids(loaders.train, synthetic_qh9_db)
    epoch_two = _streamed_molecule_ids(loaders.train, synthetic_qh9_db)
    assert sorted(epoch_one) == sorted(epoch_two)
    assert epoch_one != epoch_two


# =============================================================================
# (c) Correctness: streamed blocks match the eager loader for the same molecules
# =============================================================================


def test_stream_blocks_match_eager_for_same_molecules(synthetic_qh9_db: Path) -> None:
    """A streamed batch equals the eager collate of the same molecules' examples."""
    max_atoms, max_edges = _max_shapes()
    block_config = BlockBatchConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=3)
    stream_config = QH9BlockStreamConfig(
        max_atoms=max_atoms, max_edges=max_edges, batch_size=3, shuffle=False
    )
    loaders = create_qh9_block_stream_loader(config=stream_config, db_path=synthetic_qh9_db)

    # Recover the train-split examples in split (id) order to mirror the stream.
    decoded = {ex.molecule_id: ex for ex in read_qh9_sqlite(synthetic_qh9_db)}
    split_ids = list(loaders.train.split_ids)

    streamed_batches = list(loaders.train)
    for chunk_start, batch in enumerate(streamed_batches):
        chunk_ids = split_ids[chunk_start * 3 : chunk_start * 3 + 3]
        eager = collate_block_batch([decoded[i] for i in chunk_ids], block_config)
        for key in _BATCH_KEYS:
            np.testing.assert_allclose(
                np.asarray(batch[key]),
                np.asarray(eager[key]),
                rtol=0,
                atol=0,
                err_msg=f"streamed {key} != eager {key}",
            )


def test_stream_matches_eager_loader_overall(synthetic_qh9_db: Path) -> None:
    """Streamed + eager loaders cover the same molecules with identical block sums."""
    max_atoms, max_edges = _max_shapes()
    block_config = BlockBatchConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=2)
    stream_config = QH9BlockStreamConfig(
        max_atoms=max_atoms, max_edges=max_edges, batch_size=2, shuffle=False
    )
    eager = create_qh9_block_loader(config=block_config, db_path=synthetic_qh9_db)
    stream = create_qh9_block_stream_loader(config=stream_config, db_path=synthetic_qh9_db)

    def _real_block_sum(split) -> float:
        total = 0.0
        for batch in split:
            mask = np.asarray(batch["diagonal_mask"])
            total += float(np.asarray(batch["diagonal_blocks"])[mask].sum())
        return total

    assert _real_block_sum(eager.train) == pytest.approx(_real_block_sum(stream.train), abs=1e-4)


# =============================================================================
# (d) Memory: the source reads lazily (id lists + at most one batch)
# =============================================================================


def test_source_holds_only_id_list_not_all_examples(synthetic_qh9_db: Path) -> None:
    """The streaming source stores the split id list, NOT every decoded example."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=2)
    source = QH9BlockStreamSource(config=config, db_path=synthetic_qh9_db, split_ids=(0, 1, 2, 3))

    # The source must not have decoded/cached all examples up front.
    assert not hasattr(source, "examples")
    assert not hasattr(source, "data")
    # It knows its split purely from the cheap id list.
    assert tuple(source.split_ids) == (0, 1, 2, 3)
    assert len(source) == 4


def test_per_record_stream_yields_one_molecule_dict_at_a_time(synthetic_qh9_db: Path) -> None:
    """The per-record StreamingSourceBase yields one molecule's cut arrays per next()."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=2)
    source = QH9BlockStreamSource(config=config, db_path=synthetic_qh9_db, split_ids=(0, 1, 2))

    iterator = iter(source)
    first = next(iterator)
    # One record == one molecule: a 1-D atomic-number vector (no leading batch axis).
    assert first["atomic_numbers"].ndim == 1
    assert first["diagonal_blocks"].shape[1:] == (FULL_ORBITALS, FULL_ORBITALS)
    # n_atoms for this record matches one fixture molecule, not the whole split.
    assert first["atomic_numbers"].shape[0] <= max_atoms


def test_split_ids_read_only_id_column(synthetic_qh9_db: Path) -> None:
    """create_qh9_block_stream_loader partitions ids without decoding any Fock."""
    max_atoms, max_edges = _max_shapes()
    config = QH9BlockStreamConfig(max_atoms=max_atoms, max_edges=max_edges, batch_size=2)
    loaders = create_qh9_block_stream_loader(config=config, db_path=synthetic_qh9_db)
    all_ids = (
        set(loaders.train.split_ids) | set(loaders.val.split_ids) | set(loaders.test.split_ids)
    )
    assert all_ids == set(range(len(_SPECS)))
    # 8 molecules under the 0.8/0.1/0.1 split -> 6 train / 0 val / 2 test (trunc).
    assert len(loaders.train.split_ids) == 6


# =============================================================================
# Real-data smoke is a manual script (test_real_data_smoke), not a unit test.
# =============================================================================


@pytest.mark.skipif(not _REAL_DB.exists(), reason="real QH9Stable.db not present")
def test_real_data_stream_first_batches_finite() -> None:
    """A real-data smoke: the first few streamed batches are finite + right-shaped."""
    config = QH9BlockStreamConfig(max_atoms=512, max_edges=8192, batch_size=8, shuffle=True)
    loaders = create_qh9_block_stream_loader(config=config, db_path=_REAL_DB, limit=200)
    seen = 0
    for batch in loaders.train:
        assert batch["diagonal_blocks"].shape == (512, FULL_ORBITALS, FULL_ORBITALS)
        assert np.all(np.isfinite(np.asarray(batch["diagonal_blocks"])))
        seen += 1
        if seen >= 3:
            break
    assert seen >= 1
