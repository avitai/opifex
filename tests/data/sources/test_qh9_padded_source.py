r"""Tests for the out-of-core per-molecule padded QH9 source.

:class:`~opifex.data.sources.qh9_padded_source.QH9PaddedSource` is a lazy
:class:`datarax.core.data_source.DataSourceModule`: it holds only its split's
``id`` list and reads + pads each molecule from SQLite on demand, stacking
``size`` molecules into a leading batch axis with ``get_batch_at`` (memory = one
batch). These tests build a tiny synthetic QH9 sqlite fixture (random *symmetric*
native Fock blobs of the correct QH9-native def2-SVP size, mirroring the real
``(id, N, Z, pos, Ham)`` schema) and assert the padded shapes, masks, edge set,
the lazy stacking contract and the per-epoch shuffle.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest
from flax import nnx

from opifex.data.sources.qh9_padded_source import (
    create_qh9_padded_sources,
    iterate_padded_batches,
    QH9PaddedConfig,
    QH9PaddedSource,
)
from opifex.data.sources.qh9_source import qh9_random_split, read_qh9_sqlite
from opifex.neural.quantum.hamiltonian._orbital_layout import FULL_ORBITALS


_H2O_ATOMS = np.array([8, 1, 1], dtype=np.int32)
_HCNO_ATOMS = np.array([1, 6, 7, 8], dtype=np.int32)
_H2_ATOMS = np.array([1, 1], dtype=np.int32)


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _make_row(molecule_id: int, atoms: np.ndarray, seed: int) -> tuple:
    """Build one synthetic QH9 row ``(id, N, Z, pos, Ham)``."""
    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
    native = rng.standard_normal((_native_ao(atoms), _native_ao(atoms))).astype(np.float64)
    native = native + native.T
    return (
        molecule_id,
        len(atoms),
        atoms.tobytes(),
        positions.tobytes(),
        native.tobytes(),
    )


def _write_db(db_path: Path, specs: list[tuple[np.ndarray, int]]) -> Path:
    """Write a synthetic QH9-Stable sqlite db from ``(atoms, seed)`` specs."""
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
        connection.executemany(
            "INSERT INTO data VALUES (?, ?, ?, ?, ?)",
            [_make_row(i, atoms, seed) for i, (atoms, seed) in enumerate(specs)],
        )
    return db_path


@pytest.fixture
def synthetic_qh9_db(tmp_path: Path) -> Path:
    """Write a tiny SYNTHETIC QH9-Stable sqlite db (H2O, HCNO, H2 repeats).

    Uses enough molecules that the reference 0.8/0.1/0.1 split leaves every split
    non-empty (a single H2/H2O/HCNO triplet would give an empty val split).
    """
    cycle = (_H2O_ATOMS, _HCNO_ATOMS, _H2_ATOMS)
    specs = [(cycle[i % len(cycle)], i + 1) for i in range(15)]
    return _write_db(tmp_path / "QH9Stable.db", specs)


def _config() -> QH9PaddedConfig:
    """Padding config covering the synthetic compositions."""
    return QH9PaddedConfig(max_atoms=6, max_edges=6 * 5)


def _source(db_path: Path, *, shuffle: bool = False) -> QH9PaddedSource:
    """Build a single-split source over every molecule in ``db_path`` (ascending id)."""
    config = QH9PaddedConfig(max_atoms=6, max_edges=6 * 5, shuffle=shuffle)
    examples = read_qh9_sqlite(db_path)
    split_ids = tuple(int(ex.molecule_id) for ex in examples)
    return QH9PaddedSource(config, db_path=db_path, split_ids=split_ids)


def test_getitem_padded_shapes(synthetic_qh9_db: Path) -> None:
    """One padded element carries the documented fixed shapes and keys."""
    source = _source(synthetic_qh9_db)
    config = source.config
    element = source[0]
    assert element["native_fock"].shape == (config.max_ao, config.max_ao)
    assert element["decode_perm"].shape == (config.max_ao,)
    assert element["decode_sign"].shape == (config.max_ao,)
    assert element["atom_ao_start"].shape == (config.max_atoms,)
    assert element["atom_slot_indices"].shape == (config.max_atoms, FULL_ORBITALS)
    assert element["atomic_numbers"].shape == (config.max_atoms,)
    assert element["positions"].shape == (config.max_atoms, 3)
    assert element["edge_index"].shape == (2, config.max_edges)
    assert element["node_pad_mask"].shape == (config.max_atoms,)
    assert element["edge_pad_mask"].shape == (config.max_edges,)


def test_len_matches_molecule_count(synthetic_qh9_db: Path) -> None:
    """The source length equals the number of decoded molecules."""
    source = _source(synthetic_qh9_db)
    assert len(source) == len(read_qh9_sqlite(synthetic_qh9_db))


def test_pad_masks_and_padding_atoms(synthetic_qh9_db: Path) -> None:
    """Padded atoms carry Z=0 and node/edge pad masks mark only real entries."""
    source = _source(synthetic_qh9_db)
    h2o = source[0]  # 3 atoms -> 6 directed edges
    node_mask = np.asarray(h2o["node_pad_mask"])
    assert int(node_mask.sum()) == 3
    assert np.all(np.asarray(h2o["atomic_numbers"])[~node_mask] == 0)
    edge_mask = np.asarray(h2o["edge_pad_mask"])
    assert int(edge_mask.sum()) == 3 * 2  # complete directed graph


def test_get_batch_at_stacks_leading_axis(synthetic_qh9_db: Path) -> None:
    """``get_batch_at(start, size)`` stacks ``size`` molecules on a leading axis."""
    source = _source(synthetic_qh9_db)
    config = source.config
    batch = source.get_batch_at(0, 3)
    assert batch["native_fock"].shape == (3, config.max_ao, config.max_ao)
    assert batch["atomic_numbers"].shape == (3, config.max_atoms)
    assert batch["edge_index"].shape == (3, 2, config.max_edges)


def test_get_batch_at_matches_getitem(synthetic_qh9_db: Path) -> None:
    """Stacked batch rows equal the individual padded elements (sequential order)."""
    source = _source(synthetic_qh9_db)
    batch = source.get_batch_at(1, 2)
    for offset in range(2):
        element = source[1 + offset]
        np.testing.assert_array_equal(
            np.asarray(batch["native_fock"][offset]), np.asarray(element["native_fock"])
        )
        np.testing.assert_array_equal(
            np.asarray(batch["atom_slot_indices"][offset]),
            np.asarray(element["atom_slot_indices"]),
        )


def test_get_batch_at_wraps_partial_batch(synthetic_qh9_db: Path) -> None:
    """A batch past the end wraps around to fill ``size`` (masked downstream)."""
    source = _source(synthetic_qh9_db)
    n = len(source)
    batch = source.get_batch_at(n - 1, 3)
    # Last real molecule then wraps to molecules 0, 1.
    np.testing.assert_array_equal(
        np.asarray(batch["native_fock"][0]), np.asarray(source[n - 1]["native_fock"])
    )
    np.testing.assert_array_equal(
        np.asarray(batch["native_fock"][1]), np.asarray(source[0]["native_fock"])
    )


def test_decode_perm_is_a_permutation_of_real_ao(synthetic_qh9_db: Path) -> None:
    """The real AO block of ``decode_perm`` is a valid permutation."""
    source = _source(synthetic_qh9_db)
    examples = read_qh9_sqlite(synthetic_qh9_db)
    hcno = source[1]
    n_ao = examples[1].n_ao
    perm = np.asarray(hcno["decode_perm"])[:n_ao]
    np.testing.assert_array_equal(np.sort(perm), np.arange(n_ao))


def test_shuffle_permutes_order_with_full_coverage(synthetic_qh9_db: Path) -> None:
    """A shuffled source reorders molecules per epoch while covering each once."""
    source = _source(synthetic_qh9_db, shuffle=True)
    n = len(source)
    first_pass = _fingerprints_per_epoch(source, n)
    second_pass = _fingerprints_per_epoch(source, n)
    sequential = _fingerprints_per_epoch(_source(synthetic_qh9_db), n)
    # Coverage: each shuffled pass is a permutation of the full molecule set.
    assert sorted(first_pass) == sorted(sequential)
    assert sorted(second_pass) == sorted(sequential)
    assert len(set(first_pass)) == n  # every molecule exactly once
    # Successive shuffled epochs differ (the seed advances with the epoch counter).
    assert first_pass != second_pass


def _fingerprints_per_epoch(source: QH9PaddedSource, n: int) -> list[float]:
    """Return each molecule's native-Fock fingerprint over one epoch (unique per id)."""
    prints: list[float] = []
    for batch in iterate_padded_batches(source, 1):
        prints.append(round(float(np.asarray(batch["native_fock"][0]).sum()), 6))
    return prints[:n]


def test_create_sources_splits(synthetic_qh9_db: Path) -> None:
    """The factory returns three sources whose lengths match the reference split."""
    splits = create_qh9_padded_sources(config=_config(), db_path=synthetic_qh9_db, rngs=nnx.Rngs(0))
    examples = read_qh9_sqlite(synthetic_qh9_db)
    train_idx, val_idx, test_idx = qh9_random_split(len(examples), seed=_config().seed)
    assert len(splits.train) == len(train_idx)
    assert len(splits.val) == len(val_idx)
    assert len(splits.test) == len(test_idx)
    assert splits.units["fock"] == "Hartree"


def test_oversized_molecule_fails_fast(tmp_path: Path) -> None:
    """A molecule larger than ``max_atoms`` fails fast when its batch is read."""
    db_path = _write_db(tmp_path / "QH9Stable.db", [(_H2O_ATOMS, 1)])  # 3 atoms
    config = QH9PaddedConfig(max_atoms=2, max_edges=2)
    source = QH9PaddedSource(config, db_path=db_path, split_ids=(0,))
    with pytest.raises(ValueError, match="atoms > max_atoms"):
        source.get_batch_at(0, 1)
