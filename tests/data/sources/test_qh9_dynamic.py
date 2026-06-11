"""QH9-Dynamic loader: canonical geometry/mol splits and row decoding.

QH9-Dynamic stores ~100 molecular-dynamics geometries per molecule in one
``data`` table whose rows are ``(id, geo_id, N, Z, pos, ekin, epot, etot, time,
Ham, converged)`` -- the ``id`` is an ``int64`` blob (the *molecule* id, shared
across its geometries) and ``Ham`` is column 9, unlike QH9-Stable's
``(id, N, Z, pos, Ham)``. These tests pin the two reference splits
(``geometry`` and ``mol``, Yu et al. 2023) against an independent inline
reimplementation and check the decoder on a tiny synthetic Dynamic database.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path  # noqa: TC003

import numpy as np

from opifex.data.sources.qh9_dynamic import (
    QH9_DYNAMIC_GEOMETRIES_PER_MOL,
    qh9_dynamic_geometry_split,
    qh9_dynamic_mol_split,
    QH9_DYNAMIC_MOLECULE_COUNTS,
    read_qh9_dynamic_sqlite,
)


_GEO_PER_MOL = QH9_DYNAMIC_GEOMETRIES_PER_MOL


# --- Independent oracles mirroring AIRS/OpenDFT/QHBench/QH9/datasets.py --------
def _reference_geometry_split(
    num_mol: int, geo: int = _GEO_PER_MOL, n_train: int = 80, n_val: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-molecule 80/10/10 geometry split (RandomState seeded by boundary index)."""
    train, val, test = [], [], []
    cur = 0
    for _ in range(num_mol):
        indices = np.random.RandomState(seed=cur + geo - 1).permutation(geo)
        train.extend(cur + indices[:n_train])
        val.extend(cur + indices[n_train : n_train + n_val])
        test.extend(cur + indices[n_train + n_val :])
        cur += geo
    return np.array(train), np.array(val), np.array(test)


def _reference_mol_split(
    num_mol: int, geo: int = _GEO_PER_MOL, seed: int = 43
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Molecule-wise 80/10/10 split (stdlib random, seed 43)."""
    mol_id_list = [i for i in range(num_mol) for _ in range(geo)]
    index_list = list(range(num_mol))
    random.seed(seed)
    random.shuffle(index_list)
    n_train = int(num_mol * 0.8)
    n_val = int(num_mol * 0.1)
    train_ids = index_list[:n_train]
    val_ids = index_list[n_train : n_train + n_val]
    test_ids = index_list[n_train + n_val :]
    arr = np.array(mol_id_list)
    train = np.where(np.isin(arr, train_ids))[0].astype(np.int64)
    val = np.where(np.isin(arr, val_ids))[0].astype(np.int64)
    test = np.where(np.isin(arr, test_ids))[0].astype(np.int64)
    return train, val, test


# --- Geometry split -----------------------------------------------------------
def test_geometry_split_matches_reference() -> None:
    """The geometry split reproduces the reference indices exactly."""
    train, val, test = qh9_dynamic_geometry_split(5)
    ref_train, ref_val, ref_test = _reference_geometry_split(5)
    np.testing.assert_array_equal(train, ref_train)
    np.testing.assert_array_equal(val, ref_val)
    np.testing.assert_array_equal(test, ref_test)


def test_geometry_split_sizes_and_partition() -> None:
    """Geometry split is an 80/10/10-per-molecule partition of all rows."""
    num_mol = 7
    train, val, test = qh9_dynamic_geometry_split(num_mol)
    assert len(train) == 80 * num_mol
    assert len(val) == 10 * num_mol
    assert len(test) == 10 * num_mol
    combined = np.concatenate([train, val, test])
    np.testing.assert_array_equal(np.sort(combined), np.arange(num_mol * _GEO_PER_MOL))


# --- Mol split ----------------------------------------------------------------
def test_mol_split_matches_reference() -> None:
    """The molecule-wise split reproduces the reference indices exactly."""
    train, val, test = qh9_dynamic_mol_split(20)
    ref_train, ref_val, ref_test = _reference_mol_split(20)
    np.testing.assert_array_equal(np.sort(train), np.sort(ref_train))
    np.testing.assert_array_equal(np.sort(val), np.sort(ref_val))
    np.testing.assert_array_equal(np.sort(test), np.sort(ref_test))


def test_mol_split_keeps_whole_molecules_together() -> None:
    """Every molecule's 100 geometries land entirely in one split (no leakage)."""
    train, val, test = qh9_dynamic_mol_split(20)
    for split in (train, val, test):
        mols = np.unique(split // _GEO_PER_MOL)
        # Each molecule present contributes all its geometries.
        for mol in mols:
            assert np.sum(split // _GEO_PER_MOL == mol) == _GEO_PER_MOL


def test_molecule_counts_known() -> None:
    """The per-version molecule counts match the QH9-Dynamic benchmark."""
    assert QH9_DYNAMIC_MOLECULE_COUNTS["100k"] == 999
    assert QH9_DYNAMIC_MOLECULE_COUNTS["300k"] == 2998


# --- Decoder on a synthetic Dynamic database ----------------------------------
def _native_ao(atoms: np.ndarray) -> int:
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _write_dynamic_db(db_path: Path, specs: list[tuple[np.ndarray, int]]) -> Path:
    """Write a synthetic QH9-Dynamic db: 11 columns, id as int64 blob, Ham at col 9."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE data (id BLOB, geo_id INTEGER, N INTEGER, Z BLOB, pos BLOB, "
            "ekin REAL, epot REAL, etot REAL, time REAL, Ham BLOB, converged INTEGER)"
        )
        rows = []
        for row_index, (atoms, seed) in enumerate(specs):
            rng = np.random.default_rng(seed)
            positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
            n_ao = _native_ao(atoms)
            fock = rng.standard_normal((n_ao, n_ao)).astype(np.float64)
            fock = fock + fock.T
            rows.append(
                (
                    np.array([seed], dtype=np.int64).tobytes(),
                    row_index * 10,
                    len(atoms),
                    atoms.astype(np.int32).tobytes(),
                    positions.tobytes(),
                    0.0,
                    -1.0,
                    -1.0,
                    float(row_index),
                    fock.tobytes(),
                    1,
                )
            )
        connection.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return db_path


def test_read_dynamic_decodes_rows(tmp_path: Path) -> None:
    """The Dynamic reader decodes id/atoms/positions/Fock from the 11-column rows."""
    atoms = np.array([8, 1, 1], dtype=np.int32)
    db = _write_dynamic_db(tmp_path / "QH9Dynamic_100k.db", [(atoms, 1), (atoms, 2)])
    examples = read_qh9_dynamic_sqlite(db)
    assert len(examples) == 2
    first = examples[0]
    assert first.molecule_id == 1
    np.testing.assert_array_equal(np.asarray(first.atomic_numbers), atoms)
    assert first.native_fock.shape == (_native_ao(atoms), _native_ao(atoms))
    # The native Fock is symmetric (as in the real database).
    assert float(np.max(np.abs(first.native_fock - first.native_fock.T))) == 0.0
