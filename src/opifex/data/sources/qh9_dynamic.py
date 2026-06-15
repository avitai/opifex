r"""QH9-Dynamic quantum-Hamiltonian dataset loader (native opifex, no torch).

QH9-Dynamic (Yu et al. 2023, "QH9: A Quantum Hamiltonian Prediction Benchmark for
QM9 Molecules", arXiv:2306.04922) stores ~100 molecular-dynamics geometries per
molecule. The single ``data`` table has eleven positional columns
``(id, geo_id, N, Z, pos, ekin, epot, etot, time, Ham, converged)`` -- the ``id``
is an ``int64`` blob holding the *molecule* id (shared across that molecule's
geometries) and the Fock matrix ``Ham`` is column 9. This differs from
QH9-Stable's ``(id, N, Z, pos, Ham)``; the decoding of ``atoms``/``pos``/``Ham``
into a :class:`~opifex.data.sources.qh9_source.QH9Example` is otherwise identical,
so the shared :func:`~opifex.data.sources.qh9_source._decode_row` is reused.

Two canonical splits accompany the benchmark, both 80/10/10 and reproduced here
exactly from the reference ``QH9Dynamic.process``:

* ``geometry`` -- within each molecule, the ~100 geometries are split with a
  per-molecule ``numpy`` permutation seeded by the molecule's last row index, so
  train/val/test all see every molecule but at different timesteps.
* ``mol`` -- molecules are partitioned whole (stdlib ``random`` seed 43), so a
  molecule's geometries never straddle splits (the harder generalisation test).

Both return *row-index* masks over the table read in ascending ``rowid`` order
(the reference's enumeration order), since the molecule ``id`` is not unique.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path  # noqa: TC003
from typing import cast

import numpy as np
from jaxtyping import Int  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.data.sources.qh9_source import _decode_row, QH9Example


logger = logging.getLogger(__name__)

QH9_DYNAMIC_GEOMETRIES_PER_MOL = 100
"""Molecular-dynamics geometries stored per molecule in QH9-Dynamic."""

QH9_DYNAMIC_MOLECULE_COUNTS = {"100k": 999, "300k": 2998}
"""Distinct molecules per QH9-Dynamic version (rows = molecules x 100 geometries)."""

# Positional column indices of the eleven-column QH9-Dynamic ``data`` row.
_ID_INDEX = 0
_NUM_NODES_INDEX = 2
_ATOMS_INDEX = 3
_POSITIONS_INDEX = 4
_HAMILTONIAN_INDEX = 9

_TRAIN_RATIO = 0.8
_VAL_RATIO = 0.1
_MOL_SPLIT_SEED = 43


def qh9_dynamic_geometry_split(
    num_molecules: int,
    *,
    geometries_per_mol: int = QH9_DYNAMIC_GEOMETRIES_PER_MOL,
    num_train: int = 80,
    num_val: int = 10,
) -> tuple[
    Int[NDArray[np.int64], " n_train"],
    Int[NDArray[np.int64], " n_val"],
    Int[NDArray[np.int64], " n_test"],
]:
    """Reproduce the QH9-Dynamic geometry-wise ``80 / 10 / 10`` split.

    For each molecule (a contiguous block of ``geometries_per_mol`` rows) the
    geometries are permuted with ``np.random.RandomState`` seeded by that block's
    last row index and split into train/val/test; the masks are concatenated in
    molecule order. Every molecule appears in all three splits but at disjoint
    timesteps.

    Args:
        num_molecules: Number of distinct molecules (e.g. 2998 for the 300k set).
        geometries_per_mol: Geometries stored per molecule (100 in the benchmark).
        num_train: Geometries per molecule assigned to train.
        num_val: Geometries per molecule assigned to validation (test gets the
            remainder).

    Returns:
        Row-index arrays ``(train, val, test)`` over the ascending-``rowid`` rows.
    """
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    base = 0
    for _ in range(num_molecules):
        permutation = np.random.RandomState(seed=base + geometries_per_mol - 1).permutation(
            geometries_per_mol
        )
        train.extend(base + permutation[:num_train])
        val.extend(base + permutation[num_train : num_train + num_val])
        test.extend(base + permutation[num_train + num_val :])
        base += geometries_per_mol
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def qh9_dynamic_mol_split(
    num_molecules: int,
    *,
    geometries_per_mol: int = QH9_DYNAMIC_GEOMETRIES_PER_MOL,
    seed: int = _MOL_SPLIT_SEED,
) -> tuple[
    Int[NDArray[np.int64], " n_train"],
    Int[NDArray[np.int64], " n_val"],
    Int[NDArray[np.int64], " n_test"],
]:
    """Reproduce the QH9-Dynamic molecule-wise ``80 / 10 / 10`` split.

    Molecule ids are shuffled with the stdlib :mod:`random` (seed 43) and
    partitioned by integer-truncated ratios; every row of a molecule's
    ``geometries_per_mol`` block lands in that molecule's split, so a molecule's
    geometries never straddle the train/val/test boundary.

    Args:
        num_molecules: Number of distinct molecules.
        geometries_per_mol: Geometries stored per molecule (100 in the benchmark).
        seed: Shuffle seed (fixed at 43 in the reference).

    Returns:
        Row-index arrays ``(train, val, test)`` over the ascending-``rowid`` rows.
    """
    molecule_of_row = np.repeat(np.arange(num_molecules), geometries_per_mol)
    index_list = list(range(num_molecules))
    random.seed(seed)
    random.shuffle(index_list)
    num_train = int(num_molecules * _TRAIN_RATIO)
    num_val = int(num_molecules * _VAL_RATIO)
    train_ids = index_list[:num_train]
    val_ids = index_list[num_train : num_train + num_val]
    test_ids = index_list[num_train + num_val :]
    train = np.where(np.isin(molecule_of_row, train_ids))[0].astype(np.int64)
    val = np.where(np.isin(molecule_of_row, val_ids))[0].astype(np.int64)
    test = np.where(np.isin(molecule_of_row, test_ids))[0].astype(np.int64)
    return train, val, test


def decode_qh9_dynamic_row(row: tuple[object, ...]) -> QH9Example:
    """Decode one eleven-column QH9-Dynamic ``data`` row into a :class:`QH9Example`.

    Extracts the molecule ``id`` from its ``int64`` blob and the
    ``N``/``Z``/``pos``/``Ham`` columns at their QH9-Dynamic positions, then
    defers to the shared :func:`~opifex.data.sources.qh9_source._decode_row` for
    the def2-SVP spherical Fock transform.

    Args:
        row: A full ``SELECT *`` row of the QH9-Dynamic ``data`` table.

    Returns:
        The decoded example with its native + spherical Fock.
    """
    molecule_id = int(np.frombuffer(cast("bytes", row[_ID_INDEX]), dtype=np.int64)[0])
    num_nodes = cast("int", row[_NUM_NODES_INDEX])
    return _decode_row(
        molecule_id,
        num_nodes,
        cast("bytes", row[_ATOMS_INDEX]),
        cast("bytes", row[_POSITIONS_INDEX]),
        cast("bytes", row[_HAMILTONIAN_INDEX]),
    )


def read_qh9_dynamic_sqlite(
    db_path: Path,
    *,
    limit: int | None = None,
) -> tuple[QH9Example, ...]:
    """Decode the QH9-Dynamic ``data`` table into :class:`QH9Example` records.

    Reads rows in ascending ``rowid`` order (the reference enumeration order the
    splits index over) with the standard-library :mod:`sqlite3` driver under a
    read-only connection. This is the eager path (for validation and small
    evaluations); large-scale training uses the out-of-core padded source.

    Args:
        db_path: Path to ``QH9Dynamic_300k.db`` / ``QH9Dynamic_100k.db``.
        limit: Optional cap on the number of rows decoded (ascending ``rowid``).

    Returns:
        The decoded examples in ascending-``rowid`` order.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"QH9-Dynamic database not found: {db_path}")
    query = "SELECT * FROM data ORDER BY rowid"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    logger.info("Reading QH9-Dynamic rows from %s", db_path)
    examples: list[QH9Example] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for row in connection.execute(query):
            examples.append(decode_qh9_dynamic_row(row))
    logger.info("Decoded %d QH9-Dynamic geometries", len(examples))
    return tuple(examples)


__all__ = [
    "QH9_DYNAMIC_GEOMETRIES_PER_MOL",
    "QH9_DYNAMIC_MOLECULE_COUNTS",
    "decode_qh9_dynamic_row",
    "qh9_dynamic_geometry_split",
    "qh9_dynamic_mol_split",
    "read_qh9_dynamic_sqlite",
]
