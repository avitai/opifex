r"""QH9-Stable quantum Hamiltonian dataset loader (native opifex, no torch).

QH9 (Yu et al. 2023, "QH9: A Quantum Hamiltonian Prediction Benchmark for
QM9 Molecules", arXiv:2306.04922) ships converged DFT (B3LYP/def2-SVP) Fock
matrices for 130 831 small organic molecules (elements H, C, N, O, F). This
module reads the **QH9-Stable** SQLite database directly with the standard
library :mod:`sqlite3` and :mod:`numpy` -- no ``torch``, ``torch_geometric``,
``lmdb`` or ``apsw`` dependency -- and yields, per molecule, an opifex
:class:`~opifex.core.quantum.molecular_system.MolecularSystem` paired with the
def2-SVP Fock matrix in opifex's spherical shell/AO ordering.

Reference schema (the authority this loader matches byte-for-byte):
    ``/mnt/ssd2/Works/AIRS/OpenDFT/QHBench/QH9/datasets.py`` (``QH9Stable``).

Raw schema
----------
The database has a single table ``data``; each row is positionally
``(id: int, N: int, Z: bytes, pos: bytes, Ham: bytes)`` (the real QH9-Stable
column names are ``id, N, Z, pos, Ham``; the reference reads them positionally
via ``SELECT *``, never by name) with

* ``Z``      -- ``int32`` atomic numbers, length ``N``;
* ``pos``    -- ``float64`` Cartesian coordinates ``(N, 3)`` in Angstrom;
* ``Ham``    -- ``float64`` Fock matrix ``(n_ao, n_ao)`` flattened row-major,
  where ``n_ao = sum(5 if Z <= 2 else 14 for Z in Z)`` (the QH9 native AO
  block sizes: 5 for H/He, 14 for the second-row elements).

AO-ordering convention
----------------------
QH9 stores the Fock matrix in its own native AO layout. The reference
``matrix_transform(Ham, atoms, convention='pyscf_def2svp')`` reorders and
sign-flips it into the **PySCF def2-SVP spherical** convention: atom-major,
shell-major, with ``p`` components ordered ``(x, y, z)`` and ``d`` components
ordered ``(xy, yz, z^2, xz, x2-y2)``. That convention is *exactly* opifex's
spherical def2-SVP AO ordering (the columns of
:func:`opifex.core.quantum._spherical.build_block_transform` are PySCF's
``mol.spheric_labels()`` order, and the basis enumerates shells atom-major /
shell-major), so the transformed matrix lands directly in the predictor's
target ordering with no further permutation. The convention permutation/sign
tables (the ``_DEF2SVP_*`` module constants) are replicated verbatim from the
reference ``convention_dict['pyscf_def2svp']``.

Splitting
---------
The QH9-Stable *random* split is reproduced exactly:
``np.random.RandomState(seed=43).permutation(n)`` partitioned ``0.8 / 0.1 / 0.1``
with integer truncation (the test split absorbs the remainder).

Batching
--------
QH9 molecules are heterogeneous (``n_atoms`` 3..29; AO count varies with
composition), so they cannot be collated into one dense tensor without padding.
This module owns only the SQLite decode, the def2-SVP convention transform and
the reference split; the block-form batching that consumes them lives in
:mod:`opifex.data.sources.qh9_blocks` and
:mod:`opifex.data.sources.qh9_block_stream` (per-atom / per-edge QHNet blocks
collated into a single flat batch).

No download happens at import time; the loader reads an existing
``QH9Stable.db`` (default ``/mnt/ssd2/Data/qh9/raw/QH9Stable.db``).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.core.quantum.molecular_system import ANGSTROM_TO_BOHR, MolecularSystem


logger = logging.getLogger(__name__)

# QH9-Stable random-split seed and ratios (reference ``QH9Stable.process``).
_SPLIT_SEED: int = 43
_TRAIN_RATIO: float = 0.8
_VAL_RATIO: float = 0.1

# QH9 native per-element AO block sizes: 5 for H/He (Z <= 2), 14 otherwise.
_LIGHT_ELEMENT_AO: int = 5
_HEAVY_ELEMENT_AO: int = 14
_LIGHT_ELEMENT_MAX_Z: int = 2


# ---------------------------------------------------------------------------
# PySCF def2-SVP convention tables (replicated verbatim from the reference
# ``convention_dict['pyscf_def2svp']`` in QHBench/QH9/datasets.py). These map
# QH9's native per-shell AO layout into PySCF's spherical AO ordering.
#
#   atom_to_orbitals_map: per-element shell string (s/p/d letters in order).
#   orbital_idx_map:      within-shell component permutation per shell type.
#   orbital_sign_map:     within-shell component sign flips per shell type.
#   orbital_order_map:    per-element shell reordering (whole shells).
# ---------------------------------------------------------------------------
_DEF2SVP_ATOM_TO_ORBITALS: dict[int, str] = {
    1: "ssp",
    6: "sssppd",
    7: "sssppd",
    8: "sssppd",
    9: "sssppd",
}
_DEF2SVP_ORBITAL_IDX: dict[str, list[int]] = {
    "s": [0],
    "p": [1, 2, 0],
    "d": [0, 1, 2, 3, 4],
}
_DEF2SVP_ORBITAL_SIGN: dict[str, list[int]] = {
    "s": [1],
    "p": [1, 1, 1],
    "d": [1, 1, 1, 1, 1],
}
_DEF2SVP_ORBITAL_ORDER: dict[int, list[int]] = {
    1: [0, 1, 2],
    6: [0, 1, 2, 3, 4, 5],
    7: [0, 1, 2, 3, 4, 5],
    8: [0, 1, 2, 3, 4, 5],
    9: [0, 1, 2, 3, 4, 5],
}


@dataclass(frozen=True)
class QH9Example:
    """A single decoded QH9 molecule and its def2-SVP Fock target.

    Attributes:
        molecule_id: The ``id`` primary key of the source row.
        system: The opifex molecular system (positions in Bohr, charge 0,
            multiplicity 1).
        fock: def2-SVP Fock matrix in opifex spherical AO ordering, shape
            ``(n_ao, n_ao)``, symmetric, Hartree.
        atomic_numbers: Nuclear charges, shape ``(n_atoms,)``.
    """

    molecule_id: int
    system: MolecularSystem
    fock: Float[NDArray[np.float64], "n_ao n_ao"]
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"]

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return int(self.atomic_numbers.shape[0])

    @property
    def n_ao(self) -> int:
        """Number of spherical def2-SVP atomic orbitals."""
        return int(self.fock.shape[0])


@dataclass(frozen=True)
class QH9Data:
    """All decoded QH9-Stable examples plus the deterministic split masks.

    Attributes:
        examples: Decoded molecules in source (``id``-sorted) order.
        train_indices: Indices into ``examples`` for the training split.
        val_indices: Indices into ``examples`` for the validation split.
        test_indices: Indices into ``examples`` for the test split.
    """

    examples: tuple[QH9Example, ...]
    train_indices: Int[NDArray[np.int64], " n_train"]
    val_indices: Int[NDArray[np.int64], " n_val"]
    test_indices: Int[NDArray[np.int64], " n_test"]

    @property
    def n_examples(self) -> int:
        """Total number of decoded molecules."""
        return len(self.examples)


# =============================================================================
# Convention transform (QH9 native -> PySCF def2-SVP spherical AO ordering)
# =============================================================================


def matrix_transform_def2svp(
    matrix: Float[NDArray[np.float64], "n_ao n_ao"],
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"],
) -> Float[NDArray[np.float64], "n_ao n_ao"]:
    r"""Reorder a QH9-native Fock matrix into PySCF def2-SVP spherical ordering.

    Faithful NumPy reimplementation of the reference
    ``matrix_transform(matrices, atoms, convention='pyscf_def2svp')`` from
    ``QHBench/QH9/datasets.py``: it builds the per-AO permutation (whole-shell
    reordering composed with within-shell component reordering) and the per-AO
    sign vector, then applies them symmetrically to both matrix axes
    (``M' = S M[I][:, I] S^T`` with ``S = diag(signs)``).

    Args:
        matrix: QH9-native Fock matrix, shape ``(n_ao, n_ao)``.
        atomic_numbers: Nuclear charges of the molecule, shape ``(n_atoms,)``.

    Returns:
        The Fock matrix in PySCF/opifex def2-SVP spherical AO ordering.

    Raises:
        KeyError: If an atomic number is outside the QH9 element set
            (H, C, N, O, F).
    """
    orbitals = ""
    orbitals_order: list[int] = []
    for atomic_number in atomic_numbers:
        offset = len(orbitals_order)
        orbitals += _DEF2SVP_ATOM_TO_ORBITALS[int(atomic_number)]
        orbitals_order += [idx + offset for idx in _DEF2SVP_ORBITAL_ORDER[int(atomic_number)]]

    transform_indices: list[NDArray[np.int64]] = []
    transform_signs: list[NDArray[np.int64]] = []
    for orbital in orbitals:
        offset = sum(len(block) for block in transform_indices)
        transform_indices.append(np.asarray(_DEF2SVP_ORBITAL_IDX[orbital], dtype=np.int64) + offset)
        transform_signs.append(np.asarray(_DEF2SVP_ORBITAL_SIGN[orbital], dtype=np.int64))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    indices = np.concatenate(transform_indices).astype(np.int64)
    signs = np.concatenate(transform_signs)

    transformed = matrix[..., indices, :][..., :, indices]
    return transformed * signs[:, None] * signs[None, :]


# =============================================================================
# SQLite decoding
# =============================================================================


def _native_ao_count(atomic_numbers: Int[NDArray[np.int32], " n_atoms"]) -> int:
    """QH9 native total AO count: 5 per H/He, 14 per heavier atom."""
    return int(
        sum(
            _LIGHT_ELEMENT_AO if int(z) <= _LIGHT_ELEMENT_MAX_Z else _HEAVY_ELEMENT_AO
            for z in atomic_numbers
        )
    )


def _decode_row(
    molecule_id: int,
    num_nodes: int,
    atoms_blob: bytes,
    pos_blob: bytes,
    ham_blob: bytes,
) -> QH9Example:
    """Decode one raw QH9-Stable ``data`` row into a :class:`QH9Example`.

    Mirrors the reference ``QH9Stable.get`` decoding: ``atoms`` as ``int32``,
    ``pos`` as ``float64`` ``(num_nodes, 3)`` in Angstrom, ``Ham`` as
    ``float64`` ``(n_ao, n_ao)`` in QH9-native ordering. The Fock matrix is then
    mapped into def2-SVP spherical ordering and the positions converted to Bohr
    for the opifex :class:`MolecularSystem` (atomic-unit convention).

    Args:
        molecule_id: The row ``id``.
        num_nodes: Number of atoms.
        atoms_blob: Raw ``int32`` atomic-number bytes.
        pos_blob: Raw ``float64`` position bytes (Angstrom).
        ham_blob: Raw ``float64`` Fock-matrix bytes (QH9-native ordering).

    Returns:
        The decoded example with the def2-SVP spherical Fock target.
    """
    atomic_numbers = np.frombuffer(atoms_blob, dtype=np.int32).copy()
    positions_angstrom = np.frombuffer(pos_blob, dtype=np.float64).reshape(num_nodes, 3).copy()
    n_ao = _native_ao_count(atomic_numbers)
    native_fock = np.frombuffer(ham_blob, dtype=np.float64).reshape(n_ao, n_ao).copy()

    fock = matrix_transform_def2svp(native_fock, atomic_numbers)

    positions_bohr = positions_angstrom * ANGSTROM_TO_BOHR
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(atomic_numbers, dtype=jnp.int32),
        positions=jnp.asarray(positions_bohr, dtype=jnp.float64),
        charge=0,
        multiplicity=1,
        basis_set="def2-svp",
    )
    return QH9Example(
        molecule_id=int(molecule_id),
        system=system,
        fock=fock,
        atomic_numbers=atomic_numbers,
    )


def read_qh9_sqlite(
    db_path: Path,
    *,
    limit: int | None = None,
) -> tuple[QH9Example, ...]:
    """Decode the QH9-Stable ``data`` table into :class:`QH9Example` records.

    Uses the standard-library :mod:`sqlite3` driver under a ``with`` connection
    (no ``torch``/``apsw``/``lmdb``). Rows are read in ascending ``id`` order so
    the returned ordering is deterministic and matches the order the reference
    split permutation is applied over.

    Args:
        db_path: Path to ``QH9Stable.db``.
        limit: Optional cap on the number of rows decoded (for quick
            smoke-tests); ``None`` decodes every row.

    Returns:
        The decoded examples, ordered by ascending ``id``.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"QH9-Stable database not found: {db_path}")

    # Real QH9-Stable columns are (id, N, Z, pos, Ham); read positionally via
    # ``SELECT *`` (as the reference does) so we are agnostic to column names.
    query = "SELECT * FROM data ORDER BY id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    logger.info("Reading QH9-Stable rows from %s", db_path)
    examples: list[QH9Example] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        cursor = connection.execute(query)
        for molecule_id, num_nodes, atoms_blob, pos_blob, ham_blob in cursor:
            examples.append(_decode_row(molecule_id, num_nodes, atoms_blob, pos_blob, ham_blob))
    logger.info("Decoded %d QH9-Stable molecules", len(examples))
    return tuple(examples)


# =============================================================================
# Splitting + loader construction
# =============================================================================


def qh9_random_split(
    n_examples: int,
    *,
    seed: int = _SPLIT_SEED,
) -> tuple[
    Int[NDArray[np.int64], " n_train"],
    Int[NDArray[np.int64], " n_val"],
    Int[NDArray[np.int64], " n_test"],
]:
    """Reproduce the QH9-Stable random ``0.8 / 0.1 / 0.1`` split.

    Matches ``QH9Stable.process`` exactly: a single
    ``np.random.RandomState(seed).permutation(n)`` is split with integer
    truncation of the train and validation sizes; the test split takes the
    remainder.

    Args:
        n_examples: Total number of molecules.
        seed: Permutation seed (fixed at 43 in the reference).

    Returns:
        Tuple of ``(train_indices, val_indices, test_indices)``.
    """
    n_train = int(n_examples * _TRAIN_RATIO)
    n_val = int(n_examples * _VAL_RATIO)
    permutation = np.random.RandomState(seed).permutation(n_examples)
    train_indices = permutation[:n_train]
    val_indices = permutation[n_train : n_train + n_val]
    test_indices = permutation[n_train + n_val :]
    return (
        train_indices.astype(np.int64),
        val_indices.astype(np.int64),
        test_indices.astype(np.int64),
    )


def load_qh9_data(
    db_path: Path,
    *,
    seed: int = _SPLIT_SEED,
    limit: int | None = None,
) -> QH9Data:
    """Decode QH9-Stable and attach the deterministic random-split masks.

    Args:
        db_path: Path to ``QH9Stable.db``.
        seed: Split permutation seed (fixed at 43 in the reference).
        limit: Optional cap on decoded rows.

    Returns:
        A :class:`QH9Data` bundle of examples plus split indices.
    """
    examples = read_qh9_sqlite(db_path, limit=limit)
    train_indices, val_indices, test_indices = qh9_random_split(len(examples), seed=seed)
    return QH9Data(
        examples=examples,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


__all__ = [
    "QH9Data",
    "QH9Example",
    "load_qh9_data",
    "matrix_transform_def2svp",
    "qh9_random_split",
    "read_qh9_sqlite",
]
