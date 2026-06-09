r"""Tests for the native QH9-Stable dataset loader.

QH9 (Yu et al. 2023, "QH9", arXiv:2306.04922; reference
``/mnt/ssd2/Works/AIRS/OpenDFT/QHBench/QH9/datasets.py``) ships converged
def2-SVP Fock matrices in a SQLite ``.db`` with a single ``data`` table whose
rows are ``(id, num_nodes, atoms:int32, pos:float64 Angstrom, Ham:float64)``.

These tests build a TINY SYNTHETIC sqlite fixture (clearly labelled -- it is
NOT real QH9 data and carries no benchmark meaning) of two fake molecules (H2,
H2O) with random *symmetric* "Ham" blobs of the correct QH9-native def2-SVP
size, then assert that:

* :func:`read_qh9_sqlite` decodes rows to the right shapes/dtypes;
* the convention transform yields a symmetric ``(n_ao, n_ao)`` Fock whose AO
  count equals the opifex spherical def2-SVP AO count for that molecule;
* the random split is deterministic and reproduces the reference ratios.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path  # noqa: TC003

import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum._spherical import spherical_count
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import ANGSTROM_TO_BOHR, MolecularSystem
from opifex.data.sources.qh9_source import (
    create_qh9_loader,
    load_qh9_data,
    matrix_transform_def2svp,
    qh9_random_split,
    QH9Config,
    read_qh9_sqlite,
)


# QH9-native AO block sizes (5 for H/He, 14 for second-row elements).
_H2_ATOMS = np.array([1, 1], dtype=np.int32)
_H2O_ATOMS = np.array([8, 1, 1], dtype=np.int32)


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count for an atom array (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _random_symmetric(n: int, seed: int) -> np.ndarray:
    """Random symmetric float64 matrix (a stand-in QH9-native Fock blob)."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n)).astype(np.float64)
    return matrix + matrix.T


def _spherical_ao(atoms: np.ndarray) -> int:
    """opifex spherical def2-SVP AO count for the given atoms."""
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(atoms, dtype=jnp.int32),
        positions=jnp.zeros((len(atoms), 3), dtype=jnp.float64),
        basis_set="def2-svp",
    )
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
    return sum(spherical_count(shell.angular_momentum) for shell in basis.shells)


@pytest.fixture
def synthetic_qh9_db(tmp_path: Path) -> Path:
    """Write a tiny SYNTHETIC QH9-Stable sqlite db (H2, H2O) to tmp_path.

    NOT real QH9 data -- random symmetric Fock blobs of the correct
    QH9-native def2-SVP size, purely to exercise the production decode path
    without any download. Mirrors the reference row schema
    ``(id, num_nodes, atoms, pos, Ham)`` with the exact dtypes
    (``atoms`` int32, ``pos``/``Ham`` float64).
    """
    db_path = tmp_path / "QH9Stable.db"
    rows = [
        (
            0,
            2,
            _H2_ATOMS.tobytes(),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=np.float64).tobytes(),
            _random_symmetric(_native_ao(_H2_ATOMS), seed=1).tobytes(),
        ),
        (
            1,
            3,
            _H2O_ATOMS.tobytes(),
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]], dtype=np.float64
            ).tobytes(),
            _random_symmetric(_native_ao(_H2O_ATOMS), seed=2).tobytes(),
        ),
    ]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE data (id INTEGER, num_nodes INTEGER, atoms BLOB, pos BLOB, Ham BLOB)"
        )
        connection.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?)", rows)
        connection.commit()
    return db_path


# =============================================================================
# Decoding: shapes, dtypes, units
# =============================================================================


def test_read_decodes_expected_number_of_rows(synthetic_qh9_db: Path) -> None:
    """The loader decodes exactly the two fixture molecules."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    assert len(examples) == 2


def test_decoded_atomic_numbers_and_atom_counts(synthetic_qh9_db: Path) -> None:
    """Atomic numbers and atom counts match the source rows, ordered by id."""
    h2, h2o = read_qh9_sqlite(synthetic_qh9_db)
    np.testing.assert_array_equal(h2.atomic_numbers, _H2_ATOMS)
    np.testing.assert_array_equal(h2o.atomic_numbers, _H2O_ATOMS)
    assert h2.n_atoms == 2
    assert h2o.n_atoms == 3


def test_positions_converted_angstrom_to_bohr(synthetic_qh9_db: Path) -> None:
    """Stored Angstrom positions are exposed in Bohr on the MolecularSystem."""
    h2, _ = read_qh9_sqlite(synthetic_qh9_db)
    expected_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]) * ANGSTROM_TO_BOHR
    # MolecularSystem stores JAX arrays; under the suite's x64-off default these
    # are float32, so a float32-appropriate tolerance is used.
    np.testing.assert_allclose(np.asarray(h2.system.positions), expected_bohr, rtol=1e-5)
    assert h2.system.charge == 0
    assert h2.system.multiplicity == 1


def test_missing_database_raises(tmp_path: Path) -> None:
    """A missing database path fails fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_qh9_sqlite(tmp_path / "absent.db")


# =============================================================================
# Convention transform: AO count + symmetry alignment with opifex def2-SVP
# =============================================================================


def test_fock_ao_count_matches_opifex_spherical_basis(synthetic_qh9_db: Path) -> None:
    """The transformed Fock size equals the opifex spherical def2-SVP AO count."""
    h2, h2o = read_qh9_sqlite(synthetic_qh9_db)
    assert h2.n_ao == _spherical_ao(_H2_ATOMS)
    assert h2o.n_ao == _spherical_ao(_H2O_ATOMS)
    # H2O spherical def2-SVP AO count is the published 24 (O: 3+6+5, 2xH: 5).
    assert h2o.n_ao == 24
    assert h2.n_ao == 10


def test_transformed_fock_is_symmetric(synthetic_qh9_db: Path) -> None:
    """The def2-SVP Fock target is square and symmetric (a valid operator)."""
    for example in read_qh9_sqlite(synthetic_qh9_db):
        fock = example.fock
        assert fock.shape == (example.n_ao, example.n_ao)
        np.testing.assert_allclose(fock, fock.T, atol=1e-12)


def test_matrix_transform_is_a_signed_permutation_congruence() -> None:
    """The convention transform preserves symmetry and the spectrum.

    It is a symmetric ``S P (.) P^T S`` congruence with ``P`` a permutation and
    ``S = diag(+-1)``, so eigenvalues are invariant -- the property the reference
    ``matrix_transform`` relies on to keep a valid Fock operator.
    """
    native = _random_symmetric(_native_ao(_H2O_ATOMS), seed=7)
    transformed = matrix_transform_def2svp(native, _H2O_ATOMS)
    np.testing.assert_allclose(transformed, transformed.T, atol=1e-12)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(native)),
        np.sort(np.linalg.eigvalsh(transformed)),
        atol=1e-10,
    )


def test_matrix_transform_h_p_block_permutation() -> None:
    """Hydrogen's ``p`` shell is reordered by the QH9 ``[1, 2, 0]`` map.

    For a single H atom (def2-SVP ``ssp``: AO indices ``[s0, s1, p0, p1, p2]``)
    the convention permutes the p-block by ``[1, 2, 0]`` and leaves the s-block,
    giving the overall AO permutation ``[0, 1, 3, 4, 2]``. Asserting on a marker
    matrix pins the exact ordering replicated from the reference.
    """
    atoms = np.array([1], dtype=np.int32)
    marker = np.diag(np.arange(5, dtype=np.float64))  # 5 = H def2-SVP AOs
    transformed = matrix_transform_def2svp(marker, atoms)
    np.testing.assert_array_equal(np.diag(transformed), np.array([0.0, 1.0, 3.0, 4.0, 2.0]))


# =============================================================================
# Splitting: deterministic, reference ratios
# =============================================================================


def test_split_ratios_and_determinism() -> None:
    """The split reproduces the reference 0.8/0.1/0.1 sizes deterministically."""
    train, val, test = qh9_random_split(100, seed=43)
    assert (len(train), len(val), len(test)) == (80, 10, 10)
    train2, val2, test2 = qh9_random_split(100, seed=43)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(val, val2)
    np.testing.assert_array_equal(test, test2)


def test_split_matches_reference_permutation() -> None:
    """The split is the reference RandomState(43).permutation partition exactly."""
    expected = np.random.RandomState(43).permutation(50)
    train, val, test = qh9_random_split(50, seed=43)
    np.testing.assert_array_equal(train, expected[:40])
    np.testing.assert_array_equal(val, expected[40:45])
    np.testing.assert_array_equal(test, expected[45:])


def test_split_partitions_all_indices_disjointly() -> None:
    """Train/val/test indices are disjoint and cover every example exactly once."""
    train, val, test = qh9_random_split(37, seed=43)
    combined = np.concatenate([train, val, test])
    np.testing.assert_array_equal(np.sort(combined), np.arange(37))


# =============================================================================
# Loader bundle
# =============================================================================


def test_load_qh9_data_attaches_split_masks(synthetic_qh9_db: Path) -> None:
    """``load_qh9_data`` returns decoded examples plus split index arrays."""
    data = load_qh9_data(synthetic_qh9_db)
    assert data.n_examples == 2
    total = len(data.train_indices) + len(data.val_indices) + len(data.test_indices)
    assert total == 2


def test_create_qh9_loader_via_db_path(synthetic_qh9_db: Path) -> None:
    """The loader factory reads an explicit db_path and reports units."""
    loaders = create_qh9_loader(db_path=synthetic_qh9_db)
    assert loaders.n_train + loaders.n_val + loaders.n_test == 2
    assert loaders.units == {"positions": "Bohr", "fock": "Hartree"}


def test_create_qh9_loader_via_config(synthetic_qh9_db: Path) -> None:
    """A QH9Config pointing at the fixture's data_dir resolves the database."""
    config = QH9Config(data_dir=synthetic_qh9_db.parent.parent)
    # The fixture writes QH9Stable.db directly; place it where the config expects.
    raw_dir = config.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = config.database_path
    target.write_bytes(synthetic_qh9_db.read_bytes())
    loaders = create_qh9_loader(config=config)
    assert loaders.n_train + loaders.n_val + loaders.n_test == 2
