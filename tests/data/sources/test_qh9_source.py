r"""Tests for the native QH9-Stable dataset loader.

QH9 (Yu et al. 2023, "QH9", arXiv:2306.04922; reference
``/mnt/ssd2/Works/AIRS/OpenDFT/QHBench/QH9/datasets.py``) ships converged
def2-SVP Fock matrices in a SQLite ``.db`` with a single ``data`` table whose
rows are ``(id, N, Z:int32, pos:float64 Angstrom, Ham:float64)`` (verified
against the real 130,831-row QH9Stable.db: columns ``id, N, Z, pos, Ham``).

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
    _group_examples,
    _stack_bucket,
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
# A second 3-atom composition (HCN) -- same n_atoms as water, different AO count
# and different Z signature, so it exercises n_atoms-bucket padding/masking.
_HCN_ATOMS = np.array([1, 6, 7], dtype=np.int32)


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
    ``(id, N, Z, pos, Ham)`` with the exact dtypes
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
        connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
        connection.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?)", rows)
        connection.commit()
    return db_path


def _make_row(molecule_id: int, atoms: np.ndarray, seed: int) -> tuple:
    """Build one synthetic QH9 row (id, N, Z, pos, Ham) for ``atoms``."""
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
def multi_size_qh9_db(tmp_path: Path) -> Path:
    """A SYNTHETIC QH9-Stable db spanning three Z signatures across two sizes.

    Three H2O molecules + two H2 molecules + one HCN molecule (NOT real QH9
    data). H2O and HCN share ``n_atoms == 3`` but differ in composition (so in
    AO count and Z signature), letting the tests distinguish ``"signature"`` from
    ``"n_atoms"`` bucketing and exercise n_atoms padding/masking.
    """
    db_path = tmp_path / "QH9Stable.db"
    specs = [
        (_H2O_ATOMS, 10),
        (_H2O_ATOMS, 11),
        (_H2O_ATOMS, 12),
        (_H2_ATOMS, 20),
        (_H2_ATOMS, 21),
        (_HCN_ATOMS, 30),
    ]
    rows = [_make_row(index, atoms, seed) for index, (atoms, seed) in enumerate(specs)]
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
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


# =============================================================================
# Size-bucketed batching: grouping, padding + mask, datarax pipeline shapes
# =============================================================================


def test_signature_bucketing_groups_by_z_sequence(multi_size_qh9_db: Path) -> None:
    """``"signature"`` bucketing separates H2O, H2 and HCN into three buckets."""
    examples = read_qh9_sqlite(multi_size_qh9_db)
    buckets = _group_examples(examples, "signature")
    sizes = sorted(len(bucket) for bucket in buckets)
    assert sizes == [1, 2, 3]  # HCN (1), H2 (2), H2O (3)
    for bucket in buckets:
        signatures = {tuple(int(z) for z in ex.atomic_numbers) for ex in bucket}
        assert len(signatures) == 1  # every molecule shares the bucket's Z sequence


def test_n_atoms_bucketing_merges_same_size_compositions(multi_size_qh9_db: Path) -> None:
    """``"n_atoms"`` bucketing merges H2O and HCN (both 3-atom) into one bucket."""
    examples = read_qh9_sqlite(multi_size_qh9_db)
    buckets = _group_examples(examples, "n_atoms")
    sizes = sorted(len(bucket) for bucket in buckets)
    # 3-atom bucket = 3 H2O + 1 HCN = 4; 2-atom bucket = 2 H2.
    assert sizes == [2, 4]


def test_stack_bucket_signature_needs_no_padding(multi_size_qh9_db: Path) -> None:
    """A signature bucket stacks without padding: the mask is all-true."""
    examples = read_qh9_sqlite(multi_size_qh9_db)
    water_bucket = next(b for b in _group_examples(examples, "signature") if len(b) == 3)
    stacked, n_atoms, n_ao = _stack_bucket(water_bucket)
    assert n_atoms == 3
    assert n_ao == _spherical_ao(_H2O_ATOMS)
    assert stacked["positions"].shape == (3, 3, 3)
    assert stacked["fock"].shape == (3, n_ao, n_ao)
    assert stacked["mask"].shape == (3, n_ao, n_ao)
    assert bool(stacked["mask"].all())  # homogeneous bucket -> no padding


def test_stack_bucket_n_atoms_pads_and_masks(multi_size_qh9_db: Path) -> None:
    """An n_atoms bucket pads the smaller composition and masks the padding."""
    examples = read_qh9_sqlite(multi_size_qh9_db)
    three_atom = next(b for b in _group_examples(examples, "n_atoms") if len(b) == 4)
    stacked, n_atoms, n_ao = _stack_bucket(three_atom)
    assert n_atoms == 3
    per_molecule_ao = [int(value) for value in stacked["n_ao"]]
    assert n_ao == max(per_molecule_ao)
    # Each molecule's valid region equals its real AO count squared; padding is
    # masked out and carries zero target.
    for index, real_ao in enumerate(per_molecule_ao):
        assert int(stacked["mask"][index].sum()) == real_ao * real_ao
        if real_ao < n_ao:
            np.testing.assert_array_equal(stacked["fock"][index, real_ao:, :], 0.0)
            np.testing.assert_array_equal(stacked["fock"][index, :, real_ao:], 0.0)


def test_bucket_pipeline_yields_batched_dicts_of_right_shape(multi_size_qh9_db: Path) -> None:
    """Each datarax bucket pipeline yields stacked, correctly-shaped batches."""
    loaders = create_qh9_loader(
        db_path=multi_size_qh9_db,
        config=QH9Config(batch_size=2, bucket_by="signature", shuffle=False),
    )
    all_buckets = loaders.train + loaders.val + loaders.test
    assert all_buckets  # at least one bucket across the splits
    bucket = next(b for b in all_buckets if b.n_examples >= 1)
    batch = next(iter(bucket.pipeline))
    assert set(batch) == {"atomic_numbers", "positions", "fock", "mask", "molecule_id", "n_ao"}
    n_batch = batch["positions"].shape[0]
    assert batch["positions"].shape == (n_batch, bucket.n_atoms, 3)
    assert batch["fock"].shape == (n_batch, bucket.n_ao, bucket.n_ao)
    assert batch["mask"].shape == (n_batch, bucket.n_ao, bucket.n_ao)
    assert batch["atomic_numbers"].shape == (n_batch, bucket.n_atoms)


def test_loader_buckets_cover_every_molecule(multi_size_qh9_db: Path) -> None:
    """The bucket pipelines partition all six synthetic molecules across splits."""
    loaders = create_qh9_loader(db_path=multi_size_qh9_db, config=QH9Config(bucket_by="signature"))
    assert loaders.n_train + loaders.n_val + loaders.n_test == 6
