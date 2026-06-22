"""Tests for the rMD17 dataset loader (built on datarax).

The loader downloads the revised MD17 dataset (rMD17, Christensen & von
Lilienfeld 2020, figshare DOI 10.6084/m9.figshare.12672038) and exposes
batched, shuffled train/validation pipelines via datarax's ``MemorySource``
and ``Pipeline``.

These tests use a tiny SYNTHETIC rMD17-shaped ``.npz`` written into
``tmp_path`` so they require NO network. The real-download path is covered
by a single ``@pytest.mark.slow`` test that is skipped when offline.

Reference for the npz schema (confirmed against the figshare archive):
    keys = coords (n, n_atoms, 3), energies (n,), forces (n, n_atoms, 3),
    nuclear_charges (n_atoms,), plus old_* provenance arrays.
Units: energies in kcal/mol, forces in kcal/mol/Angstrom, coords in Angstrom.
"""

from pathlib import Path

import numpy as np
import pytest

from opifex.data.sources.rmd17_source import (
    create_rmd17_loader,
    KCAL_PER_MOL_IN_MEV,
    parse_rmd17_npz,
    RMD17Config,
    RMD17Data,
)


N_CONFIGS = 12
N_ATOMS = 5


@pytest.fixture
def synthetic_rmd17_npz(tmp_path: Path) -> Path:
    """Write a tiny rMD17-shaped npz (12 configs, 5 atoms) to tmp_path.

    Mirrors the exact key/shape/dtype layout of a real
    ``rmd17_<molecule>.npz`` so the parser and loader exercise the
    production code path without any network access.
    """
    rng = np.random.default_rng(0)
    npz_path = tmp_path / "rmd17_synthetic.npz"
    np.savez(
        npz_path,
        coords=rng.standard_normal((N_CONFIGS, N_ATOMS, 3)).astype(np.float64),
        energies=rng.standard_normal((N_CONFIGS,)).astype(np.float64),
        forces=rng.standard_normal((N_CONFIGS, N_ATOMS, 3)).astype(np.float64),
        nuclear_charges=np.array([6, 1, 8, 1, 6], dtype=np.int64),
        old_indices=np.arange(N_CONFIGS, dtype=np.int64),
        old_energies=rng.standard_normal((N_CONFIGS,)).astype(np.float64),
        old_forces=rng.standard_normal((N_CONFIGS, N_ATOMS, 3)).astype(np.float64),
    )
    return npz_path


# =============================================================================
# Parser: shapes / dtypes / units / atomic numbers
# =============================================================================


def test_parse_returns_rmd17_data_container(synthetic_rmd17_npz: Path) -> None:
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    assert isinstance(parsed, RMD17Data)


def test_parse_shapes(synthetic_rmd17_npz: Path) -> None:
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    assert parsed.positions.shape == (N_CONFIGS, N_ATOMS, 3)
    assert parsed.energy.shape == (N_CONFIGS,)
    assert parsed.forces.shape == (N_CONFIGS, N_ATOMS, 3)
    assert parsed.atomic_numbers.shape == (N_ATOMS,)


def test_parse_dtypes_are_float32_arrays(synthetic_rmd17_npz: Path) -> None:
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    assert parsed.positions.dtype == np.float32
    assert parsed.energy.dtype == np.float32
    assert parsed.forces.dtype == np.float32
    assert parsed.atomic_numbers.dtype == np.int32


def test_parse_float64_preserves_double_precision(synthetic_rmd17_npz: Path) -> None:
    """``dtype=np.float64`` keeps coordinates/energies/forces in double precision."""
    parsed = parse_rmd17_npz(synthetic_rmd17_npz, dtype=np.float64)
    assert parsed.positions.dtype == np.float64
    assert parsed.energy.dtype == np.float64
    assert parsed.forces.dtype == np.float64
    # Nuclear charges stay integral regardless of the float dtype.
    assert parsed.atomic_numbers.dtype == np.int32


def test_parse_atomic_numbers_values(synthetic_rmd17_npz: Path) -> None:
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    np.testing.assert_array_equal(parsed.atomic_numbers, np.array([6, 1, 8, 1, 6]))


def test_parse_records_units_metadata(synthetic_rmd17_npz: Path) -> None:
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    assert parsed.energy_unit == "kcal/mol"
    assert parsed.length_unit == "Angstrom"
    assert parsed.force_unit == "kcal/mol/Angstrom"


def test_kcal_per_mol_conversion_constant() -> None:
    # 1 kcal/mol == 43.364 meV (documented in the loader).
    assert pytest.approx(43.364, abs=1e-3) == KCAL_PER_MOL_IN_MEV


# =============================================================================
# Train/val split: sizes + disjointness
# =============================================================================


def test_split_sizes(synthetic_rmd17_npz: Path) -> None:
    loaders = create_rmd17_loader(
        npz_path=synthetic_rmd17_npz,
        n_train=4,
        n_val=3,
        batch_size=2,
        seed=0,
    )
    assert loaders.n_train == 4
    assert loaders.n_val == 3


def test_split_is_disjoint(synthetic_rmd17_npz: Path) -> None:
    loaders = create_rmd17_loader(
        npz_path=synthetic_rmd17_npz,
        n_train=5,
        n_val=4,
        batch_size=1,
        seed=0,
    )
    overlap = set(loaders.train_indices.tolist()) & set(loaders.val_indices.tolist())
    assert overlap == set()


def test_split_too_large_fails_fast(synthetic_rmd17_npz: Path) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        create_rmd17_loader(
            npz_path=synthetic_rmd17_npz,
            n_train=10,
            n_val=10,  # 20 > 12 available configs
            batch_size=2,
            seed=0,
        )


# =============================================================================
# datarax pipeline: batched dicts of correct shape
# =============================================================================


def test_pipeline_yields_batched_dicts(synthetic_rmd17_npz: Path) -> None:
    loaders = create_rmd17_loader(
        npz_path=synthetic_rmd17_npz,
        n_train=4,
        n_val=2,
        batch_size=2,
        seed=0,
    )
    batch = next(iter(loaders.train))
    assert set(batch).issuperset({"positions", "energy", "forces"})
    assert batch["positions"].shape == (2, N_ATOMS, 3)
    assert batch["energy"].shape == (2,)
    assert batch["forces"].shape == (2, N_ATOMS, 3)


def test_metadata_exposes_atomic_numbers(synthetic_rmd17_npz: Path) -> None:
    loaders = create_rmd17_loader(
        npz_path=synthetic_rmd17_npz,
        n_train=4,
        n_val=2,
        batch_size=2,
        seed=0,
    )
    np.testing.assert_array_equal(loaders.atomic_numbers, np.array([6, 1, 8, 1, 6], dtype=np.int32))
    assert loaders.n_atoms == N_ATOMS


def test_metadata_exposes_normalization_stats(synthetic_rmd17_npz: Path) -> None:
    loaders = create_rmd17_loader(
        npz_path=synthetic_rmd17_npz,
        n_train=6,
        n_val=2,
        batch_size=2,
        seed=0,
    )
    # Stats are computed over the TRAIN split only (no val leakage).
    parsed = parse_rmd17_npz(synthetic_rmd17_npz)
    train_energy = parsed.energy[loaders.train_indices]
    assert loaders.energy_mean == pytest.approx(float(train_energy.mean()), abs=1e-4)
    assert loaders.energy_std == pytest.approx(float(train_energy.std()), abs=1e-4)


# =============================================================================
# Determinism under fixed seed
# =============================================================================


def test_split_is_deterministic_under_fixed_seed(synthetic_rmd17_npz: Path) -> None:
    a = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=5, n_val=3, batch_size=1, seed=7)
    b = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=5, n_val=3, batch_size=1, seed=7)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.val_indices, b.val_indices)


def test_different_seed_changes_split(synthetic_rmd17_npz: Path) -> None:
    a = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=5, n_val=3, batch_size=1, seed=1)
    b = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=5, n_val=3, batch_size=1, seed=2)
    # Different seeds should produce a different train selection.
    assert not np.array_equal(a.train_indices, b.train_indices)


def test_pipeline_batches_deterministic_under_fixed_seed(synthetic_rmd17_npz: Path) -> None:
    a = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=4, n_val=2, batch_size=2, seed=3)
    b = create_rmd17_loader(npz_path=synthetic_rmd17_npz, n_train=4, n_val=2, batch_size=2, seed=3)
    batch_a = next(iter(a.train))
    batch_b = next(iter(b.train))
    np.testing.assert_array_equal(batch_a["positions"], batch_b["positions"])
    np.testing.assert_array_equal(batch_a["energy"], batch_b["energy"])


# =============================================================================
# Config validation
# =============================================================================


def test_unknown_molecule_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown molecule"):
        RMD17Config(molecule="not_a_real_molecule")


def test_known_molecule_is_accepted() -> None:
    config = RMD17Config(molecule="aspirin")
    assert config.molecule == "aspirin"


# =============================================================================
# Real download (slow; skipped when offline or figshare unreachable)
# =============================================================================


@pytest.mark.slow
def test_real_download_smallest_molecule(tmp_path: Path) -> None:
    """Download the smallest rMD17 molecule and build a real pipeline.

    Skipped when the network/figshare is unavailable so the suite stays
    green offline.
    """
    try:
        loaders = create_rmd17_loader(
            molecule="malonaldehyde",  # smallest rMD17 npz (~67 MB)
            data_dir=tmp_path,
            n_train=8,
            n_val=8,
            batch_size=4,
            seed=0,
        )
    except (OSError, RuntimeError) as exc:  # network / figshare failure
        pytest.skip(f"rMD17 download unavailable: {exc}")

    batch = next(iter(loaders.train))
    n_atoms = loaders.n_atoms
    assert batch["positions"].shape == (4, n_atoms, 3)
    assert batch["energy"].shape == (4,)
    assert batch["forces"].shape == (4, n_atoms, 3)
    assert loaders.atomic_numbers.shape == (n_atoms,)
