r"""Tests for the QHNet block-form QH9 data path.

The block-form loader (:mod:`opifex.data.sources.qh9_blocks`) cuts each
molecule's spherical def2-SVP Fock matrix into the fixed ``(14, 14)`` per-atom
diagonal blocks and per-directed-edge off-diagonal blocks consumed by the
heterogeneous-batch
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`,
then collates a heterogeneous batch into one flat padded concatenation (QHNet
``cut_matrix`` reference ``OpenDFT/QHBench/QH9/datasets.py``; Yu et al. 2023,
arXiv:2306.04922).

These tests build a TINY SYNTHETIC sqlite fixture (NOT real QH9 data -- random
*symmetric* Fock blobs of the correct QH9-native def2-SVP size, mirroring the
real ``(id, N, Z, pos, Ham)`` row schema, exactly as
:mod:`tests.data.sources.test_qh9_source` does) plus an optional real-data smoke
(``limit <= 8``) and assert:

* the block cut round-trips to the original spherical Fock (``1e-10``) for H2O
  and a heavier molecule;
* masks are correct (H-H ``5x5``, C-H ``14x5`` populated entries);
* the padded-concat collate yields the right fixed shapes + pad masks and safe
  in-range padded edges;
* a real-data smoke (``limit 8``) yields finite blocks of the right shapes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from opifex.data.sources.qh9_blocks import (
    BlockBatchConfig,
    collate_block_batch,
    create_qh9_block_loader,
    cut_fock_to_blocks,
    reconstruct_fock_from_blocks,
)
from opifex.data.sources.qh9_source import read_qh9_sqlite
from opifex.neural.quantum.hamiltonian._orbital_layout import FULL_ORBITALS


_REAL_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")

# QH9 element compositions exercised below.
_H2O_ATOMS = np.array([8, 1, 1], dtype=np.int32)  # O + 2 H -> spherical AO 14+5+5
_HCNO_ATOMS = np.array([1, 6, 7, 8], dtype=np.int32)  # heavier, mixed light/heavy
_H2_ATOMS = np.array([1, 1], dtype=np.int32)


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
    """Write a tiny SYNTHETIC QH9-Stable sqlite db (H2O, HCNO, H2) to tmp_path.

    NOT real QH9 data -- random symmetric Fock blobs of the correct QH9-native
    def2-SVP size, mirroring the reference ``(id, N, Z, pos, Ham)`` row schema.
    """
    db_path = tmp_path / "QH9Stable.db"
    specs = [
        (_H2O_ATOMS, 1),
        (_HCNO_ATOMS, 2),
        (_H2_ATOMS, 3),
        (_H2O_ATOMS, 4),
    ]
    rows = [_make_row(index, atoms, seed) for index, (atoms, seed) in enumerate(specs)]
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
        connection.executemany("INSERT INTO data VALUES (?, ?, ?, ?, ?)", rows)
        connection.commit()
    return db_path


# =============================================================================
# Per-molecule block cut: round-trip + masks
# =============================================================================


def test_cut_block_shapes(synthetic_qh9_db: Path) -> None:
    """The cut emits ``(A,14,14)`` diagonal and ``(E,14,14)`` off-diagonal blocks."""
    water = read_qh9_sqlite(synthetic_qh9_db)[0]
    diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(
        water.atomic_numbers, water.fock
    )
    n_atoms = water.n_atoms
    n_edges = n_atoms * (n_atoms - 1)
    assert diag.shape == (n_atoms, FULL_ORBITALS, FULL_ORBITALS)
    assert diag_mask.shape == (n_atoms, FULL_ORBITALS, FULL_ORBITALS)
    assert off.shape == (n_edges, FULL_ORBITALS, FULL_ORBITALS)
    assert off_mask.shape == (n_edges, FULL_ORBITALS, FULL_ORBITALS)
    assert edge_index.shape == (2, n_edges)


def test_cut_roundtrips_to_original_fock_water(synthetic_qh9_db: Path) -> None:
    """Scattering the masked blocks back reconstructs the H2O Fock to 1e-10."""
    water = read_qh9_sqlite(synthetic_qh9_db)[0]
    diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(
        water.atomic_numbers, water.fock
    )
    reconstructed = reconstruct_fock_from_blocks(
        water.atomic_numbers, diag, diag_mask, off, off_mask, edge_index
    )
    residual = float(np.max(np.abs(reconstructed - water.fock)))
    assert residual < 1e-10, f"H2O block-cut round-trip residual {residual:.2e}"


def test_cut_roundtrips_to_original_fock_heavier(synthetic_qh9_db: Path) -> None:
    """Round-trip holds for a heavier mixed light/heavy molecule (HCNO)."""
    hcno = read_qh9_sqlite(synthetic_qh9_db)[1]
    diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(hcno.atomic_numbers, hcno.fock)
    reconstructed = reconstruct_fock_from_blocks(
        hcno.atomic_numbers, diag, diag_mask, off, off_mask, edge_index
    )
    residual = float(np.max(np.abs(reconstructed - hcno.fock)))
    assert residual < 1e-10, f"HCNO block-cut round-trip residual {residual:.2e}"


def test_diagonal_mask_counts_hydrogen_and_heavy(synthetic_qh9_db: Path) -> None:
    """Diagonal masks populate 5x5 for H and 14x14 for heavy atoms."""
    water = read_qh9_sqlite(synthetic_qh9_db)[0]  # atoms [O, H, H]
    _, diag_mask, _, _, _ = cut_fock_to_blocks(water.atomic_numbers, water.fock)
    # atom 0 is oxygen (14x14), atoms 1,2 are hydrogen (5x5).
    assert int(diag_mask[0].sum()) == 14 * 14
    assert int(diag_mask[1].sum()) == 5 * 5
    assert int(diag_mask[2].sum()) == 5 * 5


def test_off_diagonal_mask_counts_hh_and_ch(synthetic_qh9_db: Path) -> None:
    """Off-diagonal masks are H-H 5x5 and C-H 14x5 (row=receiver, col=sender)."""
    hcno = read_qh9_sqlite(synthetic_qh9_db)[1]  # atoms [H, C, N, O]
    _, _, _, off_mask, edge_index = cut_fock_to_blocks(hcno.atomic_numbers, hcno.fock)
    atoms = hcno.atomic_numbers
    # edge_index[0] = receiver (row), edge_index[1] = sender (col).
    for edge in range(edge_index.shape[1]):
        receiver = int(atoms[int(edge_index[0, edge])])
        sender = int(atoms[int(edge_index[1, edge])])
        rows = 5 if receiver == 1 else 14
        cols = 5 if sender == 1 else 14
        assert int(off_mask[edge].sum()) == rows * cols


def test_off_diagonal_blocks_zero_outside_mask(synthetic_qh9_db: Path) -> None:
    """Off-diagonal block entries outside the validity mask are exactly zero."""
    hcno = read_qh9_sqlite(synthetic_qh9_db)[1]
    _, _, off, off_mask, _ = cut_fock_to_blocks(hcno.atomic_numbers, hcno.fock)
    assert np.all(off[~off_mask] == 0.0)


# =============================================================================
# Padded-concat collation
# =============================================================================


def test_collate_fixed_shapes_and_pad_masks(synthetic_qh9_db: Path) -> None:
    """The collate produces fixed padded shapes with correct node/edge pad masks."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    config = BlockBatchConfig(max_atoms=12, max_edges=140)
    batch = collate_block_batch(examples[:3], config)  # H2O(3), HCNO(4), H2(2)

    assert batch["atomic_numbers"].shape == (12,)
    assert batch["positions"].shape == (12, 3)
    assert batch["edge_index"].shape == (2, 140)
    assert batch["node_batch"].shape == (12,)
    assert batch["edge_batch"].shape == (140,)
    assert batch["diagonal_blocks"].shape == (12, FULL_ORBITALS, FULL_ORBITALS)
    assert batch["diagonal_mask"].shape == (12, FULL_ORBITALS, FULL_ORBITALS)
    assert batch["off_diagonal_blocks"].shape == (140, FULL_ORBITALS, FULL_ORBITALS)
    assert batch["off_diagonal_mask"].shape == (140, FULL_ORBITALS, FULL_ORBITALS)
    assert batch["node_pad_mask"].shape == (12,)
    assert batch["edge_pad_mask"].shape == (140,)

    # 3 + 4 + 2 = 9 real atoms; edges = 3*2 + 4*3 + 2*1 = 20 real edges.
    assert int(batch["node_pad_mask"].sum()) == 9
    assert int(batch["edge_pad_mask"].sum()) == 20


def test_collate_node_batch_segments(synthetic_qh9_db: Path) -> None:
    """node_batch assigns contiguous molecule ids; padding gets a sentinel id."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    config = BlockBatchConfig(max_atoms=12, max_edges=140)
    batch = collate_block_batch(examples[:3], config)
    node_batch = np.asarray(batch["node_batch"])
    pad = np.asarray(batch["node_pad_mask"])
    # Real atoms: molecule 0 (3 atoms), molecule 1 (4), molecule 2 (2).
    np.testing.assert_array_equal(node_batch[:3], 0)
    np.testing.assert_array_equal(node_batch[3:7], 1)
    np.testing.assert_array_equal(node_batch[7:9], 2)
    # Padded atoms are masked out.
    assert not pad[9:].any()


def test_collate_edge_index_offset_within_molecule(synthetic_qh9_db: Path) -> None:
    """Real edges connect atoms of the SAME molecule, offset into the flat batch."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    config = BlockBatchConfig(max_atoms=12, max_edges=140)
    batch = collate_block_batch(examples[:3], config)
    edge_index = np.asarray(batch["edge_index"])
    node_batch = np.asarray(batch["node_batch"])
    edge_pad = np.asarray(batch["edge_pad_mask"])
    real = edge_pad
    rows = edge_index[0, real]
    cols = edge_index[1, real]
    np.testing.assert_array_equal(node_batch[rows], node_batch[cols])


def test_collate_padded_edges_point_in_range(synthetic_qh9_db: Path) -> None:
    """Padded edges point at a safe in-range padded atom (never -1 or wrapping)."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    config = BlockBatchConfig(max_atoms=12, max_edges=140)
    batch = collate_block_batch(examples[:3], config)
    edge_index = np.asarray(batch["edge_index"])
    assert edge_index.min() >= 0
    assert edge_index.max() < config.max_atoms
    # Padded edge endpoints must be padded (non-real) atoms.
    node_pad = np.asarray(batch["node_pad_mask"])
    edge_pad = np.asarray(batch["edge_pad_mask"])
    padded_edges = ~edge_pad
    assert not node_pad[edge_index[0, padded_edges]].any()
    assert not node_pad[edge_index[1, padded_edges]].any()


def test_collate_roundtrips_each_molecule(synthetic_qh9_db: Path) -> None:
    """Each molecule's masked blocks in the batch reconstruct its own Fock."""
    examples = read_qh9_sqlite(synthetic_qh9_db)
    config = BlockBatchConfig(max_atoms=12, max_edges=140)
    batch = collate_block_batch(examples[:3], config)
    node_batch = np.asarray(batch["node_batch"])
    edge_batch = np.asarray(batch["edge_batch"])
    node_pad = np.asarray(batch["node_pad_mask"])
    edge_pad = np.asarray(batch["edge_pad_mask"])
    edge_index = np.asarray(batch["edge_index"])
    atoms = np.asarray(batch["atomic_numbers"])

    for mol_id, example in enumerate(examples[:3]):
        node_sel = (node_batch == mol_id) & node_pad
        edge_sel = (edge_batch == mol_id) & edge_pad
        node_global = np.nonzero(node_sel)[0]
        # Remap the molecule's flat indices to local 0..n_atoms-1.
        remap = {int(g): i for i, g in enumerate(node_global)}
        local_edges = np.stack(
            [
                [remap[int(r)] for r in edge_index[0, edge_sel]],
                [remap[int(c)] for c in edge_index[1, edge_sel]],
            ]
        ).astype(np.int64)
        reconstructed = reconstruct_fock_from_blocks(
            atoms[node_sel],
            np.asarray(batch["diagonal_blocks"])[node_sel],
            np.asarray(batch["diagonal_mask"])[node_sel],
            np.asarray(batch["off_diagonal_blocks"])[edge_sel],
            np.asarray(batch["off_diagonal_mask"])[edge_sel],
            local_edges,
        )
        residual = float(np.max(np.abs(reconstructed - example.fock)))
        # The collated batch arrays follow the active JAX precision (float32 under
        # the suite's x64-off default), so this round-trip is float32-bounded; the
        # exact (1e-10) cut correctness is proven on the NumPy path above.
        assert residual < 1e-5, f"molecule {mol_id} round-trip residual {residual:.2e}"


# =============================================================================
# Loader factory + real-data smoke
# =============================================================================


def test_create_block_loader_yields_batch_dicts(synthetic_qh9_db: Path) -> None:
    """The loader factory yields train/val/test iterables of batch dicts."""
    config = BlockBatchConfig(max_atoms=12, max_edges=140, batch_size=2)
    loaders = create_qh9_block_loader(db_path=synthetic_qh9_db, config=config)
    total = 0
    for split in (loaders.train, loaders.val, loaders.test):
        for batch in split:
            assert set(batch) == {
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
            assert batch["positions"].shape == (12, 3)
            total += int(np.asarray(batch["node_pad_mask"]).sum())
    # All 4 molecules' atoms (3+4+2+3 = 12) appear across the splits exactly once.
    assert total == 12


@pytest.mark.skipif(not _REAL_DB.exists(), reason="real QH9Stable.db not present")
def test_real_data_smoke_finite_blocks() -> None:
    """A real-data smoke (limit 8) yields finite blocks of the right shapes."""
    examples = read_qh9_sqlite(_REAL_DB, limit=8)
    assert len(examples) == 8
    total_atoms = sum(ex.n_atoms for ex in examples)
    total_edges = sum(ex.n_atoms * (ex.n_atoms - 1) for ex in examples)
    config = BlockBatchConfig(max_atoms=total_atoms, max_edges=total_edges, batch_size=8)
    for example in examples:
        diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(
            example.atomic_numbers, example.fock
        )
        assert np.all(np.isfinite(diag))
        assert np.all(np.isfinite(off))
        reconstructed = reconstruct_fock_from_blocks(
            example.atomic_numbers, diag, diag_mask, off, off_mask, edge_index
        )
        residual = float(np.max(np.abs(reconstructed - example.fock)))
        assert residual < 1e-10, f"real molecule round-trip residual {residual:.2e}"
    batch = collate_block_batch(examples, config)
    assert batch["diagonal_blocks"].shape == (total_atoms, FULL_ORBITALS, FULL_ORBITALS)
    assert int(np.asarray(batch["node_pad_mask"]).sum()) == total_atoms
    assert np.all(np.isfinite(np.asarray(batch["diagonal_blocks"])))
