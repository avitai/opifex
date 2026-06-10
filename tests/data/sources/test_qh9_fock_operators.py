r"""Tests for the GPU-resident QH9 Fock decode + block-cut datarax operators.

The two canonical :class:`~datarax.core.operator.OperatorModule` operators
(:mod:`opifex.data.sources.qh9_fock_operators`) replace the eager-NumPy
spherical decode (:func:`~opifex.data.sources.qh9_source.matrix_transform_def2svp`)
and block cut (:func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks`) with
pure *per-molecule* ``apply`` transforms the framework vmaps over the molecule
axis. The hard gate is **bit-for-bit equivalence** (atol ``1e-10``, float64) to
the NumPy reference on real QH9 molecules of different size/composition, plus the
two-operator chain over a small padded batch matching the per-molecule reference,
and ``jit``/``grad``/``vmap`` cleanliness of ``apply``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import experimental as jax_experimental

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.data.sources.qh9_blocks import cut_fock_to_blocks
from opifex.data.sources.qh9_fock_operators import (
    FockBlockCutConfig,
    FockBlockCutOperator,
    FockSphericalDecodeConfig,
    FockSphericalDecodeOperator,
)
from opifex.data.sources.qh9_padded_source import _pad_molecule, _stack_padded, QH9PaddedConfig
from opifex.data.sources.qh9_source import (
    matrix_transform_def2svp,
    QH9Example,
    read_qh9_sqlite,
)


_REAL_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")
_TOL = 1e-10

# Distinct compositions/sizes exercised below.
_COMPOSITIONS = (
    ([8, 1, 1], 1),  # H2O (light + heavy)
    ([1, 6, 7, 8], 2),  # mixed HCNO
    ([1, 1], 3),  # H2 (all light)
    ([6, 6, 8, 1, 1, 1], 4),  # larger
)


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _make_example(atoms: list[int], seed: int) -> tuple[QH9Example, np.ndarray]:
    """Build a synthetic decoded :class:`QH9Example` with a random native Fock."""
    charges = np.asarray(atoms, dtype=np.int32)
    rng = np.random.default_rng(seed)
    native = rng.standard_normal((_native_ao(charges), _native_ao(charges)))
    native = (native + native.T).astype(np.float64)
    spherical = matrix_transform_def2svp(native, charges)
    positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(charges, dtype=jnp.int32),
        positions=jnp.asarray(positions, dtype=jnp.float64),
        charge=0,
        multiplicity=1,
        basis_set="def2-svp",
    )
    example = QH9Example(
        molecule_id=seed,
        system=system,
        fock=spherical,
        atomic_numbers=charges,
        native_fock=native,
    )
    return example, spherical


def _config() -> QH9PaddedConfig:
    """Padding config large enough for every composition exercised here."""
    return QH9PaddedConfig(max_atoms=8, max_edges=8 * 7)


def _operators() -> tuple[FockSphericalDecodeOperator, FockBlockCutOperator]:
    """Construct the two deterministic operators."""
    return (
        FockSphericalDecodeOperator(FockSphericalDecodeConfig()),
        FockBlockCutOperator(FockBlockCutConfig()),
    )


# =============================================================================
# Per-molecule equivalence vs the NumPy reference (bit-for-bit, float64)
# =============================================================================


@pytest.mark.parametrize(("atoms", "seed"), _COMPOSITIONS)
def test_apply_per_molecule_equals_numpy_cut(atoms: list[int], seed: int) -> None:
    """One ``apply`` per molecule reproduces the NumPy decode + cut bit-for-bit."""
    with jax_experimental.enable_x64():
        example, spherical = _make_example(atoms, seed)
        diag, diag_mask, off, off_mask, edge_index = cut_fock_to_blocks(
            example.atomic_numbers, spherical
        )
        n_atoms = len(atoms)
        n_edges = int(edge_index.shape[1])

        padded = {k: jnp.asarray(v) for k, v in _pad_molecule(example, _config()).items()}
        decode_op, cut_op = _operators()
        decoded, _, _ = decode_op.apply(padded, {}, None, stats={})
        cut, _, _ = cut_op.apply(decoded, {}, None, stats={})

        np.testing.assert_allclose(np.asarray(cut["diagonal_blocks"])[:n_atoms], diag, atol=_TOL)
        np.testing.assert_allclose(np.asarray(cut["off_diagonal_blocks"])[:n_edges], off, atol=_TOL)
        np.testing.assert_array_equal(np.asarray(cut["diagonal_mask"])[:n_atoms] > 0, diag_mask)
        np.testing.assert_array_equal(np.asarray(cut["off_diagonal_mask"])[:n_edges] > 0, off_mask)


def test_decode_apply_equals_matrix_transform() -> None:
    """The decode operator reproduces ``matrix_transform_def2svp`` exactly."""
    with jax_experimental.enable_x64():
        example, spherical = _make_example([8, 1, 1], 11)
        n_ao = example.n_ao
        padded = {k: jnp.asarray(v) for k, v in _pad_molecule(example, _config()).items()}
        decode_op, _ = _operators()
        decoded, _, _ = decode_op.apply(padded, {}, None, stats={})
        np.testing.assert_allclose(np.asarray(decoded["fock"])[:n_ao, :n_ao], spherical, atol=_TOL)


# =============================================================================
# Batched chain via _apply_on_raw (Batch-free vmap) == per-molecule reference
# =============================================================================


def test_apply_on_raw_chain_equals_per_molecule_numpy() -> None:
    """The decode->cut chain vmapped over a padded batch matches NumPy per molecule."""
    with jax_experimental.enable_x64():
        examples = tuple(_make_example(atoms, seed)[0] for atoms, seed in _COMPOSITIONS)
        config = _config()
        batch = {
            k: jnp.asarray(v)
            for k, v in _stack_padded([_pad_molecule(e, config) for e in examples]).items()
        }
        decode_op, cut_op = _operators()
        decoded, _ = decode_op._apply_on_raw(batch, {}, {})
        cut, _ = cut_op._apply_on_raw(decoded, {}, {})

        for index, example in enumerate(examples):
            diag, _, off, _, edge_index = cut_fock_to_blocks(example.atomic_numbers, example.fock)
            n_atoms = example.n_atoms
            n_edges = int(edge_index.shape[1])
            np.testing.assert_allclose(
                np.asarray(cut["diagonal_blocks"][index])[:n_atoms], diag, atol=_TOL
            )
            np.testing.assert_allclose(
                np.asarray(cut["off_diagonal_blocks"][index])[:n_edges], off, atol=_TOL
            )


def test_apply_on_raw_adds_block_keys() -> None:
    """The chain adds the four block keys + the decoded fock, preserving inputs."""
    examples = tuple(_make_example(atoms, seed)[0] for atoms, seed in _COMPOSITIONS[:2])
    config = _config()
    batch = {
        k: jnp.asarray(v)
        for k, v in _stack_padded([_pad_molecule(e, config) for e in examples]).items()
    }
    decode_op, cut_op = _operators()
    decoded, _ = decode_op._apply_on_raw(batch, {}, {})
    cut, _ = cut_op._apply_on_raw(decoded, {}, {})
    added = set(cut) - set(batch)
    assert added == {
        "fock",
        "diagonal_blocks",
        "diagonal_mask",
        "off_diagonal_blocks",
        "off_diagonal_mask",
    }


# =============================================================================
# jit / grad / vmap cleanliness of apply
# =============================================================================


def test_apply_is_jit_grad_vmap_clean() -> None:
    """``apply`` survives jit, grad (through the gather) and an explicit vmap."""
    examples = tuple(_make_example(atoms, seed)[0] for atoms, seed in _COMPOSITIONS[:3])
    config = _config()
    batch = {
        k: jnp.asarray(v)
        for k, v in _stack_padded([_pad_molecule(e, config) for e in examples]).items()
    }
    decode_op, cut_op = _operators()

    def _forward(native_fock: jax.Array) -> jax.Array:
        data = {**batch, "native_fock": native_fock}
        decoded, _ = decode_op._apply_on_raw(data, {}, {})
        cut, _ = cut_op._apply_on_raw(decoded, {}, {})
        return jnp.sum(cut["diagonal_blocks"] ** 2)

    jitted = jax.jit(_forward)
    value = jitted(batch["native_fock"])
    grad = jax.grad(jitted)(batch["native_fock"])
    assert np.isfinite(float(value))
    assert grad.shape == batch["native_fock"].shape
    assert np.all(np.isfinite(np.asarray(grad)))


# =============================================================================
# Real QH9 molecules (network-free; skips if the DB is absent)
# =============================================================================


def test_real_qh9_molecules_match_numpy(tmp_path: Path) -> None:
    """On several real QH9 molecules the operators match the NumPy cut to 1e-10."""
    if not _REAL_DB.exists():
        pytest.skip("real QH9-Stable database not present")
    with jax_experimental.enable_x64():
        examples = read_qh9_sqlite(_REAL_DB, limit=8)
        # Pick molecules of differing size/composition.
        chosen = tuple(examples[: min(6, len(examples))])
        max_atoms = max(example.n_atoms for example in chosen)
        config = QH9PaddedConfig(max_atoms=max_atoms, max_edges=max_atoms * (max_atoms - 1))
        batch = {
            k: jnp.asarray(v)
            for k, v in _stack_padded([_pad_molecule(e, config) for e in chosen]).items()
        }
        decode_op, cut_op = _operators()
        decoded, _ = decode_op._apply_on_raw(batch, {}, {})
        cut, _ = cut_op._apply_on_raw(decoded, {}, {})
        for index, example in enumerate(chosen):
            diag, _, off, _, edge_index = cut_fock_to_blocks(example.atomic_numbers, example.fock)
            n_atoms = example.n_atoms
            n_edges = int(edge_index.shape[1])
            np.testing.assert_allclose(
                np.asarray(cut["diagonal_blocks"][index])[:n_atoms], diag, atol=_TOL
            )
            np.testing.assert_allclose(
                np.asarray(cut["off_diagonal_blocks"][index])[:n_edges], off, atol=_TOL
            )


def test_real_db_schema_is_readable() -> None:
    """Sanity: the real DB opens read-only with the expected ``data`` table."""
    if not _REAL_DB.exists():
        pytest.skip("real QH9-Stable database not present")
    with sqlite3.connect(f"file:{_REAL_DB}?mode=ro", uri=True) as connection:
        cursor = connection.execute("SELECT COUNT(*) FROM data")
        assert int(cursor.fetchone()[0]) > 0
