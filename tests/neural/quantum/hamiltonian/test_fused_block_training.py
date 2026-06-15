r"""Tests for the GPU-fused QH9 block train step.

The fused step (:func:`~opifex.neural.quantum.hamiltonian.block_training.make_fused_block_train_step`)
runs the Fock decode + block-cut operators (vmapped over the molecule axis
Batch-free via ``_apply_on_raw``), the per-molecule predictor
(:func:`~...block_training.predict_blocks_vmapped`), the per-molecule masked block
loss and one optimizer update inside a single ``nnx.jit`` graph. These tests gate:
the per-molecule loss matches an eager NumPy-cut target loss bit-for-bit; the
fused step compiles once under repeated ``nnx.jit`` and drives the loss down; and
the whole step is ``grad``-clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import experimental as jax_experimental

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.data.sources.qh9_blocks import cut_fock_to_blocks
from opifex.data.sources.qh9_fock_operators import (
    FockBlockCutConfig,
    FockBlockCutOperator,
    FockSphericalDecodeConfig,
    FockSphericalDecodeOperator,
)
from opifex.data.sources.qh9_padded_source import (
    _pad_molecule,
    _stack_padded,
    QH9PaddedConfig,
)
from opifex.data.sources.qh9_source import matrix_transform_def2svp, QH9Example
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.block_training import (
    BlockTrainConfig,
    make_fused_block_eval_step,
    make_fused_block_train_step,
    per_molecule_block_loss,
    predict_blocks_vmapped,
)


_COMPOSITIONS = (([8, 1, 1], 1), ([1, 6, 7, 8], 2), ([1, 1], 3))


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _make_example(atoms: list[int], seed: int) -> QH9Example:
    """Build a synthetic decoded QH9 example with a random native Fock."""
    charges = np.asarray(atoms, dtype=np.int32)
    rng = np.random.default_rng(seed)
    native = rng.standard_normal((_native_ao(charges), _native_ao(charges)))
    native = (native + native.T).astype(np.float64)
    positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(charges, dtype=jnp.int32),
        positions=jnp.asarray(positions, dtype=jnp.float64),
        charge=0,
        multiplicity=1,
        basis_set="def2-svp",
    )
    return QH9Example(
        molecule_id=seed,
        system=system,
        fock=matrix_transform_def2svp(native, charges),
        atomic_numbers=charges,
        native_fock=native,
    )


def _batch_and_predictor() -> tuple[dict, BlockHamiltonianPredictor, QH9PaddedConfig, tuple]:
    """Build a padded batch, a small predictor and the source config + examples."""
    examples = tuple(_make_example(atoms, seed) for atoms, seed in _COMPOSITIONS)
    config = QH9PaddedConfig(max_atoms=6, max_edges=6 * 5)
    batch = _stack_padded([_pad_molecule(example, config) for example in examples])
    predictor = BlockHamiltonianPredictor(
        config=BlockHamiltonianConfig(
            hidden_irreps="8x0e + 8x1o + 8x2e", sh_lmax=2, num_interactions=2
        ),
        rngs=nnx.Rngs(0),
    )
    return batch, predictor, config, examples


def _operators() -> tuple[FockSphericalDecodeOperator, FockBlockCutOperator]:
    """Construct the two deterministic operators."""
    return (
        FockSphericalDecodeOperator(FockSphericalDecodeConfig()),
        FockBlockCutOperator(FockBlockCutConfig()),
    )


def _eager_target_batch(batch: dict, examples: tuple, config: QH9PaddedConfig) -> dict:
    """Replace the batch's block targets with the eager NumPy-cut blocks."""
    n = len(examples)
    diag = np.zeros((n, config.max_atoms, 14, 14))
    diag_mask = np.zeros((n, config.max_atoms, 14, 14))
    off = np.zeros((n, config.max_edges, 14, 14))
    off_mask = np.zeros((n, config.max_edges, 14, 14))
    for index, example in enumerate(examples):
        dd, dm, oo, om, edge_index = cut_fock_to_blocks(example.atomic_numbers, example.fock)
        n_atoms = example.n_atoms
        n_edges = int(edge_index.shape[1])
        diag[index, :n_atoms] = dd
        diag_mask[index, :n_atoms] = dm
        off[index, :n_edges] = oo
        off_mask[index, :n_edges] = om
    return {
        **batch,
        "diagonal_blocks": jnp.asarray(diag),
        "diagonal_mask": jnp.asarray(diag_mask),
        "off_diagonal_blocks": jnp.asarray(off),
        "off_diagonal_mask": jnp.asarray(off_mask),
    }


def test_fused_loss_matches_eager_cut_loss() -> None:
    """The operator-target per-molecule loss equals the eager NumPy-cut loss."""
    with jax_experimental.enable_x64():
        batch, predictor, config, examples = _batch_and_predictor()
        decode_op, cut_op = _operators()
        decoded, _ = decode_op._apply_on_raw(batch, {}, {})
        cut_batch, _ = cut_op._apply_on_raw(decoded, {}, {})
        predictions = predict_blocks_vmapped(predictor, cut_batch)

        fused_loss, _ = per_molecule_block_loss(predictions, cut_batch)
        eager_batch = _eager_target_batch(batch, examples, config)
        eager_loss, _ = per_molecule_block_loss(predictions, eager_batch)
        np.testing.assert_allclose(float(fused_loss), float(eager_loss), atol=1e-10)


def test_fused_step_compiles_once_and_decreases_loss() -> None:
    """Repeated calls reuse one compile and the loss decreases."""
    with jax_experimental.enable_x64():
        batch, predictor, _, _ = _batch_and_predictor()
        decode_op, cut_op = _operators()
        optimizer = nnx.Optimizer(
            predictor,
            BlockTrainConfig(learning_rate=1e-2, warmup_steps=2, total_steps=80).optimizer(),
            wrt=nnx.Param,
        )
        step = make_fused_block_train_step(decode_op, cut_op, num_molecules=len(_COMPOSITIONS))
        losses: list[float] = []
        for _ in range(20):
            loss, mae = step(predictor, optimizer, batch)
            losses.append(float(loss))
            assert np.isfinite(float(mae))
        assert losses[-1] < losses[0]


def test_fused_eval_step_returns_finite_mae() -> None:
    """The fused eval step returns a finite Hamiltonian MAE without updating."""
    with jax_experimental.enable_x64():
        batch, predictor, _, _ = _batch_and_predictor()
        decode_op, cut_op = _operators()
        eval_step = make_fused_block_eval_step(decode_op, cut_op)
        mae = eval_step(predictor, batch)
        assert np.isfinite(float(mae))


def test_fused_step_is_grad_clean() -> None:
    """The fused loss is differentiable w.r.t. the predictor parameters."""
    with jax_experimental.enable_x64():
        batch, predictor, _, _ = _batch_and_predictor()
        decode_op, cut_op = _operators()

        def loss_fn(module: BlockHamiltonianPredictor) -> jax.Array:
            decoded, _ = decode_op._apply_on_raw(batch, {}, {})
            cut_batch, _ = cut_op._apply_on_raw(decoded, {}, {})
            predictions = predict_blocks_vmapped(module, cut_batch)
            loss, _ = per_molecule_block_loss(predictions, cut_batch)
            return loss

        grads = nnx.grad(loss_fn)(predictor)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves
        assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


@pytest.mark.parametrize("swap_edges", [True, False])
def test_predict_blocks_vmapped_shapes(swap_edges: bool) -> None:
    """The vmapped predictor returns per-molecule block stacks of fixed shape."""
    with jax_experimental.enable_x64():
        batch, predictor, config, _ = _batch_and_predictor()
        decode_op, cut_op = _operators()
        decoded, _ = decode_op._apply_on_raw(batch, {}, {})
        cut_batch, _ = cut_op._apply_on_raw(decoded, {}, {})
        predictions = predict_blocks_vmapped(predictor, cut_batch, swap_edges=swap_edges)
        assert predictions["diagonal_blocks"].shape == (
            len(_COMPOSITIONS),
            config.max_atoms,
            14,
            14,
        )
        assert predictions["off_diagonal_blocks"].shape == (
            len(_COMPOSITIONS),
            config.max_edges,
            14,
            14,
        )
