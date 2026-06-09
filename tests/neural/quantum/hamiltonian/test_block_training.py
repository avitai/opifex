r"""Tests for the QHNet-style segment-batched masked block loss + train step.

The block training harness wires the heterogeneous-batch
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
to the QH9 block-form targets produced by
:mod:`opifex.data.sources.qh9_blocks`. The loss is QHNet's ``criterion``
(reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/main.py``): per-block masked
squared / absolute error summed over the ``(14, 14)`` block dims, segment-summed
to per-molecule, divided by the per-molecule valid-element count, with
``loss = MSE + MAE`` and the reported metric the Hamiltonian MAE in Hartree over
valid (masked) elements. Both the per-element validity mask and the node/edge pad
mask are applied (padded atoms/edges contribute zero).

The gates are:

* **Masked segment correctness** -- on hand-built tiny inputs the per-molecule
  summed error respects the validity mask, the pad mask, and the segment grouping.
* **Overfit / trains** -- a synthetic mixed-composition batch drives the masked
  block loss down (loss + predictor + collation wire together; the masked gradient
  flows through the padded concatenation), and a single *real* QH9 molecule
  overfits ``>= 5x`` -- the strong end-to-end gate that the predicted per-atom /
  per-edge blocks regress against the same edge-indexed QH9 block targets.
* **jit / grad + repeated jitted step** -- the training-loop contract (one compile,
  reused for every batch of the fixed padded shape).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.data.sources.qh9_blocks import (
    BlockBatchConfig,
    collate_block_batch,
    create_qh9_block_loader,
)
from opifex.data.sources.qh9_source import QH9Example
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.block_training import (
    BlockTrainConfig,
    make_block_train_step,
    masked_block_loss,
    predict_blocks,
    qh9_block_loss,
)


_REAL_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")


# Small, fast predictor config (lmax 2, two interactions).
_CONFIG = BlockHamiltonianConfig(
    hidden_irreps="8x0e + 8x1o + 8x2e",
    sh_lmax=2,
    num_interactions=2,
    cutoff=10.0,
    embed_dim=16,
)


def _predictor(seed: int = 0) -> BlockHamiltonianPredictor:
    """Build a small predictor with the shared test config."""
    return BlockHamiltonianPredictor(config=_CONFIG, rngs=nnx.Rngs(seed))


# ---------------------------------------------------------------------------
# (a) masked_block_loss correctness
# ---------------------------------------------------------------------------
def test_masked_block_loss_respects_validity_and_pad_masks() -> None:
    """Validity mask and pad mask both zero out their entries before summing."""
    pred = jnp.ones((3, 2, 2))
    target = jnp.zeros((3, 2, 2))
    # Validity: item 0 uses only its (0,0) slot; items 1,2 use all 4 slots.
    validity = jnp.zeros((3, 2, 2)).at[0, 0, 0].set(1.0)
    validity = validity.at[1].set(1.0).at[2].set(1.0)
    pad = jnp.array([1.0, 1.0, 0.0])  # item 2 is padding -> contributes zero.
    segment = jnp.array([0, 0, 0])

    totals, counts = masked_block_loss(
        pred, target, validity, pad, segment, num_molecules=1, kind="mae"
    )
    # Residual is 1 everywhere; item0 contributes 1 valid slot, item1 contributes
    # 4, item2 is padded out -> total 5, count 5.
    np.testing.assert_allclose(np.asarray(totals), [5.0])
    np.testing.assert_allclose(np.asarray(counts), [5.0])


def test_masked_block_loss_segments_per_molecule() -> None:
    """Segment-sum groups per-block errors into the right molecule bucket."""
    pred = jnp.array([2.0, 3.0, 4.0]).reshape(3, 1, 1)
    target = jnp.zeros((3, 1, 1))
    validity = jnp.ones((3, 1, 1))
    pad = jnp.ones((3,))
    segment = jnp.array([0, 1, 1])

    totals, counts = masked_block_loss(
        pred, target, validity, pad, segment, num_molecules=2, kind="mse"
    )
    # mol0: 2^2 = 4 (1 element); mol1: 3^2 + 4^2 = 25 (2 elements).
    np.testing.assert_allclose(np.asarray(totals), [4.0, 25.0])
    np.testing.assert_allclose(np.asarray(counts), [1.0, 2.0])


def test_qh9_block_loss_combines_diagonal_and_offdiagonal() -> None:
    """The combined loss is MSE + MAE over the joined diagonal+offdiag counts."""
    batch = _synthetic_block_batch()
    predictor = _predictor()
    predictions = predict_blocks(predictor, batch)
    loss, metrics = qh9_block_loss(predictions, batch)
    assert jnp.isfinite(loss)
    assert "hamiltonian_mae" in metrics
    assert "mse" in metrics and "mae" in metrics
    # loss = mse + mae by construction.
    np.testing.assert_allclose(
        float(loss), float(metrics["mse"]) + float(metrics["mae"]), rtol=1e-5
    )
    assert float(metrics["hamiltonian_mae"]) > 0.0


# ---------------------------------------------------------------------------
# Synthetic block batch (real collation path, ragged molecules)
# ---------------------------------------------------------------------------
def _toy_example(molecule_id: int, atomic_numbers: list[int], seed: int) -> QH9Example:
    """Build a tiny QH9Example with a random *symmetric* def2-SVP Fock matrix."""
    rng = np.random.default_rng(seed)
    n_atoms = len(atomic_numbers)
    positions = rng.normal(scale=1.5, size=(n_atoms, 3))
    # AO count: 5 for H, 14 for C/N/O/F (matches ORBITAL_MASK).
    counts = [5 if z == 1 else 14 for z in atomic_numbers]
    n_ao = int(sum(counts))
    raw = rng.normal(scale=0.3, size=(n_ao, n_ao))
    fock = 0.5 * (raw + raw.T)
    system = MolecularSystem(
        atomic_numbers=jnp.asarray(atomic_numbers, dtype=jnp.int32),
        positions=jnp.asarray(positions, dtype=jnp.float64),
        charge=0,
        multiplicity=1,
        basis_set="def2-svp",
    )
    return QH9Example(
        molecule_id=molecule_id,
        system=system,
        fock=fock,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
    )


def _synthetic_block_batch() -> dict[str, jax.Array]:
    """Collate 3 small ragged molecules into one padded block batch."""
    examples = [
        _toy_example(0, [8, 1, 1], seed=1),  # water
        _toy_example(1, [7, 1, 1, 1], seed=2),  # ammonia-like
        _toy_example(2, [6, 1, 1], seed=3),  # CH2-like
    ]
    config = BlockBatchConfig(max_atoms=12, max_edges=40, batch_size=3)
    return collate_block_batch(examples, config)


# ---------------------------------------------------------------------------
# (b) overfit-one-batch (the edge-pairing correctness gate)
# ---------------------------------------------------------------------------
def test_overfit_synthetic_batch_trains() -> None:
    """A padded mixed-composition block batch trains: combined loss decreases.

    On a synthetic ragged batch the repeated jitted step (one compile, reused for
    every step of the fixed padded shape) drives the masked block loss down,
    confirming the loss + predictor + collation wire together and the gradient
    flows through the padded concatenation (padded atoms/edges contribute zero).
    """
    batch = _synthetic_block_batch()
    predictor = _predictor(seed=0)
    optimizer = nnx.Optimizer(predictor, optax.adam(3e-3), wrt=nnx.Param)
    train_step = make_block_train_step(num_molecules=3)

    losses: list[float] = []
    for _ in range(150):
        loss = train_step(predictor, optimizer, batch)
        losses.append(float(loss))

    assert np.isfinite(losses[-1])
    assert losses[-1] < 0.5 * losses[0], f"loss only dropped {losses[0] / losses[-1]:.2f}x"


@pytest.mark.skipif(not _REAL_DB.exists(), reason="real QH9Stable.db not present")
def test_overfit_one_real_molecule_drops_loss() -> None:
    """One real QH9 molecule overfits: combined block loss drops >= 5x.

    The strong integration gate uses a real QH9 Fock target (physically
    structured blocks, not random noise) so the predicted per-atom / per-edge
    blocks must regress against the SAME edge-indexed target block produced by the
    collation; a drop of ``>= 5x`` over a few hundred jitted steps proves the
    masked block loss, the predictor and the QH9 block targets are correctly
    paired and train end-to-end.
    """
    config = BlockBatchConfig(max_atoms=32, max_edges=900, batch_size=1)
    loaders = create_qh9_block_loader(config=config, db_path=_REAL_DB, limit=20)
    batch = loaders.train[0]

    predictor = BlockHamiltonianPredictor(
        config=BlockHamiltonianConfig(
            hidden_irreps="16x0e + 8x1o + 8x2e",
            sh_lmax=2,
            num_interactions=2,
            cutoff=20.0,
            embed_dim=32,
        ),
        rngs=nnx.Rngs(0),
    )
    optimizer = nnx.Optimizer(predictor, optax.adam(2e-3), wrt=nnx.Param)
    train_step = make_block_train_step(num_molecules=1)

    losses: list[float] = []
    for _ in range(400):
        loss = train_step(predictor, optimizer, batch)
        losses.append(float(loss))

    assert np.isfinite(losses[-1])
    assert losses[0] / losses[-1] >= 5.0, f"loss only dropped {losses[0] / losses[-1]:.2f}x"


# ---------------------------------------------------------------------------
# (c) jit / grad + repeated jitted train-step call
# ---------------------------------------------------------------------------
def test_block_loss_grad_is_finite() -> None:
    """nnx.value_and_grad through the block loss yields finite gradients."""
    batch = _synthetic_block_batch()
    predictor = _predictor()

    def loss_fn(module: BlockHamiltonianPredictor) -> jax.Array:
        predictions = predict_blocks(module, batch)
        loss, _ = qh9_block_loss(predictions, batch)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(predictor)
    assert jnp.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_repeated_jitted_train_step_runs() -> None:
    """The jitted train step compiles once and is reusable across calls."""
    batch = _synthetic_block_batch()
    predictor = _predictor()
    optimizer = nnx.Optimizer(predictor, optax.adam(1e-3), wrt=nnx.Param)
    train_step = make_block_train_step(num_molecules=3)

    first = float(train_step(predictor, optimizer, batch))
    second = float(train_step(predictor, optimizer, batch))
    assert np.isfinite(first) and np.isfinite(second)


def test_block_train_config_defaults_match_qhnet() -> None:
    """The frozen config carries the QHNet reference hyper-parameters."""
    config = BlockTrainConfig()
    assert config.learning_rate == 5e-4
    assert config.warmup_steps == 1000
    assert config.total_steps == 300000
    assert config.grad_clip_norm == 5.0
