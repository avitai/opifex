r"""Segment-batched masked block loss + train driver for the QHNet block predictor.

Wires the heterogeneous-batch
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
to the QH9 block-form targets of :mod:`opifex.data.sources.qh9_blocks` with
QHNet's ``criterion`` (Yu et al. 2023, "QHNet", arXiv:2306.04922; reference
``divelab/AIRS`` ``OpenDFT/QHBench/QH9/main.py``): per-block masked squared /
absolute error summed over the ``(14, 14)`` block dims, segment-summed to
per-molecule, divided by the per-molecule valid-element count, with
``loss = MSE + MAE`` and the reported metric the Hamiltonian MAE in Hartree over
valid (masked) elements. Both the per-element validity mask and the node/edge pad
mask are applied -- padded atoms/edges contribute exactly zero.

Edge orientation (the integration-correctness crux)
---------------------------------------------------
The pairing that the loss compares is unambiguous: ``predictions[...][e]`` and
``batch[...][e]`` are indexed by the *same* directed edge ``e`` of the *same*
collated batch, so each predicted off-diagonal block is regressed against the
target block of exactly its own ``(row, col)`` atom pair regardless of internal
wiring. What the edge orientation controls is the predictor's *own* convention:
the QH9 block targets store ``edge_index[:, e] = (receiver, sender)`` (row 0 =
block row, row 1 = block column) whereas
:meth:`BlockHamiltonianPredictor.__call__` reads ``edge_index[0]`` as the
*sender* and ``edge_index[1]`` as the *receiver* (its pair feature tensors the
*sender* feature with the edge spherical harmonics and its block head's row axis
tracks the *receiver*). :func:`predict_blocks` therefore feeds the predictor the
**row-swapped** edge index (``edge_index[::-1]``) by default, so the predictor's
*receiver* (block-row) atom is the target's row atom and the predicted block
transforms as ``D^{l_row} B D^{l_col,\top}`` consistently with the target's
``(row, col)`` AO axes -- the orientation needed for the assembly symmetrisation
``H = H~ + H~^T`` to reproduce QHNet's off-diagonal law. The fix lives entirely
in this driver wiring; neither the predictor nor the data path is edited.

A note on the overfit gate: because the predictor emits an *independent* block
for every directed edge, a high-capacity model can drive the per-edge masked loss
down under either orientation on a single batch, so the overfit-one-batch test is
an end-to-end *trains-correctly* gate (the loss, predictor and QH9 block targets
wire together and the masked gradient flows through the padded concatenation), not
an orientation discriminator. The orientation itself is pinned by the predictor's
rotational-equivariance test (``tests/.../test_block_predictor.py``).

Fixed padded shape -> one compile
---------------------------------
:func:`make_block_train_step` returns a single ``nnx.jit`` step closed over no
batch-shape statics: the padded ``(max_atoms, max_edges)`` batch dict drives the
forward, so the *same* compiled step serves every mixed-composition batch of that
shape -- no per-composition recompile. The step threads one ``nnx.Optimizer``
state (optax AdamW + warmup-polynomial schedule + global-norm clip) wrapping the
single predictor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import optax
from flax import nnx
from jax.ops import segment_sum
from jaxtyping import Array, Bool, Float, Int  # noqa: TC002

from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianPredictor,  # noqa: TC001
)


logger = logging.getLogger(__name__)

BlockLossKind = Literal["mae", "mse"]
"""Per-block reduction: mean-absolute (``"mae"``) or mean-squared (``"mse"``)."""

_HARTREE_TO_MICRO_HARTREE: float = 1.0e6
"""Conversion factor for reporting the Hamiltonian MAE in micro-Hartree."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockTrainConfig:
    """QHNet training hyper-parameters for the block Hamiltonian predictor.

    Defaults reproduce the QH9/QHNet reference setup (``OpenDFT/QHBench/QH9``):
    AdamW with ``lr = 5e-4`` and ``betas = (0.99, 0.999)``, a polynomial
    (``power = 1``) decay schedule with a ``1000``-step warmup over ``300000``
    total steps to ``lr_end = 1e-7``, and global-norm gradient clipping at ``5.0``.

    Attributes:
        learning_rate: Peak AdamW learning rate (post-warmup).
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay.
        weight_decay: AdamW decoupled weight decay.
        warmup_steps: Linear warmup steps to the peak learning rate.
        total_steps: Total schedule steps (decay horizon).
        lr_end: Final (floor) learning rate after polynomial decay.
        power: Polynomial-decay power (``1`` = linear decay).
        grad_clip_norm: Global gradient-norm clip threshold.
    """

    learning_rate: float = 5e-4
    beta1: float = 0.99
    beta2: float = 0.999
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    total_steps: int = 300000
    lr_end: float = 1e-7
    power: float = 1.0
    grad_clip_norm: float = 5.0

    def schedule(self) -> optax.Schedule:
        """Return the warmup + polynomial-decay learning-rate schedule.

        Mirrors HuggingFace ``get_polynomial_decay_schedule_with_warmup`` used by
        the QHNet reference: a linear warmup from ``0`` to ``learning_rate`` over
        ``warmup_steps``, then a polynomial decay to ``lr_end`` over the remaining
        ``total_steps - warmup_steps`` steps.
        """
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        warmup = optax.linear_schedule(0.0, self.learning_rate, self.warmup_steps)
        decay = optax.polynomial_schedule(
            init_value=self.learning_rate,
            end_value=self.lr_end,
            power=self.power,
            transition_steps=decay_steps,
        )
        return optax.join_schedules([warmup, decay], boundaries=[self.warmup_steps])

    def optimizer(self) -> optax.GradientTransformation:
        """Return the AdamW + global-norm-clip optax transform for this config."""
        return optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adamw(
                learning_rate=self.schedule(),
                b1=self.beta1,
                b2=self.beta2,
                weight_decay=self.weight_decay,
            ),
        )


def predict_blocks(
    predictor: BlockHamiltonianPredictor,
    batch: dict[str, Array],
    *,
    swap_edges: bool = True,
) -> dict[str, Float[Array, "... 14 14"]]:
    """Run the predictor on a padded block batch in the target's edge orientation.

    The data path stores ``edge_index = (receiver, sender)`` whereas the predictor
    reads ``edge_index = (sender, receiver)``; with ``swap_edges`` (the default)
    the edge index is row-swapped before the call so the predicted
    ``off_diagonal_blocks[e]`` pairs with the *same* ``(row, col)`` atom pair as
    the target block (see the module docstring). Padded atoms (``Z = 0``) and
    padded edges (pointing at a reserved in-range atom) flow through harmlessly --
    the loss masks them out.

    Args:
        predictor: The block Hamiltonian predictor.
        batch: A padded block batch dict (see
            :func:`opifex.data.sources.qh9_blocks.collate_block_batch`).
        swap_edges: Whether to present the predictor its native
            ``(sender, receiver)`` edge order by row-swapping the data's
            ``(receiver, sender)`` index. Defaults to ``True`` (correct pairing).

    Returns:
        ``{"diagonal_blocks": (A, 14, 14), "off_diagonal_blocks": (E, 14, 14)}``.
    """
    edge_index = batch["edge_index"]
    if swap_edges:
        edge_index = edge_index[::-1]
    return predictor(batch["atomic_numbers"], batch["positions"], edge_index)


def masked_block_loss(
    pred_blocks: Float[Array, "n 14 14"],
    target_blocks: Float[Array, "n 14 14"],
    validity_mask: Bool[Array, "n 14 14"] | Float[Array, "n 14 14"],
    pad_mask: Bool[Array, " n"] | Float[Array, " n"],
    segment_ids: Int[Array, " n"],
    num_molecules: int,
    kind: BlockLossKind,
) -> tuple[Float[Array, " num_molecules"], Float[Array, " num_molecules"]]:
    r"""Per-molecule masked block error and valid-element count (QHNet criterion).

    Each ``(14, 14)`` block's squared / absolute residual is masked by the
    per-element validity mask **and** the per-block pad mask (padded atoms/edges
    contribute zero), summed over the block dims, then segment-summed to
    per-molecule totals -- exactly QHNet's ``scatter_sum(..., batch)`` /
    ``scatter_sum(..., edge_batch)``. The matching valid-element counts are
    segment-summed identically so the caller can normalise by the combined
    diagonal + off-diagonal count.

    Args:
        pred_blocks: Predicted blocks ``(n, 14, 14)``.
        target_blocks: Target blocks ``(n, 14, 14)``.
        validity_mask: ``{0, 1}`` per-element AO validity mask ``(n, 14, 14)``.
        pad_mask: ``{0, 1}`` per-block pad mask ``(n,)`` (``0`` = padded).
        segment_ids: Per-block molecule id ``(n,)`` (padded blocks may carry any
            in-range id; their masked contribution is zero).
        num_molecules: Number of molecule segments (the batch size).
        kind: ``"mae"`` (absolute) or ``"mse"`` (squared) residual.

    Returns:
        ``(per_molecule_error, per_molecule_count)``, each shape
        ``(num_molecules,)``.

    Raises:
        ValueError: If ``kind`` is neither ``"mae"`` nor ``"mse"``.
    """
    residual = pred_blocks - target_blocks
    if kind == "mae":
        elementwise = jnp.abs(residual)
    elif kind == "mse":
        elementwise = residual**2
    else:
        raise ValueError(f"loss kind must be 'mae' or 'mse', got {kind!r}.")

    block_pad = pad_mask.astype(elementwise.dtype)[:, None, None]
    mask = validity_mask.astype(elementwise.dtype) * block_pad
    per_block_error = jnp.sum(elementwise * mask, axis=(1, 2))
    per_block_count = jnp.sum(mask, axis=(1, 2))
    segments = jnp.asarray(segment_ids)
    error = segment_sum(per_block_error, segments, num_segments=num_molecules)
    count = segment_sum(per_block_count, segments, num_segments=num_molecules)
    return error, count


def qh9_block_loss(
    predictions: dict[str, Float[Array, "... 14 14"]],
    batch: dict[str, Array],
    num_molecules: int | None = None,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    r"""Combine the QHNet block loss (``MSE + MAE``) + Hamiltonian-MAE metric.

    Computes the masked, segment-summed squared and absolute errors over both the
    diagonal (per-atom, segmented by ``node_batch``) and off-diagonal (per-edge,
    segmented by ``edge_batch``) blocks, normalises each molecule's combined error
    by its combined valid-element count, and averages over molecules -- the QH9
    ``criterion`` (reference ``OpenDFT/QHBench/QH9/main.py``). The training loss is
    ``MSE + MAE``; the reported ``hamiltonian_mae`` is the Hartree MAE over valid
    (masked) elements.

    Args:
        predictions: ``{"diagonal_blocks", "off_diagonal_blocks"}`` from
            :func:`predict_blocks`.
        batch: A padded block batch dict (targets, masks, segment ids, pad masks).
        num_molecules: Number of molecule segments (the batch size). Must be a
            Python static (the segment-sum ``num_segments``); when ``None`` it is
            inferred host-side from ``batch`` (valid only outside ``jit`` -- inside
            a jitted step pass the fixed batch size).

    Returns:
        ``(loss, metrics)`` where ``loss = mse + mae`` and ``metrics`` carries
        ``"mae"``, ``"mse"``, ``"rmse"``, ``"hamiltonian_mae"`` (Hartree) and
        ``"hamiltonian_mae_micro"`` (micro-Hartree).
    """
    if num_molecules is None:
        num_molecules = _segment_count(batch)
    node_segments = jnp.clip(batch["node_batch"], 0, num_molecules - 1)
    edge_segments = jnp.clip(batch["edge_batch"], 0, num_molecules - 1)

    diag_mse, diag_count = masked_block_loss(
        predictions["diagonal_blocks"],
        batch["diagonal_blocks"],
        batch["diagonal_mask"],
        batch["node_pad_mask"],
        node_segments,
        num_molecules,
        "mse",
    )
    diag_mae, _ = masked_block_loss(
        predictions["diagonal_blocks"],
        batch["diagonal_blocks"],
        batch["diagonal_mask"],
        batch["node_pad_mask"],
        node_segments,
        num_molecules,
        "mae",
    )
    off_mse, off_count = masked_block_loss(
        predictions["off_diagonal_blocks"],
        batch["off_diagonal_blocks"],
        batch["off_diagonal_mask"],
        batch["edge_pad_mask"],
        edge_segments,
        num_molecules,
        "mse",
    )
    off_mae, _ = masked_block_loss(
        predictions["off_diagonal_blocks"],
        batch["off_diagonal_blocks"],
        batch["off_diagonal_mask"],
        batch["edge_pad_mask"],
        edge_segments,
        num_molecules,
        "mae",
    )

    total_count = jnp.clip(diag_count + off_count, a_min=1.0)
    valid_molecule = (diag_count + off_count) > 0
    denom = jnp.clip(jnp.sum(valid_molecule.astype(total_count.dtype)), a_min=1.0)

    per_molecule_mae = (diag_mae + off_mae) / total_count
    per_molecule_mse = (diag_mse + off_mse) / total_count
    mae = jnp.sum(per_molecule_mae * valid_molecule) / denom
    mse = jnp.sum(per_molecule_mse * valid_molecule) / denom
    loss = mse + mae

    metrics: dict[str, Float[Array, ""]] = {
        "mae": mae,
        "mse": mse,
        "rmse": jnp.sqrt(mse),
        "hamiltonian_mae": mae,
        "hamiltonian_mae_micro": mae * _HARTREE_TO_MICRO_HARTREE,
    }
    return loss, metrics


def _segment_count(batch: dict[str, Array]) -> int:
    """Return the number of molecule segments in a padded block batch.

    Uses the maximum node-segment id over real (non-padded) atoms; this is a
    Python int (host side, the batch dict is materialised), so it stays a static
    for the segment-sum ``num_segments``.
    """
    node_batch = batch["node_batch"]
    node_pad = batch["node_pad_mask"].astype(bool)
    real_ids = jnp.where(node_pad, node_batch, -1)
    return int(jnp.max(real_ids)) + 1


def make_block_train_step(*, num_molecules: int, swap_edges: bool = True):
    """Build the jitted block train step (one compile per fixed padded shape).

    The returned ``nnx.jit`` closure runs one ``nnx.value_and_grad`` of
    :func:`qh9_block_loss` and one ``optimizer.update`` over a padded block batch.
    Nothing batch-shape-specific is traced, so the *same* compiled step serves
    every mixed-composition batch of the configured ``(max_atoms, max_edges)``
    shape with a fixed ``num_molecules`` segment count -- the whole point of the
    block redesign.

    Args:
        num_molecules: The fixed per-batch molecule count (the batch size); closed
            over as the static segment-sum ``num_segments``.
        swap_edges: Whether :func:`predict_blocks` presents the predictor its
            native ``(sender, receiver)`` edge order (correct pairing; default).

    Returns:
        A jitted ``(predictor, optimizer, batch) -> loss`` step.
    """

    def loss_fn(module: BlockHamiltonianPredictor, batch: dict[str, Array]) -> Float[Array, ""]:
        predictions = predict_blocks(module, batch, swap_edges=swap_edges)
        loss, _ = qh9_block_loss(predictions, batch, num_molecules)
        return loss

    @nnx.jit
    def train_step(
        module: BlockHamiltonianPredictor,
        optimizer: nnx.Optimizer,
        batch: dict[str, Array],
    ) -> Float[Array, ""]:
        loss, grads = nnx.value_and_grad(loss_fn)(module, batch)
        optimizer.update(module, grads)
        return loss

    return train_step


def make_block_eval_step(*, num_molecules: int, swap_edges: bool = True):
    """Build a jitted Hamiltonian-MAE (Hartree) evaluation step for one batch.

    Args:
        num_molecules: The fixed per-batch molecule count (segment-sum static).
        swap_edges: Edge-orientation flag forwarded to :func:`predict_blocks`.

    Returns:
        A jitted ``(predictor, batch) -> hamiltonian_mae`` step (Hartree).
    """

    @nnx.jit
    def eval_step(module: BlockHamiltonianPredictor, batch: dict[str, Array]) -> Float[Array, ""]:
        predictions = predict_blocks(module, batch, swap_edges=swap_edges)
        _, metrics = qh9_block_loss(predictions, batch, num_molecules)
        return metrics["hamiltonian_mae"]

    return eval_step


__all__ = [
    "BlockLossKind",
    "BlockTrainConfig",
    "make_block_eval_step",
    "make_block_train_step",
    "masked_block_loss",
    "predict_blocks",
    "qh9_block_loss",
]
