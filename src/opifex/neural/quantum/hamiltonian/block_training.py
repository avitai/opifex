r"""Per-molecule masked block loss + GPU-fused train driver for the QHNet predictor.

Wires the per-molecule
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
to the QH9 block-form targets produced on device by the Fock operators
(:mod:`opifex.data.sources.qh9_fock_operators`) over a leading-axis padded batch
from :class:`~opifex.data.sources.qh9_padded_source.QH9PaddedSource`, with QHNet's
``criterion`` (Yu et al. 2023, "QHNet", arXiv:2306.04922; reference
``divelab/AIRS`` ``OpenDFT/QHBench/QH9/main.py``): per-block masked squared /
absolute error summed over the ``(14, 14)`` block dims, reduced per molecule,
divided by the per-molecule valid-element count, with ``loss = MSE + MAE`` and the
reported metric the Hamiltonian MAE in Hartree over valid (masked) elements. Both
the per-element validity mask and the node/edge pad mask are applied -- padded
atoms/edges contribute exactly zero.

Each molecule is one padded element on a leading batch axis (not a flat segment
concatenation), so :func:`per_molecule_block_loss` is a plain per-molecule
reduction (:func:`_per_molecule_block_error`) rather than a segment sum.

Edge orientation (the integration-correctness crux)
---------------------------------------------------
The pairing the loss compares is unambiguous: ``predictions[...][e]`` and
``batch[...][e]`` are indexed by the *same* directed edge ``e`` of the *same*
molecule, so each predicted off-diagonal block is regressed against the target
block of exactly its own ``(row, col)`` atom pair regardless of internal wiring.
What the edge orientation controls is the predictor's *own* convention: the QH9
block targets store ``edge_index[:, e] = (receiver, sender)`` (row 0 = block row,
row 1 = block column) whereas :meth:`BlockHamiltonianPredictor.__call__` reads
``edge_index[0]`` as the *sender* and ``edge_index[1]`` as the *receiver* (its
pair feature tensors the *sender* feature with the edge spherical harmonics and
its block head's row axis tracks the *receiver*). :func:`predict_blocks_vmapped`
therefore feeds the predictor the **row-swapped** edge index by default, so the
predictor's *receiver* (block-row) atom is the target's row atom and the predicted
block transforms as ``D^{l_row} B D^{l_col,\top}`` consistently with the target's
``(row, col)`` AO axes -- the orientation the assembly symmetrisation
``H = H~ + H~^T`` needs to reproduce QHNet's off-diagonal law. The orientation
itself is pinned by the predictor's rotational-equivariance test
(``tests/.../test_block_predictor.py``).

Fixed padded shape -> one compile
---------------------------------
:func:`make_fused_block_train_step` returns a single ``nnx.jit`` step over the
fixed per-molecule padded batch (leading molecule axis): the operators decode +
cut the Fock on device (vmapped over the molecule axis Batch-free), the predictor
is vmapped per molecule, and one ``nnx.value_and_grad`` of
:func:`per_molecule_block_loss` plus one ``optimizer.update`` close the step. The
*same* compiled step serves every batch of that shape -- no per-composition
recompile -- threading one ``nnx.Optimizer`` (optax AdamW + warmup-polynomial
schedule + global-norm clip) wrapping the single predictor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import optax
from datarax.core.operator import OperatorModule  # noqa: TC002
from flax import nnx
from jaxtyping import Array, Float, Int  # noqa: TC002

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


def _per_molecule_block_error(
    pred_blocks: Float[Array, "b n 14 14"],
    target_blocks: Float[Array, "b n 14 14"],
    validity_mask: Float[Array, "b n 14 14"],
    pad_mask: Float[Array, "b n"],
    kind: BlockLossKind,
) -> tuple[Float[Array, " b"], Float[Array, " b"]]:
    r"""Per-molecule masked block error and valid-element count (no segment sum).

    Each ``(14, 14)``
    block's squared / absolute residual is masked by the per-element validity mask
    **and** the per-block pad mask, then summed over both the block dims and the
    per-molecule block axis ``n`` (atoms or edges) -- a plain reduction because the
    batch axis ``b`` already separates molecules (each molecule is one padded
    element, not a concatenated segment).

    Args:
        pred_blocks: Predicted blocks ``(b, n, 14, 14)``.
        target_blocks: Target blocks ``(b, n, 14, 14)``.
        validity_mask: ``{0, 1}`` per-element AO validity mask ``(b, n, 14, 14)``.
        pad_mask: ``{0, 1}`` per-block pad mask ``(b, n)`` (``0`` = padded).
        kind: ``"mae"`` (absolute) or ``"mse"`` (squared) residual.

    Returns:
        ``(per_molecule_error, per_molecule_count)``, each shape ``(b,)``.

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
    block_pad = pad_mask.astype(elementwise.dtype)[..., None, None]
    mask = validity_mask.astype(elementwise.dtype) * block_pad
    error = jnp.sum(elementwise * mask, axis=(1, 2, 3))
    count = jnp.sum(mask, axis=(1, 2, 3))
    return error, count


def per_molecule_block_loss(
    predictions: dict[str, Float[Array, "b ... 14 14"]],
    batch: dict[str, Array],
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    r"""Combine the QHNet block loss (``MSE + MAE``) over a per-molecule batch.

    The batch is ``(b, max_atoms, ...)`` / ``(b, max_edges, ...)`` per-molecule
    padded arrays (from
    :class:`~opifex.data.sources.qh9_padded_source.QH9PaddedSource` after the Fock
    operators), not one flat segment-concatenation, so each molecule's masked
    squared / absolute error is a plain per-molecule reduction
    (:func:`_per_molecule_block_error`). Each molecule's combined diagonal +
    off-diagonal error is normalised by its combined valid-element count and
    averaged over molecules.

    Args:
        predictions: ``{"diagonal_blocks" (b, max_atoms, 14, 14),
            "off_diagonal_blocks" (b, max_edges, 14, 14)}`` (per-molecule vmapped
            predictor outputs).
        batch: A per-molecule padded batch dict carrying the operator-produced
            targets/masks and the node/edge pad masks.

    Returns:
        ``(loss, metrics)`` with ``loss = mse + mae`` and ``metrics`` carrying
        ``"mae"``, ``"mse"``, ``"rmse"``, ``"hamiltonian_mae"`` (Hartree) and
        ``"hamiltonian_mae_micro"`` (micro-Hartree).
    """
    diag_mse, diag_count = _per_molecule_block_error(
        predictions["diagonal_blocks"],
        batch["diagonal_blocks"],
        batch["diagonal_mask"],
        batch["node_pad_mask"],
        "mse",
    )
    diag_mae, _ = _per_molecule_block_error(
        predictions["diagonal_blocks"],
        batch["diagonal_blocks"],
        batch["diagonal_mask"],
        batch["node_pad_mask"],
        "mae",
    )
    off_mse, off_count = _per_molecule_block_error(
        predictions["off_diagonal_blocks"],
        batch["off_diagonal_blocks"],
        batch["off_diagonal_mask"],
        batch["edge_pad_mask"],
        "mse",
    )
    off_mae, _ = _per_molecule_block_error(
        predictions["off_diagonal_blocks"],
        batch["off_diagonal_blocks"],
        batch["off_diagonal_mask"],
        batch["edge_pad_mask"],
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


def predict_blocks_vmapped(
    predictor: BlockHamiltonianPredictor,
    batch: dict[str, Array],
    *,
    swap_edges: bool = True,
) -> dict[str, Float[Array, "b ... 14 14"]]:
    """Run the predictor per molecule over a leading-axis padded batch.

    The batch carries a leading molecule axis (``atomic_numbers``
    ``(b, max_atoms)``, ``positions`` ``(b, max_atoms, 3)``, ``edge_index``
    ``(b, 2, max_edges)``), so the single-molecule predictor is
    :func:`nnx.vmap`-ed over that axis. The data path stores
    ``edge_index = (receiver, sender)`` while the predictor reads
    ``(sender, receiver)``; with ``swap_edges`` (default) each molecule's edge
    index is row-swapped before the call (see the module docstring's edge
    orientation note).

    Args:
        predictor: The single-molecule block Hamiltonian predictor.
        batch: A per-molecule padded batch dict (leading molecule axis).
        swap_edges: Whether to present the predictor its native
            ``(sender, receiver)`` edge order. Defaults to ``True``.

    Returns:
        ``{"diagonal_blocks" (b, max_atoms, 14, 14),
        "off_diagonal_blocks" (b, max_edges, 14, 14)}``.
    """
    edge_index = batch["edge_index"]
    if swap_edges:
        edge_index = edge_index[:, ::-1, :]

    @nnx.vmap(in_axes=(None, 0, 0, 0), out_axes=0)
    def _run(
        module: BlockHamiltonianPredictor,
        atomic_numbers: Int[Array, " max_atoms"],
        positions: Float[Array, "max_atoms 3"],
        edges: Int[Array, "2 max_edges"],
    ) -> dict[str, Float[Array, "... 14 14"]]:
        return module(atomic_numbers, positions, edges)

    return _run(predictor, batch["atomic_numbers"], batch["positions"], edge_index)


def make_fused_block_train_step(
    decode_op: OperatorModule,
    cut_op: OperatorModule,
    *,
    num_molecules: int,
    swap_edges: bool = True,
) -> Callable[..., tuple[Float[Array, ""], Float[Array, ""]]]:
    """Build the fused decode + cut + predict + loss + update train step.

    The returned ``nnx.jit`` closure runs, inside one compiled graph over a
    per-molecule padded batch (leading molecule axis):

    1. the Fock spherical decode and block cut operators, vmapped over the
       molecule axis Batch-free via
       :meth:`~datarax.core.operator.OperatorModule._apply_on_raw` (no
       ``apply_batch``, no ``Batch`` object);
    2. the single-molecule predictor vmapped per molecule
       (:func:`predict_blocks_vmapped`);
    3. one ``nnx.value_and_grad`` (``has_aux=True``) of
       :func:`per_molecule_block_loss` against the operator-produced target
       blocks -- a single forward yielding ``(loss, mae)``;
    4. one ``optimizer.update``.

    No per-step host sync (no ``float()`` / ``block_until_ready``) happens here;
    the caller syncs at log cadence. The operators carry no parameters, so the
    optimizer differentiates only the predictor.

    Args:
        decode_op: The :class:`~...qh9_fock_operators.FockSphericalDecodeOperator`.
        cut_op: The :class:`~...qh9_fock_operators.FockBlockCutOperator`.
        num_molecules: The fixed per-batch molecule count (the leading axis size).
        swap_edges: Edge-orientation flag forwarded to
            :func:`predict_blocks_vmapped`.

    Returns:
        A jitted ``(predictor, optimizer, raw_batch) -> (loss, mae)`` step.
    """
    del num_molecules  # The leading molecule axis carries the batch; no segment static needed.

    def _cut_batch(raw_batch: dict[str, Array]) -> dict[str, Array]:
        # Empty per-element state + explicit empty stats: the operators carry no
        # state and need no statistics; passing `stats={}` avoids the flax-0.12
        # `get_statistics().get_value()` path. The framework vmaps `apply` over
        # the molecule axis of `raw_batch` Batch-free.
        decoded, _ = decode_op._apply_on_raw(raw_batch, {}, {})
        cut_batch, _ = cut_op._apply_on_raw(decoded, {}, {})
        return cut_batch

    def loss_fn(
        module: BlockHamiltonianPredictor, cut_batch: dict[str, Array]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        predictions = predict_blocks_vmapped(module, cut_batch, swap_edges=swap_edges)
        loss, metrics = per_molecule_block_loss(predictions, cut_batch)
        return loss, metrics["hamiltonian_mae"]

    @nnx.jit
    def train_step(
        module: BlockHamiltonianPredictor,
        optimizer: nnx.Optimizer,
        raw_batch: dict[str, Array],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        cut_batch = _cut_batch(raw_batch)
        (loss, mae), grads = nnx.value_and_grad(loss_fn, has_aux=True)(module, cut_batch)
        optimizer.update(module, grads)
        return loss, mae

    return train_step


def make_fused_block_eval_step(
    decode_op: OperatorModule,
    cut_op: OperatorModule,
    *,
    swap_edges: bool = True,
) -> Callable[..., Float[Array, ""]]:
    """Build a fused decode + cut + predict Hamiltonian-MAE eval step.

    The evaluation analogue of :func:`make_fused_block_train_step`: it reuses the
    same operators to produce the target blocks on device, runs the per-molecule
    predictor and returns the Hamiltonian MAE (Hartree) without an
    ``optimizer.update``.

    Args:
        decode_op: The Fock spherical-decode operator.
        cut_op: The Fock block-cut operator.
        swap_edges: Edge-orientation flag forwarded to
            :func:`predict_blocks_vmapped`.

    Returns:
        A jitted ``(predictor, raw_batch) -> hamiltonian_mae`` step (Hartree).
    """

    @nnx.jit
    def eval_step(
        module: BlockHamiltonianPredictor, raw_batch: dict[str, Array]
    ) -> Float[Array, ""]:
        decoded, _ = decode_op._apply_on_raw(raw_batch, {}, {})
        cut_batch, _ = cut_op._apply_on_raw(decoded, {}, {})
        predictions = predict_blocks_vmapped(module, cut_batch, swap_edges=swap_edges)
        _, metrics = per_molecule_block_loss(predictions, cut_batch)
        return metrics["hamiltonian_mae"]

    return eval_step


__all__ = [
    "BlockLossKind",
    "BlockTrainConfig",
    "make_fused_block_eval_step",
    "make_fused_block_train_step",
    "per_molecule_block_loss",
    "predict_blocks_vmapped",
]
