#!/usr/bin/env python
r"""Heterogeneous-batch training driver for QH9 block-form Hamiltonian prediction.

Trains a single
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
on QH9-Stable (Yu et al. 2023, arXiv:2306.04922) using the QHNet block criterion
(:func:`~opifex.neural.quantum.hamiltonian.block_training.per_molecule_block_loss`).
Each molecule is read by
:class:`~opifex.data.sources.qh9_padded_source.QH9PaddedSource` padded to a fixed
per-molecule ``(max_atoms, max_edges)`` shape (host-side integer index prep only)
and ``--batch-size`` molecules are stacked on a leading axis. The Fock spherical
decode and block cut then run on device as the canonical datarax operators in
:mod:`opifex.data.sources.qh9_fock_operators`, fused into the jitted train step
(:func:`~opifex.neural.quantum.hamiltonian.block_training.make_fused_block_train_step`):
the operators are vmapped over the molecule axis Batch-free, the per-molecule
predictor is vmapped over the batch, and the loss + optimizer update share one
forward inside a single ``nnx.jit`` graph. The fixed shape compiles once, so the
block cut runs on device rather than as a host-side NumPy loop.

Outputs (under ``--out``): a flushed ``train.log`` (mirrored to stdout) of per-step
and per-epoch train Hamiltonian-MAE (Hartree), a per-``--val-every``-epoch
validation Hamiltonian-MAE (Hartree), orbax checkpoints of the best-val parameters
under ``checkpoints/``, and a ``metrics.json`` of the per-epoch metrics (each
number measured from a forward pass).

Run ``JAX_ENABLE_X64=1`` for training (the QH9 Fock targets are float64); prefix
with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` to avoid grabbing all GPU memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from opifex.data.sources.qh9_fock_operators import (
    FockBlockCutConfig,
    FockBlockCutOperator,
    FockSphericalDecodeConfig,
    FockSphericalDecodeOperator,
)
from opifex.data.sources.qh9_padded_source import (
    create_qh9_padded_sources,
    iterate_padded_batches,
    QH9PaddedConfig,
    QH9PaddedSource,
    QH9PaddedSplits,
)
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.block_training import (
    BlockTrainConfig,
    make_fused_block_eval_step,
    make_fused_block_train_step,
)


logger = logging.getLogger("train_qh9_blocks")

_DEFAULT_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")
_DEFAULT_OUT = Path("/mnt/ssd2/Data/qh9/runs/blocks1")


class _BatchSource(Protocol):
    """A re-iterable, sized source of per-molecule padded raw batch dicts."""

    def __iter__(self) -> Iterator[dict[str, jax.Array]]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _PaddedBatches:
    """Re-iterable, sized view of fixed-size molecule batches over a source.

    Wraps :func:`~opifex.data.sources.qh9_padded_source.iterate_padded_batches`
    so each epoch re-iterates the source from the start; ``len`` is the number of
    ``batch_size``-molecule batches (the last batch wraps to a full ``batch_size``,
    so the masked loss ignores the wrapped molecules' padded blocks).
    """

    source: QH9PaddedSource
    batch_size: int

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Yield consecutive fixed-size batches over the source."""
        return iterate_padded_batches(self.source, self.batch_size)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (len(self.source) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrainArgs:
    """Parsed command-line arguments for the block training driver."""

    db: Path
    limit: int | None
    max_atoms: int
    max_edges: int
    batch_size: int
    epochs: int
    learning_rate: float
    hidden_irreps: str
    sh_lmax: int
    num_interactions: int
    out: Path
    val_every: int


def _parse_args(argv: list[str] | None) -> TrainArgs:
    """Parse the command-line arguments into a :class:`TrainArgs`."""
    parser = argparse.ArgumentParser(
        description="Train QH9 block-form Hamiltonian prediction (heterogeneous batch)."
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None, help="Cap decoded molecules (None=all).")
    parser.add_argument("--max-atoms", type=int, default=29, dest="max_atoms")
    parser.add_argument("--max-edges", type=int, default=812, dest="max_edges")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4, dest="learning_rate")
    parser.add_argument("--hidden", type=str, default="64x0e + 32x1o + 16x2e", dest="hidden_irreps")
    parser.add_argument("--sh-lmax", type=int, default=2, dest="sh_lmax")
    parser.add_argument("--num-interactions", type=int, default=3, dest="num_interactions")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--val-every", type=int, default=1, dest="val_every")
    namespace = parser.parse_args(argv)
    return TrainArgs(
        db=namespace.db,
        limit=namespace.limit,
        max_atoms=namespace.max_atoms,
        max_edges=namespace.max_edges,
        batch_size=namespace.batch_size,
        epochs=namespace.epochs,
        learning_rate=namespace.learning_rate,
        hidden_irreps=namespace.hidden_irreps,
        sh_lmax=namespace.sh_lmax,
        num_interactions=namespace.num_interactions,
        out=namespace.out,
        val_every=namespace.val_every,
    )


def _configure_logging(out_dir: Path) -> None:
    """Configure flushed logging to ``{out}/train.log`` and stdout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(out_dir / "train.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Model / optimizer construction
# ---------------------------------------------------------------------------
def _build_predictor(args: TrainArgs) -> BlockHamiltonianPredictor:
    """Build the single shared block predictor from the CLI hyper-parameters."""
    config = BlockHamiltonianConfig(
        hidden_irreps=args.hidden_irreps,
        sh_lmax=args.sh_lmax,
        num_interactions=args.num_interactions,
    )
    return BlockHamiltonianPredictor(config=config, rngs=nnx.Rngs(0))


def _train_config(args: TrainArgs, steps_per_epoch: int) -> BlockTrainConfig:
    """Build the QHNet train config, scaling the schedule horizon to the run.

    The reference schedule decays over 300 000 steps; for a capped run the horizon
    is set to the actual number of optimisation steps so the warmup + polynomial
    decay still spans the whole run (a no-op for the full run if it exceeds the
    reference horizon).
    """
    reference_warmup = BlockTrainConfig().warmup_steps
    total_steps = max(steps_per_epoch * args.epochs, args.epochs, 1)
    warmup = min(reference_warmup, max(total_steps // 10, 1))
    return BlockTrainConfig(
        learning_rate=args.learning_rate,
        warmup_steps=warmup,
        total_steps=max(total_steps, warmup + 1),
    )


# ---------------------------------------------------------------------------
# Train / eval epochs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _EpochResult:
    """Aggregated metrics of one training epoch."""

    train_mae: float
    n_molecules: int
    n_batches: int
    seconds: float
    first_step_seconds: float


def _train_epoch(
    batches: _BatchSource,
    predictor: BlockHamiltonianPredictor,
    optimizer: nnx.Optimizer,
    train_step,
    *,
    epoch: int,
    log_every: int,
) -> _EpochResult:
    """Run one training epoch over the per-molecule padded batches.

    The fused step decodes + cuts the Fock and runs the predictor + loss + update
    in one ``nnx.jit`` graph and returns ``(loss, mae)`` from a single forward, so
    there is no separate eval pass. Host syncs are deferred to the log cadence
    (and epoch end): the per-step ``(loss, mae)`` device arrays are accumulated
    and only converted to Python floats when logged, so the GPU is not stalled by
    a per-step ``float()`` / ``block_until_ready``.
    """
    losses: list[jax.Array] = []
    maes: list[jax.Array] = []
    n_molecules = 0
    epoch_start = time.time()
    first_step_seconds = 0.0
    for index, batch in enumerate(batches):
        step_start = time.time()
        loss, mae = train_step(predictor, optimizer, batch)
        losses.append(loss)
        maes.append(mae)
        n_real = int(np.sum(np.asarray(batch["node_pad_mask"]) > 0))
        n_molecules += _molecule_count(batch)
        if index == 0:
            jax.block_until_ready(mae)  # Bound the one-off compile to step 0's timing.
            first_step_seconds = time.time() - step_start
        if index % log_every == 0:
            logger.info(
                "epoch %d step %d/%d  train H-MAE %.6e Ha  (loss %.4e, %d atoms)",
                epoch,
                index,
                len(batches),
                float(mae),
                float(loss),
                n_real,
            )
    seconds = time.time() - epoch_start
    mean_mae = float(np.mean([float(m) for m in maes])) if maes else float("nan")
    return _EpochResult(
        train_mae=mean_mae,
        n_molecules=n_molecules,
        n_batches=len(batches),
        seconds=seconds,
        first_step_seconds=first_step_seconds,
    )


def _evaluate(
    batches: _BatchSource,
    predictor: BlockHamiltonianPredictor,
    eval_step,
) -> float | None:
    """Return the molecule-weighted mean validation Hamiltonian-MAE (Hartree)."""
    if len(batches) == 0:
        return None
    total = 0.0
    count = 0
    for batch in batches:
        mae = float(eval_step(predictor, batch))
        n = _molecule_count(batch)
        total += mae * n
        count += n
    return total / count if count else None


def _molecule_count(batch: dict[str, jax.Array]) -> int:
    """Number of molecules in a per-molecule padded batch (the leading axis)."""
    return int(np.asarray(batch["node_pad_mask"]).shape[0])


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _save_checkpoint(
    checkpoint_dir: Path, predictor: BlockHamiltonianPredictor, epoch: int
) -> None:
    """Orbax-checkpoint the best-val parameter state at ``epoch``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    params = nnx.state(predictor, nnx.Param)
    pure = nnx.to_pure_dict(params)
    target = checkpoint_dir / f"best_epoch_{epoch}"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(target.absolute(), pure, force=True)


# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------
def _startup_summary(
    args: TrainArgs,
    splits: QH9PaddedSplits,
    predictor: BlockHamiltonianPredictor,
    n_params: int,
) -> None:
    """Log a clear startup summary of the run."""
    logger.info("QH9 block-form Hamiltonian training driver (per-molecule GPU operators)")
    logger.info("  database          : %s", args.db)
    logger.info("  limit             : %s", "all" if args.limit is None else args.limit)
    logger.info("  train molecules   : %d", len(splits.train))
    logger.info("  val   molecules   : %d", len(splits.val))
    logger.info("  test  molecules   : %d", len(splits.test))
    logger.info(
        "  per-mol shape     : %d atoms x %d edges  (batch %d molecules)",
        args.max_atoms,
        args.max_edges,
        args.batch_size,
    )
    logger.info("  batch size        : %d molecules/batch", args.batch_size)
    logger.info("  epochs            : %d  (validate every %d)", args.epochs, args.val_every)
    logger.info("  learning rate     : %g (AdamW, warmup-poly, clip 5.0)", args.learning_rate)
    logger.info("  hidden irreps     : %s", args.hidden_irreps)
    logger.info("  sh_lmax / layers  : %d / %d", args.sh_lmax, args.num_interactions)
    logger.info("  trainable params  : %d", n_params)
    logger.info("  output dir        : %s", args.out)
    logger.info(
        "  note: the fused step decodes + cuts the Fock and runs the predictor + "
        "loss + update on device in one nnx.jit graph; the first step pays a "
        "single compile for the (batch, max_atoms, max_edges) shape and every "
        "later step reuses it."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run(args: TrainArgs) -> dict[str, object]:
    """Execute the full training run and return the metrics record.

    ``--max-atoms`` / ``--max-edges`` are the *per-molecule* maxima (the largest
    QH9 molecule has 29 atoms, hence ``29 * 28 = 812`` directed edges); each
    molecule is padded to that fixed per-molecule shape and ``--batch-size``
    molecules are stacked on a leading axis. The Fock spherical decode and block
    cut run on device as datarax operators, fused into the jitted train step --
    no host-side eager cut.
    """
    config = QH9PaddedConfig(
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        shuffle=True,
    )
    splits = create_qh9_padded_sources(
        config=config, db_path=args.db, limit=args.limit, rngs=nnx.Rngs(0)
    )

    predictor = _build_predictor(args)
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(predictor, nnx.Param)))
    _startup_summary(args, splits, predictor, n_params)

    if len(splits.train) == 0:
        logger.warning("empty training split (tiny --limit); nothing to train.")
        return {"epochs": [], "best_val_hamiltonian_mae_hartree": None}

    train_batches = _PaddedBatches(splits.train, args.batch_size)
    val_batches = _PaddedBatches(splits.val, args.batch_size)
    train_config = _train_config(args, steps_per_epoch=len(train_batches))
    optimizer = nnx.Optimizer(predictor, train_config.optimizer(), wrt=nnx.Param)

    decode_op = FockSphericalDecodeOperator(FockSphericalDecodeConfig())
    cut_op = FockBlockCutOperator(FockBlockCutConfig())
    train_step = make_fused_block_train_step(decode_op, cut_op, num_molecules=args.batch_size)
    eval_step = make_fused_block_eval_step(decode_op, cut_op)

    checkpoint_dir = args.out / "checkpoints"
    best_val: float | None = None
    epoch_records: list[dict[str, object]] = []
    log_every = max(len(train_batches) // 10, 1)

    for epoch in range(1, args.epochs + 1):
        result = _train_epoch(
            train_batches,
            predictor,
            optimizer,
            train_step,
            epoch=epoch,
            log_every=log_every,
        )
        throughput = result.n_molecules / max(result.seconds, 1e-9)
        record: dict[str, object] = {
            "epoch": epoch,
            "train_hamiltonian_mae_hartree": result.train_mae,
            "n_train_molecules": result.n_molecules,
            "n_batches": result.n_batches,
            "seconds": result.seconds,
            "first_step_seconds": result.first_step_seconds,
            "molecules_per_second": throughput,
        }
        logger.info(
            "epoch %d/%d  train H-MAE %.6e Ha  | %d mols, %d batches, %.1fs, %.1f mol/s "
            "(first-step compile %.1fs)",
            epoch,
            args.epochs,
            result.train_mae,
            result.n_molecules,
            result.n_batches,
            result.seconds,
            throughput,
            result.first_step_seconds,
        )

        if epoch % args.val_every == 0:
            val_mae = _evaluate(val_batches, predictor, eval_step)
            if val_mae is None:
                logger.warning("epoch %d  val set is EMPTY (tiny --limit); no val MAE.", epoch)
                record["val_hamiltonian_mae_hartree"] = None
            else:
                record["val_hamiltonian_mae_hartree"] = val_mae
                logger.info("epoch %d  val   H-MAE %.6e Ha", epoch, val_mae)
                if best_val is None or val_mae < best_val:
                    best_val = val_mae
                    _save_checkpoint(checkpoint_dir, predictor, epoch)
                    logger.info("epoch %d  new best val H-MAE -> checkpointed.", epoch)
        epoch_records.append(record)

    return {
        "config": {
            "limit": args.limit,
            "max_atoms": args.max_atoms,
            "max_edges": args.max_edges,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hidden_irreps": args.hidden_irreps,
            "sh_lmax": args.sh_lmax,
            "num_interactions": args.num_interactions,
        },
        "trainable_params": n_params,
        "best_val_hamiltonian_mae_hartree": best_val,
        "epochs": epoch_records,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the QH9 block-form Hamiltonian training driver."""
    args = _parse_args(argv)
    _configure_logging(args.out)
    metrics = _run(args)
    metrics_path = args.out / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("wrote metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
