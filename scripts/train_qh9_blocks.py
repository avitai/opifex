#!/usr/bin/env python
r"""Heterogeneous-batch training driver for QH9 block-form Hamiltonian prediction.

Trains a single
:class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
on QH9-Stable (Yu et al. 2023, arXiv:2306.04922) using the QHNet block criterion
(:func:`~opifex.neural.quantum.hamiltonian.block_training.qh9_block_loss`). Unlike
the old per-composition driver (``scripts/train_qh9.py``, which compiled a fresh
jitted step per atomic-number *signature* -- minutes of compile and no batching of
mixed molecules), the block path concatenates many molecules of differing
composition into one flat batch padded to a fixed ``(max_atoms, max_edges)``
shape. That single shape compiles the train step *once*; every subsequent
mixed-composition batch reuses the same compiled step, so the GPU is fed one
large heterogeneous batch per step.

Outputs (under ``--out``): a flushed ``train.log`` (mirrored to stdout) of per-step
and per-epoch train Hamiltonian-MAE (Hartree), a per-``--val-every``-epoch
validation Hamiltonian-MAE (Hartree), orbax checkpoints of the best-val parameters
under ``checkpoints/``, and a ``metrics.json`` of the measured per-epoch metrics.
No metric is fabricated -- every number is measured from a forward pass.

Run ``JAX_ENABLE_X64=1`` for training (the QH9 Fock targets are float64); prefix
with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` to avoid grabbing all GPU memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from opifex.data.sources.qh9_blocks import (
    BlockBatchConfig,
    create_qh9_block_loader,
    QH9BlockLoaders,
)
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.block_training import (
    BlockTrainConfig,
    make_block_eval_step,
    make_block_train_step,
)


logger = logging.getLogger("train_qh9_blocks")

_DEFAULT_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")
_DEFAULT_OUT = Path("/mnt/ssd2/Data/qh9/runs/blocks1")


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
    parser.add_argument("--max-atoms", type=int, default=32, dest="max_atoms")
    parser.add_argument("--max-edges", type=int, default=900, dest="max_edges")
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
    batches: tuple[dict[str, jax.Array], ...],
    predictor: BlockHamiltonianPredictor,
    optimizer: nnx.Optimizer,
    train_step,
    eval_step,
    *,
    epoch: int,
    log_every: int,
) -> _EpochResult:
    """Run one training epoch over the padded block batches.

    The first step pays the one-off JIT compile; every later step of the fixed
    padded shape reuses it. The per-batch train Hamiltonian-MAE (Hartree) is
    measured (not the optimisation loss) so the logged metric is directly
    comparable to QHNet's reported H-MAE.
    """
    losses: list[float] = []
    maes: list[float] = []
    n_molecules = 0
    epoch_start = time.time()
    first_step_seconds = 0.0
    for index, batch in enumerate(batches):
        step_start = time.time()
        loss = train_step(predictor, optimizer, batch)
        mae = eval_step(predictor, batch)
        float_loss = float(loss)
        float_mae = float(mae)
        jax.block_until_ready(mae)
        step_seconds = time.time() - step_start
        if index == 0:
            first_step_seconds = step_seconds
        losses.append(float_loss)
        maes.append(float_mae)
        n_real = int(np.sum(np.asarray(batch["node_pad_mask"]) > 0))
        n_molecules += _molecule_count(batch)
        if index % log_every == 0:
            logger.info(
                "epoch %d step %d/%d  train H-MAE %.6e Ha  (loss %.4e, %d atoms, %.3fs)",
                epoch,
                index,
                len(batches),
                float_mae,
                float_loss,
                n_real,
                step_seconds,
            )
    seconds = time.time() - epoch_start
    return _EpochResult(
        train_mae=float(np.mean(maes)) if maes else float("nan"),
        n_molecules=n_molecules,
        n_batches=len(batches),
        seconds=seconds,
        first_step_seconds=first_step_seconds,
    )


def _evaluate(
    batches: tuple[dict[str, jax.Array], ...],
    predictor: BlockHamiltonianPredictor,
    eval_step,
) -> float | None:
    """Return the molecule-weighted mean validation Hamiltonian-MAE (Hartree)."""
    if not batches:
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
    """Number of real (non-padded) molecules in a padded block batch."""
    node_batch = np.asarray(batch["node_batch"])
    node_pad = np.asarray(batch["node_pad_mask"]) > 0
    real_ids = node_batch[node_pad]
    return int(real_ids.max()) + 1 if real_ids.size else 0


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
    loaders: QH9BlockLoaders,
    predictor: BlockHamiltonianPredictor,
    n_params: int,
) -> None:
    """Log a clear startup summary of the run."""
    n_train_mol = sum(_molecule_count(batch) for batch in loaders.train)
    n_val_mol = sum(_molecule_count(batch) for batch in loaders.val)
    logger.info("QH9 block-form Hamiltonian training driver (heterogeneous batch)")
    logger.info("  database          : %s", args.db)
    logger.info("  limit             : %s", "all" if args.limit is None else args.limit)
    logger.info("  train molecules   : %d  (%d padded batches)", n_train_mol, len(loaders.train))
    logger.info("  val   molecules   : %d  (%d padded batches)", n_val_mol, len(loaders.val))
    logger.info(
        "  padded shape      : %d atoms x %d edges  (per-mol %d/%d x batch %d)",
        args.max_atoms * args.batch_size,
        args.max_edges * args.batch_size,
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
        "  note: the first step pays a single JIT compile for the padded "
        "(max_atoms, max_edges) shape; every later step (any composition) reuses it."
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run(args: TrainArgs) -> dict[str, object]:
    """Execute the full training run and return the metrics record.

    ``--max-atoms`` / ``--max-edges`` are the *per-molecule* maxima (the largest
    QH9 molecule has 29 atoms, hence ``29 * 28 = 812`` directed edges); the
    collator pads the *whole batch* to the running totals, so the padded
    concatenation is sized at ``max_atoms * batch_size`` atoms and
    ``max_edges * batch_size`` edges -- one fixed shape that compiles once.
    """
    config = BlockBatchConfig(
        max_atoms=args.max_atoms * args.batch_size,
        max_edges=args.max_edges * args.batch_size,
        batch_size=args.batch_size,
    )
    loaders = create_qh9_block_loader(config=config, db_path=args.db, limit=args.limit)

    predictor = _build_predictor(args)
    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(predictor, nnx.Param)))
    _startup_summary(args, loaders, predictor, n_params)

    if not loaders.train:
        logger.warning("empty training split (tiny --limit); nothing to train.")
        return {"epochs": [], "best_val_hamiltonian_mae_hartree": None}

    train_config = _train_config(args, steps_per_epoch=len(loaders.train))
    optimizer = nnx.Optimizer(predictor, train_config.optimizer(), wrt=nnx.Param)
    train_step = make_block_train_step(num_molecules=args.batch_size)
    eval_step = make_block_eval_step(num_molecules=args.batch_size)

    checkpoint_dir = args.out / "checkpoints"
    best_val: float | None = None
    epoch_records: list[dict[str, object]] = []
    log_every = max(len(loaders.train) // 10, 1)

    for epoch in range(1, args.epochs + 1):
        result = _train_epoch(
            loaders.train,
            predictor,
            optimizer,
            train_step,
            eval_step,
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
            val_mae = _evaluate(loaders.val, predictor, eval_step)
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
