#!/usr/bin/env python
r"""Evaluate a trained QH9 block-Hamiltonian checkpoint on the held-out test split.

Thin CLI over
:func:`opifex.neural.quantum.hamiltonian.qh9_eval.evaluate_qh9_test_set`: it builds
a :class:`~opifex.neural.quantum.hamiltonian.block_predictor.BlockHamiltonianPredictor`
with the *same* config the checkpoint was trained with, restores the best-val orbax
checkpoint, and aggregates the QH9 benchmark metrics over the deterministic
``0.8/0.1/0.1`` test split (the same split the training driver held out, computed
from the id count without decoding the full 130k-row table).

The reported metrics mirror the QH9 / QHNet benchmark (Yu et al. 2023,
arXiv:2306.04922) so a run is directly comparable to the literature (QHNet
Hamiltonian-MAE ~76 µHa, QHNetV2 ~31.5 µHa on QH9-Stable):

* ``hamiltonian_mae`` -- Fock-matrix MAE (reported in µHa),
* ``orbital_energy_mae`` / ``orbital_energy_mae_occ`` -- ε-MAE over all / occupied
  orbitals (µHa),
* ``coefficient_similarity`` -- occupied-orbital ψ-cosine similarity,
* ``homo_lumo_gap_mae`` -- HOMO-LUMO-gap MAE (µHa).

Runs under ``jax_enable_x64`` (the QH9 Fock targets are float64 and the Löwdin
eigensolve needs the precision). Example::

    JAX_ENABLE_X64=1 python scripts/eval_qh9_blocks.py \
        --dataset stable --run-dir /root/results/qh9_run \
        --hidden "128x0e + 128x1o + 128x2e + 128x3o + 128x4e" \
        --sh-lmax 4 --num-interactions 5 --start-refinement-layer 2 \
        --bottleneck-mul 32 --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jax
from flax import nnx

from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.qh9_eval import (
    evaluate_qh9_test_set,
    latest_checkpoint,
)


logger = logging.getLogger("eval_qh9_blocks")

_HARTREE_TO_MICRO_HARTREE = 1e6
"""Hartree -> micro-Hartree, the QH9 benchmark's reporting unit for MAEs."""

_DATASET_DB: dict[str, Path] = {
    "stable": Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db"),
}
"""Default database path per dataset (only QH9-Stable has a held-out random split)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalArgs:
    """Parsed command-line arguments for the checkpoint evaluation driver."""

    dataset: str
    db: Path | None
    run_dir: Path | None
    checkpoint: Path | None
    limit: int | None
    hidden_irreps: str
    sh_lmax: int
    num_interactions: int
    start_refinement_layer: int
    bottleneck_multiplicity: int
    out: Path | None


def _resolve_db_path(args: EvalArgs) -> Path:
    """Return the explicit ``--db`` or the dataset default."""
    if args.db is not None:
        return args.db
    if args.dataset not in _DATASET_DB:
        raise ValueError(f"no default db for dataset {args.dataset!r}; pass --db")
    return _DATASET_DB[args.dataset]


def _resolve_checkpoint(args: EvalArgs) -> Path:
    """Return the explicit ``--checkpoint`` or the newest one under ``--run-dir``."""
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is None:
        raise ValueError("pass either --checkpoint or --run-dir")
    found = latest_checkpoint(args.run_dir / "checkpoints")
    if found is None:
        raise FileNotFoundError(f"no best_epoch_* checkpoint under {args.run_dir}/checkpoints")
    return found


def _build_predictor(args: EvalArgs) -> BlockHamiltonianPredictor:
    """Build the predictor with the config the checkpoint was trained with."""
    config = BlockHamiltonianConfig(
        hidden_irreps=args.hidden_irreps,
        sh_lmax=args.sh_lmax,
        num_interactions=args.num_interactions,
        start_refinement_layer=args.start_refinement_layer,
        bottleneck_multiplicity=args.bottleneck_multiplicity,
    )
    return BlockHamiltonianPredictor(config=config, rngs=nnx.Rngs(0))


def _report(metrics_dict: dict[str, float | int]) -> dict[str, float | int]:
    """Augment the raw metric dict with µHa conversions for the Hartree MAEs."""
    report = dict(metrics_dict)
    for key in (
        "hamiltonian_mae",
        "orbital_energy_mae",
        "orbital_energy_mae_occ",
        "homo_lumo_gap_mae",
    ):
        report[f"{key}_micro_hartree"] = float(metrics_dict[key]) * _HARTREE_TO_MICRO_HARTREE
    return report


def evaluate(args: EvalArgs) -> dict[str, float | int]:
    """Restore the checkpoint and evaluate the QH9 test-set benchmark metrics."""
    db_path = _resolve_db_path(args)
    checkpoint = _resolve_checkpoint(args)
    predictor = _build_predictor(args)
    predictor.eval()
    logger.info("evaluating checkpoint %s on %s test split", checkpoint, args.dataset)
    metrics = evaluate_qh9_test_set(
        predictor, db_path, checkpoint_path=checkpoint, limit=args.limit
    )
    report = _report(metrics.as_dict())
    logger.info(
        "QH9 test (%d mols): H-MAE %.2f µHa | eps-MAE %.2f µHa | eps-MAE(occ) %.2f µHa "
        "| psi-sim %.4f | gap-MAE %.2f µHa",
        report["n_molecules"],
        report["hamiltonian_mae_micro_hartree"],
        report["orbital_energy_mae_micro_hartree"],
        report["orbital_energy_mae_occ_micro_hartree"],
        report["coefficient_similarity"],
        report["homo_lumo_gap_mae_micro_hartree"],
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        logger.info("wrote report to %s", args.out)
    return report


def _parse_args(argv: list[str] | None) -> EvalArgs:
    """Parse command-line arguments into an :class:`EvalArgs`."""
    parser = argparse.ArgumentParser(description="Evaluate a QH9 block-Hamiltonian checkpoint.")
    parser.add_argument("--dataset", default="stable", choices=sorted(_DATASET_DB))
    parser.add_argument(
        "--db", type=Path, default=None, help="Database path (default per dataset)."
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Training run dir (uses its newest checkpoint)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint dir (overrides --run-dir).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap evaluated test molecules.")
    parser.add_argument(
        "--hidden", dest="hidden_irreps", default=BlockHamiltonianConfig.hidden_irreps
    )
    parser.add_argument("--sh-lmax", type=int, default=BlockHamiltonianConfig.sh_lmax)
    parser.add_argument(
        "--num-interactions", type=int, default=BlockHamiltonianConfig.num_interactions
    )
    parser.add_argument(
        "--start-refinement-layer",
        type=int,
        default=BlockHamiltonianConfig.start_refinement_layer,
    )
    parser.add_argument(
        "--bottleneck-mul", type=int, default=BlockHamiltonianConfig.bottleneck_multiplicity
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the metric report JSON here.")
    namespace = parser.parse_args(argv)
    return EvalArgs(
        dataset=namespace.dataset,
        db=namespace.db,
        run_dir=namespace.run_dir,
        checkpoint=namespace.checkpoint,
        limit=namespace.limit,
        hidden_irreps=namespace.hidden_irreps,
        sh_lmax=namespace.sh_lmax,
        num_interactions=namespace.num_interactions,
        start_refinement_layer=namespace.start_refinement_layer,
        bottleneck_multiplicity=namespace.bottleneck_mul,
        out=namespace.out,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging, enable x64 and run the evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jax.config.update("jax_enable_x64", True)
    evaluate(_parse_args(argv))


if __name__ == "__main__":
    main()
