r"""QH9 test-split evaluation of a block Hamiltonian predictor.

Aggregates the per-molecule Fock metrics of
:func:`~opifex.neural.quantum.hamiltonian.qh9_eval.evaluate_fock` over decoded QH9-Stable
molecules and over the benchmark's test split (Yu et al. 2023, "QH9", NeurIPS 2023
Datasets and Benchmarks, arXiv:2306.09549). It joins the model to the dataset, so it sits
above both.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from opifex.data.sources.qh9_source import QH9Example, read_qh9_test_split
from opifex.neural.quantum.hamiltonian.block_predictor import BlockHamiltonianPredictor
from opifex.neural.quantum.hamiltonian.qh9_eval import (
    evaluate_fock,
    FOCK_METRIC_KEYS,
    load_predictor_checkpoint,
    predict_fock,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class QH9TestSetMetrics:
    """Aggregated QH9 test-set metrics over the evaluated molecules.

    Attributes:
        n_molecules: Number of molecules evaluated.
        orbital_energy_mae: Mean ε-MAE over all orbitals (Hartree).
        orbital_energy_mae_occ: Mean ε-MAE over occupied orbitals (Hartree).
        coefficient_similarity: Mean occupied-orbital ψ-cosine similarity.
        homo_lumo_gap_mae: Mean HOMO-LUMO-gap MAE (Hartree).
        hamiltonian_mae: Mean Fock-matrix MAE (Hartree).
    """

    n_molecules: int
    orbital_energy_mae: float
    orbital_energy_mae_occ: float
    coefficient_similarity: float
    homo_lumo_gap_mae: float
    hamiltonian_mae: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain ``dict`` (for JSON/logging)."""
        return {
            "n_molecules": self.n_molecules,
            "orbital_energy_mae": self.orbital_energy_mae,
            "orbital_energy_mae_occ": self.orbital_energy_mae_occ,
            "coefficient_similarity": self.coefficient_similarity,
            "homo_lumo_gap_mae": self.homo_lumo_gap_mae,
            "hamiltonian_mae": self.hamiltonian_mae,
        }


def evaluate_examples(
    predictor: BlockHamiltonianPredictor,
    examples: Iterable[QH9Example],
) -> QH9TestSetMetrics:
    """Aggregate :func:`evaluate_fock` over decoded QH9 examples.

    For each example the predicted Fock is assembled from the predictor
    (:func:`predict_fock`) and compared with the example's target Fock; the
    per-molecule metrics are averaged (unweighted) over the molecules.

    Args:
        predictor: The block Hamiltonian predictor (trained or fresh).
        examples: Decoded :class:`~opifex.data.sources.qh9_source.QH9Example`
            records to evaluate.

    Returns:
        The aggregated :class:`QH9TestSetMetrics`.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    predictor.eval()  # Inference: switch to eval mode (canonical NNX boundary).
    totals = dict.fromkeys(FOCK_METRIC_KEYS, 0.0)
    count = 0
    for example in examples:
        atomic_numbers = np.asarray(example.atomic_numbers)
        positions = np.asarray(example.system.positions, dtype=np.float64)
        predicted_fock = predict_fock(predictor, atomic_numbers, positions)
        metrics = evaluate_fock(
            predicted_fock,
            jnp.asarray(example.fock),
            atomic_numbers,
            positions,
            int(atomic_numbers.sum()),
        )
        for key in FOCK_METRIC_KEYS:
            totals[key] += float(metrics[key])
        count += 1

    if count == 0:
        raise ValueError("no examples to evaluate.")
    return QH9TestSetMetrics(
        n_molecules=count,
        orbital_energy_mae=totals["orbital_energy_mae"] / count,
        orbital_energy_mae_occ=totals["orbital_energy_mae_occ"] / count,
        coefficient_similarity=totals["coefficient_similarity"] / count,
        homo_lumo_gap_mae=totals["homo_lumo_gap_mae"] / count,
        hamiltonian_mae=totals["hamiltonian_mae"] / count,
    )


def evaluate_qh9_test_set(
    predictor: BlockHamiltonianPredictor,
    db_path: Path,
    *,
    checkpoint_path: Path | None = None,
    limit: int | None = None,
) -> QH9TestSetMetrics:
    r"""Evaluate the QH9 benchmark metrics over the QH9-Stable test split.

    Reads only the deterministic ``0.8/0.1/0.1`` test-split molecules
    (:func:`~opifex.data.sources.qh9_source.read_qh9_test_split`, which computes the
    split from the cheap id count and decodes the test subset lazily -- no full
    130k-molecule decode), optionally restores a best-val orbax checkpoint into
    ``predictor`` (:func:`load_predictor_checkpoint`), then aggregates
    :func:`evaluate_fock` over the test molecules (:func:`evaluate_examples`).

    Args:
        predictor: The block Hamiltonian predictor, built with the config matching
            ``checkpoint_path`` when one is given.
        db_path: Path to ``QH9Stable.db``.
        checkpoint_path: Optional best-val checkpoint to restore before evaluating
            (e.g. ``<run>/checkpoints/best_epoch_N``).
        limit: Optional cap on the number of *test-split* molecules evaluated (the
            split itself is computed over the full database for fidelity).

    Returns:
        The aggregated :class:`QH9TestSetMetrics` over the evaluated test molecules.
    """
    if checkpoint_path is not None:
        predictor = load_predictor_checkpoint(predictor, checkpoint_path)

    test_examples = read_qh9_test_split(db_path, limit=limit)
    logger.info("Evaluating QH9 test set over %d molecules", len(test_examples))
    return evaluate_examples(predictor, test_examples)


__all__ = ["QH9TestSetMetrics", "evaluate_examples", "evaluate_qh9_test_set"]
