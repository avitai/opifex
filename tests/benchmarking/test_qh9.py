"""QH9 test-set aggregation of the block Hamiltonian predictor on real molecules."""

import jax
import numpy as np
from flax import nnx
from tests.qh9_support import first_qh9_examples, requires_pyscf_dft, requires_qh9_db

from opifex.benchmarking.qh9 import evaluate_examples, QH9TestSetMetrics
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)


@requires_qh9_db
@requires_pyscf_dft
def test_evaluate_examples_aggregates_over_molecules() -> None:
    """:func:`evaluate_examples` returns finite, well-formed aggregate metrics."""
    with jax.enable_x64(True):
        examples = first_qh9_examples(3)
        predictor = BlockHamiltonianPredictor(config=BlockHamiltonianConfig(), rngs=nnx.Rngs(0))
        metrics = evaluate_examples(predictor, examples)

    assert isinstance(metrics, QH9TestSetMetrics)
    assert metrics.n_molecules == 3
    as_dict = metrics.as_dict()
    for key in (
        "hamiltonian_mae",
        "orbital_energy_mae",
        "orbital_energy_mae_occ",
        "homo_lumo_gap_mae",
        "coefficient_similarity",
    ):
        assert np.isfinite(float(as_dict[key])), f"{key} is not finite"
    # A cosine similarity is bounded; an untrained predictor must still be in range.
    assert -1.0 - 1e-6 <= metrics.coefficient_similarity <= 1.0 + 1e-6
    assert metrics.hamiltonian_mae >= 0.0
