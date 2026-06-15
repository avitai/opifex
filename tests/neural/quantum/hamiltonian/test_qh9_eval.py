r"""Tests for the QH9 benchmark evaluation suite.

Gate the ported QH9 evaluation
(:mod:`opifex.neural.quantum.hamiltonian.qh9_eval`):

* **Convention-correctness gate** -- diagonalizing a *ground-truth* QH9 Fock (from
  ``QH9Stable.db``, opifex spherical ordering) against the PySCF def2-SVP overlap
  via :func:`cal_orbital_and_energies` reproduces PySCF's own converged
  B3LYP/def2-SVP orbital energies to chemical accuracy, proving the
  ``back2pyscf`` basis reconciliation and the Löwdin transform are correct.
* **Metric sanity** -- identical Fock gives ε-MAE 0 and ψ-similarity 1.0; a
  perturbed Fock degrades both monotonically.
* **Transform compatibility** -- :func:`cal_orbital_and_energies` is ``jit`` and
  ``vmap`` clean.
* **End-to-end** -- :func:`evaluate_fock` returns the expected keys with finite
  values on a real molecule.

The convention gate and the data-backed tests read a couple of small molecules
from ``/mnt/ssd2/Data/qh9/raw/QH9Stable.db`` and run PySCF B3LYP/def2-SVP; they
are skipped if the database is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import experimental as jax_experimental

from opifex.core.quantum.molecular_system import ANGSTROM_TO_BOHR
from opifex.data.sources.qh9_source import read_qh9_sqlite
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.qh9_eval import (
    cal_orbital_and_energies,
    evaluate_examples,
    evaluate_fock,
    latest_checkpoint,
    occupied_orbital_count,
    orbital_coefficient_similarity,
    orbital_energy_mae,
    overlap_matrix_def2svp,
    predict_fock,
    QH9TestSetMetrics,
    to_pyscf_internal_ordering,
)


_QH9_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")
_CHEMICAL_TOLERANCE_HARTREE = 5e-3  # ~3.6 mHa stored-Fock vs fresh-SCF residual + margin.

requires_qh9_db = pytest.mark.skipif(not _QH9_DB.exists(), reason="QH9Stable.db not available")
requires_pyscf_dft = pytest.mark.skipif(
    importlib.util.find_spec("pyscf.dft") is None,
    reason="pyscf.dft not available",
)


def _first_examples(n: int) -> list:
    """Decode the first ``n`` QH9-Stable molecules (id-sorted)."""
    return list(read_qh9_sqlite(_QH9_DB, limit=n))


# ---------------------------------------------------------------------------
# Convention-correctness gate
# ---------------------------------------------------------------------------
@requires_qh9_db
@requires_pyscf_dft
@pytest.mark.parametrize("index", [0, 1, 2])
def test_ground_truth_fock_matches_pyscf_orbital_energies(index: int) -> None:
    """A ground-truth QH9 Fock + PySCF overlap reproduces PySCF's B3LYP energies.

    This is the convention gate: it proves the opifex-spherical -> PySCF-internal
    reorder (:func:`to_pyscf_internal_ordering`) plus the Löwdin eigensolve
    (:func:`cal_orbital_and_energies`) are correct by matching PySCF's own
    converged ``mo_energy`` for the same geometry/Fock to chemical accuracy.
    """
    from pyscf import dft, gto

    with jax_experimental.enable_x64():
        examples = _first_examples(index + 1)
        example = examples[index]
        atomic_numbers = np.asarray(example.atomic_numbers)
        positions_bohr = np.asarray(example.system.positions, dtype=np.float64)

        overlap = overlap_matrix_def2svp(atomic_numbers, positions_bohr)
        fock_pyscf = to_pyscf_internal_ordering(jnp.asarray(example.fock), atomic_numbers)
        energies, _ = cal_orbital_and_energies(overlap, fock_pyscf)

        positions_angstrom = positions_bohr / ANGSTROM_TO_BOHR
        atom = [[int(z), tuple(p)] for z, p in zip(atomic_numbers, positions_angstrom, strict=True)]
        mol = gto.M(atom=atom, basis="def2svp", unit="ang", verbose=0)
        mean_field = dft.RKS(mol)
        mean_field.xc = "b3lyp"
        mean_field.kernel()

        residual = float(np.max(np.abs(np.asarray(energies) - mean_field.mo_energy)))
    assert residual < _CHEMICAL_TOLERANCE_HARTREE, (
        f"orbital-energy residual {residual} Ha too large"
    )


# ---------------------------------------------------------------------------
# Metric sanity
# ---------------------------------------------------------------------------
def _symmetric_matrix(seed: int, n: int) -> jnp.ndarray:
    """A reproducible symmetric ``(n, n)`` float64 matrix."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    return jnp.asarray(matrix + matrix.T, dtype=jnp.float64)


def test_identical_fock_gives_zero_energy_mae_and_unit_similarity() -> None:
    """Identical Fock matrices give ε-MAE 0 and ψ-similarity 1.0."""
    with jax_experimental.enable_x64():
        overlap = jnp.eye(8, dtype=jnp.float64)
        fock = _symmetric_matrix(0, 8)
        energies, coefficients = cal_orbital_and_energies(overlap, fock)
        mae = orbital_energy_mae(energies, energies)
        similarity = orbital_coefficient_similarity(coefficients, coefficients)
    assert float(mae) == pytest.approx(0.0, abs=1e-12)
    assert float(similarity) == pytest.approx(1.0, abs=1e-10)


def test_perturbed_fock_degrades_metrics_monotonically() -> None:
    """Increasing Fock perturbation monotonically raises ε-MAE and lowers ψ-sim."""
    with jax_experimental.enable_x64():
        overlap = jnp.eye(8, dtype=jnp.float64)
        fock = _symmetric_matrix(0, 8)
        delta = _symmetric_matrix(1, 8)
        base_energies, base_coefficients = cal_orbital_and_energies(overlap, fock)

        maes: list[float] = []
        similarities: list[float] = []
        for scale in (0.0, 0.05, 0.2, 0.5):
            energies, coefficients = cal_orbital_and_energies(overlap, fock + scale * delta)
            maes.append(float(orbital_energy_mae(energies, base_energies)))
            similarities.append(
                float(orbital_coefficient_similarity(coefficients, base_coefficients))
            )

    assert maes == sorted(maes), f"energy MAE not monotonic: {maes}"
    assert similarities == sorted(similarities, reverse=True), (
        f"similarity not monotonic: {similarities}"
    )
    assert similarities[0] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Transform compatibility (jit / vmap)
# ---------------------------------------------------------------------------
def test_cal_orbital_and_energies_jit_matches_eager() -> None:
    """The jitted Löwdin eigensolve matches the eager path."""
    with jax_experimental.enable_x64():
        overlap = _symmetric_matrix(2, 6) + 6.0 * jnp.eye(6, dtype=jnp.float64)
        fock = _symmetric_matrix(3, 6)
        eager_energies, _ = cal_orbital_and_energies(overlap, fock)
        jit_energies, _ = jax.jit(cal_orbital_and_energies)(overlap, fock)
    np.testing.assert_allclose(np.asarray(eager_energies), np.asarray(jit_energies), atol=1e-10)


def test_cal_orbital_and_energies_vmap_batches() -> None:
    """The Löwdin eigensolve vmaps over a batch of (overlap, Fock) pairs."""
    with jax_experimental.enable_x64():
        overlaps = jnp.stack(
            [_symmetric_matrix(s, 5) + 6.0 * jnp.eye(5, dtype=jnp.float64) for s in (10, 11)]
        )
        focks = jnp.stack([_symmetric_matrix(s, 5) for s in (20, 21)])
        energies, coefficients = jax.vmap(cal_orbital_and_energies)(overlaps, focks)
        single_energies, _ = cal_orbital_and_energies(overlaps[0], focks[0])
    assert energies.shape == (2, 5)
    assert coefficients.shape == (2, 5, 5)
    np.testing.assert_allclose(np.asarray(energies[0]), np.asarray(single_energies), atol=1e-10)


# ---------------------------------------------------------------------------
# occupied count
# ---------------------------------------------------------------------------
def test_occupied_orbital_count_is_half_electrons() -> None:
    """Occupied count is ``sum(Z) // 2`` for closed-shell neutral molecules."""
    # CH4: 6 + 4*1 = 10 electrons -> 5 occupied.
    assert occupied_orbital_count(np.array([6, 1, 1, 1, 1], dtype=np.int32)) == 5


# ---------------------------------------------------------------------------
# End-to-end evaluate_fock on a real molecule
# ---------------------------------------------------------------------------
@requires_qh9_db
def test_evaluate_fock_returns_expected_keys_and_finite_values() -> None:
    """:func:`evaluate_fock` returns the documented keys with finite values."""
    expected_keys = {
        "orbital_energy_mae",
        "orbital_energy_mae_occ",
        "coefficient_similarity",
        "homo_lumo_gap_mae",
        "hamiltonian_mae",
    }
    with jax_experimental.enable_x64():
        example = _first_examples(1)[0]
        atomic_numbers = np.asarray(example.atomic_numbers)
        positions = np.asarray(example.system.positions, dtype=np.float64)
        target_fock = jnp.asarray(example.fock)

        predictor = BlockHamiltonianPredictor(config=BlockHamiltonianConfig(), rngs=nnx.Rngs(0))
        predicted_fock = predict_fock(predictor, atomic_numbers, positions)
        metrics = evaluate_fock(
            predicted_fock,
            target_fock,
            atomic_numbers,
            positions,
            int(atomic_numbers.sum()),
        )

    assert set(metrics) == expected_keys
    for key, value in metrics.items():
        assert np.isfinite(float(value)), f"{key} is not finite"


@requires_qh9_db
def test_evaluate_fock_identical_fock_is_perfect() -> None:
    """Comparing a target Fock against itself gives zero ε-MAE and unit ψ-sim."""
    with jax_experimental.enable_x64():
        example = _first_examples(1)[0]
        atomic_numbers = np.asarray(example.atomic_numbers)
        positions = np.asarray(example.system.positions, dtype=np.float64)
        target_fock = jnp.asarray(example.fock)
        metrics = evaluate_fock(
            target_fock,
            target_fock,
            atomic_numbers,
            positions,
            int(atomic_numbers.sum()),
        )
    assert float(metrics["orbital_energy_mae"]) == pytest.approx(0.0, abs=1e-8)
    assert float(metrics["hamiltonian_mae"]) == pytest.approx(0.0, abs=1e-12)
    assert float(metrics["coefficient_similarity"]) == pytest.approx(1.0, abs=1e-6)
    assert float(metrics["homo_lumo_gap_mae"]) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Checkpoint discovery (pure filesystem logic, no DB / GPU)
# ---------------------------------------------------------------------------
class TestLatestCheckpoint:
    def test_picks_highest_epoch_not_mtime(self, tmp_path: Path) -> None:
        """The newest checkpoint is chosen by epoch number, not creation order."""
        for epoch in (3, 12, 7):  # created out of epoch order on purpose
            (tmp_path / f"best_epoch_{epoch}").mkdir()
        latest = latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.name == "best_epoch_12"

    def test_returns_none_when_empty_or_absent(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path) is None
        assert latest_checkpoint(tmp_path / "does_not_exist") is None


# ---------------------------------------------------------------------------
# Test-set aggregation (evaluate_examples) on real molecules
# ---------------------------------------------------------------------------
@requires_qh9_db
@requires_pyscf_dft
def test_evaluate_examples_aggregates_over_molecules() -> None:
    """:func:`evaluate_examples` returns finite, well-formed aggregate metrics."""
    with jax_experimental.enable_x64():
        examples = _first_examples(3)
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
