r"""QH9 benchmark evaluation: orbital-energy / coefficient / gap metrics + Fock MAE.

Ports the standardized QH9 (Yu et al. 2023, "QH9", arXiv:2306.04922) evaluation of
a predicted DFT Fock matrix into native JAX, faithful to the reference
``divelab/AIRS`` ``OpenDFT/QHBench/QH9/test.py``:

* :func:`cal_orbital_and_energies` -- the Löwdin (symmetric) orthogonalization
  eigensolve ``F -> (orbital_energies, orbital_coefficients)`` given the AO
  overlap ``S`` (reference ``test.py`` lines 112-120). ``jit``/``vmap`` clean.
* :func:`orbital_energy_mae`, :func:`orbital_coefficient_similarity`,
  :func:`homo_lumo_gap` -- the ε-MAE, sign-invariant per-orbital ψ-cosine
  similarity (reference ``test.py`` lines 57-59) and HOMO/LUMO gap.
* :func:`evaluate_fock` -- a single molecule's metric dict (all + occupied ε-MAE,
  occupied ψ-similarity, HOMO-LUMO-gap MAE, Hamiltonian MAE).
* :func:`evaluate_qh9_test_set` -- aggregate :func:`evaluate_fock` over the QH9
  test split, assembling each predicted Fock via the block predictor and reusing
  the source decode for the target Fock; optionally loads a best-val orbax
  checkpoint of a trained predictor.

The overlap matrix and the basis convention
-------------------------------------------
The AO overlap ``S`` is computed host-side with PySCF (an eval-only path; opifex
already depends on ``pyscf>=2.3.0`` and QH9's reference uses it) by building the
molecule at the QH9 geometry (positions in Bohr in opifex's
:class:`~opifex.core.quantum.molecular_system.MolecularSystem`) and reading
``mol.intor('int1e_ovlp_sph')`` (:func:`overlap_matrix_def2svp`, cached per
geometry).

PySCF's spherical AO ordering (``int1e_ovlp_sph``) is the QH9 ``back2pyscf``
convention -- it differs from opifex's stored Fock ordering (the QH9
``pyscf_def2svp`` convention produced by
:func:`~opifex.data.sources.qh9_source.matrix_transform_def2svp`) **only** in the
within-``p``-shell component order (``pyscf_def2svp`` ``p = [1, 2, 0]`` vs.
``back2pyscf`` ``p = [2, 0, 1]``). The reference ``test.py`` reconciles them by
applying ``matrix_transform(..., convention='back2pyscf')`` to the (already
spherical) data/predicted Fock *before* pairing it with the raw PySCF overlap
(``test.py`` lines 149-167); :func:`to_pyscf_internal_ordering` ports exactly that
transform, deriving its per-AO permutation from the **same** convention tables
:func:`~opifex.data.sources.qh9_source.def2svp_decode_indices` uses (DRY), with
only the ``p`` component map overridden to the ``back2pyscf`` order. With this
reorder, diagonalizing a ground-truth QH9 Fock against the PySCF overlap
reproduces PySCF's own converged B3LYP/def2-SVP orbital energies to chemical
accuracy (~3.6 mHa, the QH9-stored-Fock vs. fresh-SCF grid/convergence residual),
which is the convention-correctness gate exercised by the tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence  # noqa: TC003
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path  # noqa: TC003

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pyscf
from flax import nnx
from jaxtyping import Array, Float, Int  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002

from opifex.data.sources.qh9_source import (
    _DEF2SVP_ATOM_TO_ORBITALS,
    _DEF2SVP_ORBITAL_ORDER,
    _DEF2SVP_ORBITAL_SIGN,
    QH9Example,
    read_qh9_sqlite,
)
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianPredictor,  # noqa: TC001
)


logger = logging.getLogger(__name__)

_OVERLAP_EIGENVALUE_FLOOR: float = 1e-8
"""Floor on overlap eigenvalues before the inverse-sqrt (reference ``test.py``)."""

# back2pyscf within-shell component order: identical to the QH9 ``pyscf_def2svp``
# convention except the ``p`` components, which PySCF's spherical AO layout
# (``int1e_ovlp_sph``) orders as ``[2, 0, 1]`` rather than ``[1, 2, 0]`` (reference
# ``convention_dict['back2pyscf']``). Reusing the source's atom/shell-order and
# sign tables keeps the permutation derivation DRY.
_BACK2PYSCF_ORBITAL_IDX: dict[str, list[int]] = {
    "s": [0],
    "p": [2, 0, 1],
    "d": [0, 1, 2, 3, 4],
}


def _back2pyscf_indices(
    atomic_numbers: Sequence[int] | Int[NDArray[np.int32], " n_atoms"],
) -> tuple[Int[NDArray[np.int64], " n_ao"], Int[NDArray[np.int64], " n_ao"]]:
    r"""Return the opifex-spherical -> PySCF-internal AO permutation and signs.

    Replays the exact reference ``matrix_transform`` index/sign construction
    (``OpenDFT/QHBench/QH9/test.py``) with the ``back2pyscf`` convention, reusing
    the source's shared shell-order/sign tables and overriding only the
    within-shell component map (``_BACK2PYSCF_ORBITAL_IDX``). The returned
    ``(indices, signs)`` define the symmetric congruence applied to a spherical
    Fock matrix in :func:`to_pyscf_internal_ordering`.

    Args:
        atomic_numbers: Nuclear charges of the molecule (H, C, N, O, F only).

    Returns:
        ``(indices, signs)`` each shape ``(n_ao,)`` (``int64``).
    """
    orbitals = ""
    orbitals_order: list[int] = []
    for atomic_number in atomic_numbers:
        offset = len(orbitals_order)
        orbitals += _DEF2SVP_ATOM_TO_ORBITALS[int(atomic_number)]
        orbitals_order += [idx + offset for idx in _DEF2SVP_ORBITAL_ORDER[int(atomic_number)]]

    transform_indices: list[NDArray[np.int64]] = []
    transform_signs: list[NDArray[np.int64]] = []
    for orbital in orbitals:
        offset = sum(len(block) for block in transform_indices)
        transform_indices.append(
            np.asarray(_BACK2PYSCF_ORBITAL_IDX[orbital], dtype=np.int64) + offset
        )
        transform_signs.append(np.asarray(_DEF2SVP_ORBITAL_SIGN[orbital], dtype=np.int64))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    indices = np.concatenate(transform_indices).astype(np.int64)
    signs = np.concatenate(transform_signs).astype(np.int64)
    return indices, signs


def to_pyscf_internal_ordering(
    fock: Float[Array, "n_ao n_ao"],
    atomic_numbers: Sequence[int] | Int[NDArray[np.int32], " n_atoms"],
) -> Float[Array, "n_ao n_ao"]:
    r"""Reorder an opifex-spherical Fock into PySCF's internal spherical AO order.

    Ports the reference ``matrix_transform(..., convention='back2pyscf')`` applied
    to the spherical data/predicted Fock before pairing it with the PySCF overlap
    (``test.py`` lines 149-167). With ``(I, s)`` from :func:`_back2pyscf_indices`
    this is the symmetric congruence ``F'[i, j] = F[I[i], I[j]] * s[i] * s[j]``,
    aligning ``F`` with ``mol.intor('int1e_ovlp_sph')`` so
    :func:`cal_orbital_and_energies` is in one consistent basis.

    Args:
        fock: Fock matrix in opifex spherical (``pyscf_def2svp``) AO ordering.
        atomic_numbers: Nuclear charges of the molecule (H, C, N, O, F only).

    Returns:
        The Fock matrix in PySCF's internal spherical AO ordering.
    """
    indices, signs = _back2pyscf_indices(atomic_numbers)
    index = jnp.asarray(indices)
    sign = jnp.asarray(signs, dtype=fock.dtype)
    reordered = fock[..., index, :][..., :, index]
    return reordered * sign[:, None] * sign[None, :]


def cal_orbital_and_energies(
    overlap: Float[Array, "n_ao n_ao"],
    hamiltonian: Float[Array, "n_ao n_ao"],
) -> tuple[Float[Array, " n_ao"], Float[Array, "n_ao n_ao"]]:
    r"""Solve the generalized eigenproblem ``F C = S C diag(eps)`` via Löwdin.

    Faithful JAX port of the reference ``cal_orbital_and_energies``
    (``OpenDFT/QHBench/QH9/test.py`` lines 112-120): symmetric (Löwdin)
    orthogonalization ``S^{-1/2} = U diag(1/sqrt(s)) U^T`` (built as
    ``U / sqrt(s)``), transform ``Fs = (S^{-1/2})^T F S^{-1/2}``, eigendecompose
    ``Fs -> (orbital_energies, C_orth)`` and rotate the coefficients back to the
    AO basis ``C = S^{-1/2} C_orth``. Eigenvalues of ``S`` are floored at
    ``1e-8`` before the inverse square root (numerical guard, as in the reference).

    Args:
        overlap: The AO overlap matrix ``S`` (symmetric positive-definite).
        hamiltonian: The Fock matrix ``F`` in the *same* AO ordering as ``S``.

    Returns:
        ``(orbital_energies, orbital_coefficients)`` -- ascending orbital energies
        ``(n_ao,)`` and AO-basis coefficients ``(n_ao, n_ao)`` whose column ``k``
        is orbital ``k``.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(overlap)
    floored = jnp.maximum(eigenvalues, _OVERLAP_EIGENVALUE_FLOOR)
    frac_overlap = eigenvectors / jnp.sqrt(floored)[..., None, :]
    transformed = jnp.swapaxes(frac_overlap, -1, -2) @ hamiltonian @ frac_overlap
    orbital_energies, orth_coefficients = jnp.linalg.eigh(transformed)
    orbital_coefficients = frac_overlap @ orth_coefficients
    return orbital_energies, orbital_coefficients


@lru_cache(maxsize=4096)
def _overlap_cached(
    atomic_numbers: tuple[int, ...],
    positions_bohr: tuple[tuple[float, float, float], ...],
) -> NDArray[np.float64]:
    """Build the PySCF def2-SVP spherical overlap for a hashable geometry (cached).

    Args:
        atomic_numbers: Nuclear charges as a hashable tuple.
        positions_bohr: Atom positions (Bohr) as a hashable nested tuple.

    Returns:
        The ``(n_ao, n_ao)`` overlap matrix ``mol.intor('int1e_ovlp_sph')``.
    """
    atom = [[int(z), tuple(pos)] for z, pos in zip(atomic_numbers, positions_bohr, strict=True)]
    mol = pyscf.gto.M(atom=atom, basis="def2svp", unit="Bohr", verbose=0)
    return np.asarray(mol.intor("int1e_ovlp_sph"), dtype=np.float64)


def overlap_matrix_def2svp(
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"] | Sequence[int],
    positions_bohr: Float[NDArray[np.float64], "n_atoms 3"],
) -> Float[Array, "n_ao n_ao"]:
    r"""Return the PySCF def2-SVP spherical AO overlap ``S`` at a QH9 geometry.

    Builds the molecule with ``pyscf.gto.M(basis='def2svp', unit='Bohr')`` at the
    QH9 positions (Bohr, opifex convention) and reads ``int1e_ovlp_sph``. Cached
    per geometry (:func:`_overlap_cached`) since the eval revisits geometries; the
    PySCF call is host-side only (this is an eval, not a training, path).

    Args:
        atomic_numbers: Nuclear charges, shape ``(n_atoms,)``.
        positions_bohr: Atom positions in Bohr, shape ``(n_atoms, 3)``.

    Returns:
        The overlap matrix ``S`` of shape ``(n_ao, n_ao)`` (PySCF internal
        spherical AO ordering).
    """
    numbers = tuple(int(z) for z in np.asarray(atomic_numbers).reshape(-1))
    positions = tuple(
        (float(row[0]), float(row[1]), float(row[2]))
        for row in np.asarray(positions_bohr, dtype=np.float64)
    )
    return jnp.asarray(_overlap_cached(numbers, positions))


def occupied_orbital_count(atomic_numbers: Int[NDArray[np.int32], " n_atoms"]) -> int:
    """Number of doubly-occupied orbitals of a closed-shell neutral molecule.

    ``n_occ = sum(Z) / 2`` for these closed-shell neutral QH9 molecules (reference
    ``test.py`` ``num_orb = int(batch.atoms.sum() / 2)``).

    Args:
        atomic_numbers: Nuclear charges, shape ``(n_atoms,)``.

    Returns:
        The integer occupied-orbital count.
    """
    return int(np.asarray(atomic_numbers).sum() // 2)


def orbital_energy_mae(
    predicted_energies: Float[Array, " n"],
    target_energies: Float[Array, " n"],
) -> Float[Array, ""]:
    """Mean absolute error between predicted and target orbital energies.

    Args:
        predicted_energies: Predicted orbital energies (any matching shape).
        target_energies: Target orbital energies.

    Returns:
        Scalar mean absolute orbital-energy error (Hartree).
    """
    return jnp.mean(jnp.abs(predicted_energies - target_energies))


def orbital_coefficient_similarity(
    predicted_coefficients: Float[Array, "n_ao n"],
    target_coefficients: Float[Array, "n_ao n"],
) -> Float[Array, ""]:
    r"""Mean sign-invariant per-orbital cosine similarity of orbital coefficients.

    Ports the reference ψ-similarity (``test.py`` lines 57-59):
    ``cosine_similarity(pred, target, dim=0).abs().mean()`` -- the cosine
    similarity is taken per orbital (over the AO axis, ``dim=0``), made
    sign-invariant via ``abs`` (orbital coefficients are defined up to a global
    sign), and averaged over orbitals.

    Args:
        predicted_coefficients: Predicted AO-basis coefficients ``(n_ao, n)``
            (column ``k`` is orbital ``k``).
        target_coefficients: Target AO-basis coefficients ``(n_ao, n)``.

    Returns:
        Scalar mean absolute per-orbital cosine similarity in ``[0, 1]``.
    """
    dot = jnp.sum(predicted_coefficients * target_coefficients, axis=0)
    predicted_norm = jnp.linalg.norm(predicted_coefficients, axis=0)
    target_norm = jnp.linalg.norm(target_coefficients, axis=0)
    cosine = dot / (predicted_norm * target_norm)
    return jnp.mean(jnp.abs(cosine))


def homo_lumo_gap(
    orbital_energies: Float[Array, " n_ao"],
    n_occupied: int,
) -> Float[Array, ""]:
    """HOMO-LUMO gap ``eps[n_occ] - eps[n_occ - 1]`` from ascending energies.

    Args:
        orbital_energies: Ascending orbital energies ``(n_ao,)``.
        n_occupied: Number of doubly-occupied orbitals.

    Returns:
        The scalar HOMO-LUMO gap (Hartree).
    """
    return orbital_energies[n_occupied] - orbital_energies[n_occupied - 1]


def hamiltonian_mae(
    predicted_fock: Float[Array, "n_ao n_ao"],
    target_fock: Float[Array, "n_ao n_ao"],
) -> Float[Array, ""]:
    """Mean absolute error between predicted and target Fock matrices (Hartree).

    Args:
        predicted_fock: Predicted Fock matrix.
        target_fock: Target Fock matrix (same ordering and shape).

    Returns:
        Scalar mean absolute Fock-element error (Hartree).
    """
    return jnp.mean(jnp.abs(predicted_fock - target_fock))


def evaluate_fock(
    predicted_fock: Float[Array, "n_ao n_ao"],
    target_fock: Float[Array, "n_ao n_ao"],
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"],
    positions: Float[NDArray[np.float64], "n_atoms 3"],
    n_electrons: int,
) -> dict[str, Float[Array, ""]]:
    r"""QH9 evaluation metrics for one molecule's predicted vs. target Fock.

    Both Fock matrices are reordered into PySCF's internal spherical AO ordering
    (:func:`to_pyscf_internal_ordering`), paired with the PySCF overlap
    (:func:`overlap_matrix_def2svp`) and diagonalized via
    :func:`cal_orbital_and_energies`. The ε-MAE is reported over all orbitals and
    over the occupied set, the ψ-similarity over the occupied orbitals, and the
    HOMO-LUMO-gap MAE and Fock MAE complete the dict.

    Args:
        predicted_fock: Predicted Fock in opifex spherical AO ordering.
        target_fock: Target Fock in opifex spherical AO ordering.
        atomic_numbers: Nuclear charges, shape ``(n_atoms,)``.
        positions: Atom positions in Bohr, shape ``(n_atoms, 3)``.
        n_electrons: Total electron count ``sum(Z)`` of the closed-shell molecule;
            the occupied-orbital count is ``n_electrons // 2``.

    Returns:
        ``{"orbital_energy_mae", "orbital_energy_mae_occ", "coefficient_similarity",
        "homo_lumo_gap_mae", "hamiltonian_mae"}`` of scalar JAX arrays.
    """
    overlap = overlap_matrix_def2svp(atomic_numbers, positions)
    predicted = to_pyscf_internal_ordering(predicted_fock, atomic_numbers)
    target = to_pyscf_internal_ordering(target_fock, atomic_numbers)

    predicted_energies, predicted_coefficients = cal_orbital_and_energies(overlap, predicted)
    target_energies, target_coefficients = cal_orbital_and_energies(overlap, target)

    n_occupied = n_electrons // 2
    return {
        "orbital_energy_mae": orbital_energy_mae(predicted_energies, target_energies),
        "orbital_energy_mae_occ": orbital_energy_mae(
            predicted_energies[:n_occupied], target_energies[:n_occupied]
        ),
        "coefficient_similarity": orbital_coefficient_similarity(
            predicted_coefficients[:, :n_occupied], target_coefficients[:, :n_occupied]
        ),
        "homo_lumo_gap_mae": jnp.abs(
            homo_lumo_gap(predicted_energies, n_occupied)
            - homo_lumo_gap(target_energies, n_occupied)
        ),
        "hamiltonian_mae": hamiltonian_mae(predicted_fock, target_fock),
    }


_METRIC_KEYS: tuple[str, ...] = (
    "orbital_energy_mae",
    "orbital_energy_mae_occ",
    "coefficient_similarity",
    "homo_lumo_gap_mae",
    "hamiltonian_mae",
)
"""The metric keys :func:`evaluate_fock` returns (and :func:`evaluate_qh9_test_set` aggregates)."""


def _complete_directed_edges(n_atoms: int) -> Int[Array, "2 n_edges"]:
    r"""Complete directed ``(sender, receiver)`` edge index for the predictor.

    The predictor reads ``edge_index[0]`` as the sender and ``edge_index[1]`` as
    the receiver (and :meth:`BlockHamiltonianPredictor.assemble_matrix` follows the
    same convention), so this builds the complete directed graph in that order --
    every ordered ``i != j`` pair as ``(sender=i, receiver=j)``.

    Args:
        n_atoms: Number of atoms in the molecule.

    Returns:
        ``(2, n_atoms * (n_atoms - 1))`` edge index (``sender, receiver``).
    """
    senders: list[int] = []
    receivers: list[int] = []
    for sender in range(n_atoms):
        for receiver in range(n_atoms):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return jnp.asarray([senders, receivers], dtype=jnp.int64)


def predict_fock(
    predictor: BlockHamiltonianPredictor,
    atomic_numbers: Int[NDArray[np.int32], " n_atoms"],
    positions_bohr: Float[NDArray[np.float64], "n_atoms 3"],
) -> Float[Array, "n_ao n_ao"]:
    r"""Assemble a single molecule's predicted dense Fock from the block predictor.

    Runs the predictor on the complete directed graph and assembles the symmetric
    dense Fock via :meth:`BlockHamiltonianPredictor.assemble_matrix` (reused). The
    edge index is in the predictor's ``(sender, receiver)`` convention so the
    assembled off-diagonal law matches the predictor's training orientation.

    Args:
        predictor: The trained (or fresh) block Hamiltonian predictor.
        atomic_numbers: Nuclear charges, shape ``(n_atoms,)``.
        positions_bohr: Atom positions in Bohr, shape ``(n_atoms, 3)``.

    Returns:
        The assembled symmetric dense Fock ``(n_ao, n_ao)`` in opifex spherical
        AO ordering.
    """
    numbers = jnp.asarray(np.asarray(atomic_numbers).reshape(-1), dtype=jnp.int32)
    positions = jnp.asarray(np.asarray(positions_bohr, dtype=np.float64))
    edge_index = _complete_directed_edges(int(numbers.shape[0]))
    blocks = predictor(numbers, positions, edge_index)
    return predictor.assemble_matrix(
        blocks["diagonal_blocks"],
        blocks["off_diagonal_blocks"],
        numbers,
        edge_index,
    )


def load_predictor_checkpoint(
    predictor: BlockHamiltonianPredictor,
    checkpoint_path: Path,
) -> BlockHamiltonianPredictor:
    """Restore a best-val orbax checkpoint into ``predictor`` (in place) and return it.

    Mirrors ``scripts/train_qh9_blocks.py``'s save format: the checkpoint is the
    ``nnx.to_pure_dict(nnx.state(predictor, nnx.Param))`` pure-dict written by an
    :class:`orbax.checkpoint.StandardCheckpointer`. Restoration reads back into the
    same pure-dict structure and replaces the predictor's parameter state.

    Args:
        predictor: A predictor built with the *same* config as the checkpoint.
        checkpoint_path: Path to the saved orbax checkpoint directory.

    Returns:
        The same ``predictor`` with restored parameters.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    # Only the Param state was checkpointed (non-Param buffers such as the Bessel
    # frequencies are deterministic from config); split them off with ``...`` so
    # the filter is exhaustive, restore the Params, then update in place.
    graph_def, params, rest = nnx.split(predictor, nnx.Param, ...)
    pure = nnx.to_pure_dict(params)
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(checkpoint_path.absolute(), target=pure)
    nnx.replace_by_pure_dict(params, restored)
    return nnx.merge(graph_def, params, rest)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the newest ``best_epoch_*`` checkpoint under ``checkpoint_dir``.

    Args:
        checkpoint_dir: The run's ``checkpoints/`` directory.

    Returns:
        The path of the highest-epoch checkpoint, or ``None`` if none exist.
    """
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(
        checkpoint_dir.glob("best_epoch_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None


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
    totals = dict.fromkeys(_METRIC_KEYS, 0.0)
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
        for key in _METRIC_KEYS:
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

    Decodes QH9-Stable (reusing :func:`~opifex.data.sources.qh9_source.read_qh9_sqlite`
    and the deterministic ``0.8/0.1/0.1`` split), optionally restores a best-val
    orbax checkpoint into ``predictor`` (:func:`load_predictor_checkpoint`), then
    aggregates :func:`evaluate_fock` over the test-split molecules
    (:func:`evaluate_examples`).

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

    from opifex.data.sources.qh9_source import qh9_random_split

    examples = read_qh9_sqlite(db_path)
    _, _, test_indices = qh9_random_split(len(examples))
    if limit is not None:
        test_indices = test_indices[:limit]
    test_examples = [examples[int(index)] for index in test_indices]
    logger.info("Evaluating QH9 test set over %d molecules", len(test_examples))
    return evaluate_examples(predictor, test_examples)


__all__ = [
    "QH9TestSetMetrics",
    "cal_orbital_and_energies",
    "evaluate_examples",
    "evaluate_fock",
    "evaluate_qh9_test_set",
    "hamiltonian_mae",
    "homo_lumo_gap",
    "latest_checkpoint",
    "load_predictor_checkpoint",
    "occupied_orbital_count",
    "orbital_coefficient_similarity",
    "orbital_energy_mae",
    "overlap_matrix_def2svp",
    "predict_fock",
    "to_pyscf_internal_ordering",
]
