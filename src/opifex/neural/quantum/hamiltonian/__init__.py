"""Equivariant DFT Hamiltonian prediction in QHNet block form (native JAX/NNX).

A ``jax``/``flax.nnx`` implementation of equivariant electronic-structure matrix
prediction (Yu et al. 2023, "QHNet", arXiv:2306.04922) built on opifex's Q0
equivariant kit (:mod:`opifex.neural.equivariant`) and the NequIP steerable trunk
(:mod:`opifex.neural.atomistic.backbones.nequip`). Rather than assembling one
dense matrix per fixed composition, the predictor emits a fixed ``(14, 14)``
diagonal block per atom and ``(14, 14)`` off-diagonal block per directed edge, so
heterogeneous molecules concatenate into a single flat batch (the def2-SVP
``FULL_ORBITALS = 14`` AO slots per second-row atom, masked per element).

The public surface:

* :class:`HamiltonianBlockExpansion` -- the QHNet expansion head (last-index
  Clebsch-Gordan contraction) turning a steerable feature plus an invariant
  embedding into a dense ``(14, 14)`` block.
* :class:`BlockHamiltonianPredictor` / :class:`BlockHamiltonianConfig` -- the
  heterogeneous-batchable per-atom / per-edge block predictor; its
  :meth:`~...block_predictor.BlockHamiltonianPredictor.assemble_matrix` scatters
  the blocks into a single molecule's symmetric dense Fock matrix.
* The orbital-layout primitives (:data:`BLOCK_IRREPS`, :data:`FULL_ORBITALS`,
  :data:`ORBITAL_MASK`, :func:`atom_orbital_counts`, :func:`block_validity_mask`)
  fixing the def2-SVP AO slots each block occupies.
* The GPU-fused block training surface (:class:`BlockTrainConfig`,
  :func:`per_molecule_block_loss`, :func:`make_fused_block_train_step` /
  :func:`make_fused_block_eval_step`) used by ``scripts/train_qh9_blocks.py``.
"""

from opifex.neural.quantum.hamiltonian._block_expansion import (
    HamiltonianBlockExpansion,
)
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    atom_orbital_counts,
    BLOCK_IRREPS,
    block_validity_mask,
    FULL_ORBITALS,
    ORBITAL_MASK,
)
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)
from opifex.neural.quantum.hamiltonian.block_training import (
    BlockTrainConfig,
    make_fused_block_eval_step,
    make_fused_block_train_step,
    per_molecule_block_loss,
    predict_blocks_vmapped,
)
from opifex.neural.quantum.hamiltonian.qh9_eval import (
    cal_orbital_and_energies,
    evaluate_examples,
    evaluate_fock,
    evaluate_qh9_test_set,
    hamiltonian_mae,
    homo_lumo_gap,
    latest_checkpoint,
    load_predictor_checkpoint,
    occupied_orbital_count,
    orbital_coefficient_similarity,
    orbital_energy_mae,
    overlap_matrix_def2svp,
    predict_fock,
    QH9TestSetMetrics,
    to_pyscf_internal_ordering,
)


__all__ = [
    "BLOCK_IRREPS",
    "FULL_ORBITALS",
    "ORBITAL_MASK",
    "BlockHamiltonianConfig",
    "BlockHamiltonianPredictor",
    "BlockTrainConfig",
    "HamiltonianBlockExpansion",
    "QH9TestSetMetrics",
    "atom_orbital_counts",
    "block_validity_mask",
    "cal_orbital_and_energies",
    "evaluate_examples",
    "evaluate_fock",
    "evaluate_qh9_test_set",
    "hamiltonian_mae",
    "homo_lumo_gap",
    "latest_checkpoint",
    "load_predictor_checkpoint",
    "make_fused_block_eval_step",
    "make_fused_block_train_step",
    "occupied_orbital_count",
    "orbital_coefficient_similarity",
    "orbital_energy_mae",
    "overlap_matrix_def2svp",
    "per_molecule_block_loss",
    "predict_blocks_vmapped",
    "predict_fock",
    "to_pyscf_internal_ordering",
]
