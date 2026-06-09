"""Equivariant DFT Hamiltonian/overlap prediction (QHNet-style).

A native, ``jax``/``flax.nnx`` implementation of equivariant electronic-structure
matrix prediction (Yu et al. 2023, "QHNet", arXiv:2306.04922) built entirely on
opifex's Q0 equivariant kit (:mod:`opifex.neural.equivariant`) and the NequIP
steerable trunk (:mod:`opifex.neural.atomistic.backbones.nequip`). The predictor
emits the dense AO-basis Fock/Hamiltonian matrix ``H`` (and, with a second head,
the overlap matrix ``S``) for a molecular system, exactly in opifex's shell/AO
ordering.

The public surface:

* :func:`block_from_irreps` / :class:`PairExpansion` -- the QHNet expansion
  mechanism (last-index Clebsch-Gordan contraction) that turns a steerable pair
  feature into a dense Hamiltonian block.
* :class:`HamiltonianPredictor` -- the registered ``"hamiltonian"`` property head
  assembling node (diagonal) and edge (off-diagonal) blocks into a symmetric
  dense matrix.
"""

from opifex.neural.quantum.hamiltonian._block_expansion import (
    HamiltonianBlockExpansion,
)
from opifex.neural.quantum.hamiltonian._expansion import (
    block_from_irreps,
    pair_feature_irreps,
    PairExpansion,
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
    make_block_eval_step,
    make_block_train_step,
    masked_block_loss,
    qh9_block_loss,
)
from opifex.neural.quantum.hamiltonian.predictor import (
    HamiltonianPredictor,
    HamiltonianPredictorConfig,
)
from opifex.neural.quantum.hamiltonian.qh9_training import (
    batched_fock_loss,
    fit_qh9,
    fit_qh9_bucket,
    fit_qh9_examples,
    fock_loss,
    make_batched_train_step,
    predict_spherical_fock,
    predict_spherical_fock_batch,
    QH9FitResult,
    QH9TrainConfig,
    spherical_transform_for,
)
from opifex.neural.quantum.hamiltonian.so2_convolution import SO2EdgeConvolution


__all__ = [
    "BLOCK_IRREPS",
    "FULL_ORBITALS",
    "ORBITAL_MASK",
    "BlockHamiltonianConfig",
    "BlockHamiltonianPredictor",
    "BlockTrainConfig",
    "HamiltonianBlockExpansion",
    "HamiltonianPredictor",
    "HamiltonianPredictorConfig",
    "PairExpansion",
    "QH9FitResult",
    "QH9TrainConfig",
    "SO2EdgeConvolution",
    "atom_orbital_counts",
    "batched_fock_loss",
    "block_from_irreps",
    "block_validity_mask",
    "fit_qh9",
    "fit_qh9_bucket",
    "fit_qh9_examples",
    "fock_loss",
    "make_batched_train_step",
    "make_block_eval_step",
    "make_block_train_step",
    "masked_block_loss",
    "pair_feature_irreps",
    "predict_spherical_fock",
    "predict_spherical_fock_batch",
    "qh9_block_loss",
    "spherical_transform_for",
]
