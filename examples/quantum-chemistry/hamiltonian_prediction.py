# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
"""
# Equivariant DFT Hamiltonian Prediction (QHNet block form)

| Property      | Value                                                       |
|---------------|-------------------------------------------------------------|
| Level         | Advanced                                                    |
| Runtime       | ~1 min (GPU)                                                |
| Memory        | ~2 GB                                                       |
| Prerequisites | JAX, Flax NNX, SE(3)-equivariant networks, DFT/Hartree-Fock |

## Overview

Predict the **dense atomic-orbital DFT/Hartree-Fock Hamiltonian (Fock) matrix
`H`** of a molecule directly from its geometry, with the QHNet-style
`BlockHamiltonianPredictor` (Yu et al. 2023, arXiv:2306.04922). Rather than
assembling one dense matrix per fixed composition, the predictor emits a
**fixed `(14, 14)` block per atom (the on-site diagonal Fock block) and per
directed edge (the off-site off-diagonal block)** in the def2-SVP irrep layout
`BLOCK_IRREPS = 3x0e + 2x1e + 1x2e`. Because the blocks are fixed-size and the
NequIP convolution scatters messages only over within-molecule edges, **any
concatenation of heterogeneous molecules runs through one compiled forward** —
the property that makes batched QH9 training tractable.

The predicted blocks are SE(3)-equivariant *by construction*: rotating the
molecule rotates each block by the real Wigner-D matrix `D_14(R)` of
`BLOCK_IRREPS`, and the assembled dense matrix therefore obeys
`H(R x) = D(R) H(x) D(R)^T`. No ground-truth fit is needed to demonstrate the
structure — this example is a thin, untrained demo of the *block mechanics*:

- build a `BlockHamiltonianPredictor`,
- run it on **two different molecules concatenated into one flat batch**,
- show the per-atom and per-edge blocks and that the batch result matches running
  each molecule alone,
- `assemble_matrix` the per-molecule blocks into a **symmetric dense Fock**,
- verify the assembled-matrix equivariance `H(R x) = D(R) H(x) D(R)^T`.

For *training* the predictor against the QH9 benchmark (the converged
B3LYP/def2-SVP Fock matrices), see `scripts/train_qh9_blocks.py`, which streams
QH9-Stable through the block-form data pipeline
(`opifex.data.sources.qh9_blocks` / `qh9_block_stream`) and the block loss
(`qh9_block_loss`, `make_block_train_step`).

This example composes opifex's committed electronic-structure stack and changes
no library internals.

## Learning Goals

1. Build the heterogeneous-batchable `BlockHamiltonianPredictor` from the library
2. Run it on a concatenated batch of two different molecules and read the
   per-atom / per-edge blocks
3. Confirm the flat batch reproduces the per-molecule blocks (the segment design)
4. Assemble a single molecule's symmetric dense Fock with `assemble_matrix`
5. Verify the assembled-matrix equivariance `H(R x) = D(R) H(x) D(R)^T`
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np


mpl.use("Agg")
import matplotlib.pyplot as plt
from flax import nnx

from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.quantum.hamiltonian import (
    atom_orbital_counts,
    BLOCK_IRREPS,
    block_validity_mask,
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
    FULL_ORBITALS,
)


# float64 + tightened matmul precision so the SE(3)-equivariance residual is
# dominated by real model error, not reduced-precision GPU matmul.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

print("=" * 70)
print("Opifex Example: Equivariant DFT Hamiltonian Prediction (QHNet block form)")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(f"Block irreps: {BLOCK_IRREPS}  (per-atom / per-edge block size {FULL_ORBITALS})")

# %% [markdown]
"""
## Configuration

The molecules are **water** (`O, H, H`) and a **methane-like `C, H, H, H, H`**
fragment, two different compositions concatenated into one flat batch — the whole
point of the block predictor. Positions are in Bohr; the def2-SVP basis gives
each second-row atom (`C`, `O`) the full 14-slot block and each hydrogen the
masked `2s + 1p` (5-slot) block.

The predictor carries steerable hidden features up to `l = 2` (so the trunk can
represent every degree the `s`/`p`/`d` blocks need: `0e`, `1e`, `2e`), two NequIP
convolution layers, an 8-function Bessel radial basis, and a generous cutoff so
the small molecules form a complete within-molecule graph. No weights are trained
here: the equivariance is structural and holds for *any* weights.
"""

# %%
SEED = 0
OUTPUT_DIR = Path("docs/assets/examples/hamiltonian_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = BlockHamiltonianConfig(
    hidden_irreps="32x0e + 32x1o + 32x2e + 32x3o + 32x4e",  # uniform mul, l up to the d-d block
    sh_lmax=4,
    num_interactions=4,
    start_refinement_layer=1,
    num_radial_basis=8,
    cutoff=20.0,  # Bohr; large enough for a complete within-molecule graph
)

# Water (O, H, H) and a methane-like (C, H, H, H, H), in Bohr.
WATER_ATOMIC_NUMBERS = jnp.array([8, 1, 1])
WATER_POSITIONS = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
METHANE_ATOMIC_NUMBERS = jnp.array([6, 1, 1, 1, 1])
METHANE_POSITIONS = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [1.19, 1.19, 1.19],
        [-1.19, -1.19, 1.19],
        [1.19, -1.19, -1.19],
        [-1.19, 1.19, -1.19],
    ]
)

predictor = BlockHamiltonianPredictor(config=CONFIG, rngs=nnx.Rngs(SEED))
num_params = sum(
    int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(nnx.state(predictor, nnx.Param))
)
print(f"Predictor: irreps={CONFIG.hidden_irreps}, layers={CONFIG.num_interactions}")
print(f"Trainable parameters: {num_params}")


# %% [markdown]
"""
## Building the Flat Batch

The predictor consumes a flat concatenated batch: `(A,)` atomic numbers, `(A, 3)`
positions and a `(2, E)` directed `(senders, receivers)` edge index whose indices
are *offset per molecule* so edges never cross molecule boundaries. Each molecule
gets a complete within-molecule directed graph (every ordered atom pair).
"""


# %%
def complete_edges(offset: int, n_atoms: int) -> tuple[list[int], list[int]]:
    """Return the directed complete-graph edges of one molecule, offset into the batch.

    Args:
        offset: Index of the molecule's first atom in the concatenated batch.
        n_atoms: Number of atoms in the molecule.

    Returns:
        A pair ``(senders, receivers)`` of within-molecule directed edges (no
        self-loops), shifted by ``offset``.
    """
    senders: list[int] = []
    receivers: list[int] = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                senders.append(offset + i)
                receivers.append(offset + j)
    return senders, receivers


def single_batch(
    atomic_numbers: jax.Array, positions: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the flat-batch inputs for a single molecule (complete graph, segment 0).

    Args:
        atomic_numbers: Nuclear charges [Shape: (n_atoms,)].
        positions: Atomic positions in Bohr [Shape: (n_atoms, 3)].

    Returns:
        ``(atomic_numbers, positions, edge_index, node_batch)`` for the molecule.
    """
    n_atoms = int(atomic_numbers.shape[0])
    senders, receivers = complete_edges(0, n_atoms)
    edge_index = jnp.asarray([senders, receivers], dtype=jnp.int32)
    node_batch = jnp.zeros((n_atoms,), dtype=jnp.int32)
    return atomic_numbers, positions, edge_index, node_batch


n_water = int(WATER_ATOMIC_NUMBERS.shape[0])
n_methane = int(METHANE_ATOMIC_NUMBERS.shape[0])

batch_atomic_numbers = jnp.concatenate([WATER_ATOMIC_NUMBERS, METHANE_ATOMIC_NUMBERS])
batch_positions = jnp.concatenate([WATER_POSITIONS, METHANE_POSITIONS])
water_senders, water_receivers = complete_edges(0, n_water)
methane_senders, methane_receivers = complete_edges(n_water, n_methane)
batch_edge_index = jnp.asarray(
    [water_senders + methane_senders, water_receivers + methane_receivers], dtype=jnp.int32
)
batch_node_batch = jnp.concatenate(
    [jnp.zeros((n_water,), jnp.int32), jnp.full((n_methane,), 1, jnp.int32)]
)

print()
print(f"Flat batch: {n_water + n_methane} atoms, {batch_edge_index.shape[1]} directed edges")
print(f"  water   (O,H,H)      atoms 0..{n_water - 1}")
print(f"  methane (C,H,H,H,H)  atoms {n_water}..{n_water + n_methane - 1}")

# %% [markdown]
"""
## Running the Predictor

One jitted forward over the concatenated batch emits a `(14, 14)` diagonal block
per atom and a `(14, 14)` off-diagonal block per directed edge. The diagonal
blocks are symmetrised inside the forward (QHNet `D + D^T`); the off-diagonal
blocks are symmetrised at assembly.
"""


# %%
@nnx.jit
def predict_blocks(
    module: BlockHamiltonianPredictor,
    atomic_numbers: jax.Array,
    positions: jax.Array,
    edge_index: jax.Array,
    node_batch: jax.Array,
) -> dict[str, jax.Array]:
    """Run the predictor over a flat batch, returning the per-atom/per-edge blocks."""
    return module(atomic_numbers, positions, edge_index, node_batch)


batch_out = predict_blocks(
    predictor, batch_atomic_numbers, batch_positions, batch_edge_index, batch_node_batch
)
diagonal_blocks = batch_out["diagonal_blocks"]
off_diagonal_blocks = batch_out["off_diagonal_blocks"]

print(f"diagonal_blocks:     {diagonal_blocks.shape}  (one (14,14) block per atom)")
print(f"off_diagonal_blocks: {off_diagonal_blocks.shape}  (one (14,14) block per edge)")

# The diagonal blocks are symmetric (the forward applies the QHNet D + D^T).
diag_symmetry = float(jnp.max(jnp.abs(diagonal_blocks - jnp.swapaxes(diagonal_blocks, -1, -2))))
print(f"max diagonal-block asymmetry: {diag_symmetry:.2e}")

# %% [markdown]
"""
## Heterogeneous-Batch Consistency

The segment design must give *exactly* the per-molecule blocks when the molecules
are run alone — proof that concatenating heterogeneous molecules does not leak
information across molecule boundaries. The first `n_water` diagonal blocks must
equal water's, the rest methane's; the off-diagonal blocks split at the edge
boundary.
"""

# %%
water_out = predict_blocks(predictor, *single_batch(WATER_ATOMIC_NUMBERS, WATER_POSITIONS))
methane_out = predict_blocks(predictor, *single_batch(METHANE_ATOMIC_NUMBERS, METHANE_POSITIONS))

n_water_edges = n_water * (n_water - 1)
diag_water_match = float(jnp.max(jnp.abs(diagonal_blocks[:n_water] - water_out["diagonal_blocks"])))
diag_methane_match = float(
    jnp.max(jnp.abs(diagonal_blocks[n_water:] - methane_out["diagonal_blocks"]))
)
off_water_match = float(
    jnp.max(jnp.abs(off_diagonal_blocks[:n_water_edges] - water_out["off_diagonal_blocks"]))
)
off_methane_match = float(
    jnp.max(jnp.abs(off_diagonal_blocks[n_water_edges:] - methane_out["off_diagonal_blocks"]))
)

print("Flat-batch blocks vs per-molecule blocks (max abs difference):")
print(f"  water   diagonal: {diag_water_match:.2e}   off-diagonal: {off_water_match:.2e}")
print(f"  methane diagonal: {diag_methane_match:.2e}   off-diagonal: {off_methane_match:.2e}")

# %% [markdown]
"""
## Assembling the Dense Fock Matrix

`assemble_matrix` masks each `(14, 14)` block to its element's valid AO slots
(`block_validity_mask`), scatters it to the per-atom AO offsets
(`atom_orbital_counts`), writes off-diagonal blocks at both `(i, j)` and `(j, i)`,
and finally symmetrises `H = H~ + H~^T`. For water (`O` 14 + `H` 5 + `H` 5) the
result is a symmetric `(24, 24)` matrix.
"""

# %%
water_z, water_pos, water_edges, _ = single_batch(WATER_ATOMIC_NUMBERS, WATER_POSITIONS)
water_matrix = predictor.assemble_matrix(
    water_out["diagonal_blocks"], water_out["off_diagonal_blocks"], water_z, water_edges
)
expected_ao = int(jnp.sum(atom_orbital_counts(WATER_ATOMIC_NUMBERS)))
matrix_symmetry = float(jnp.max(jnp.abs(water_matrix - water_matrix.T)))

print(
    f"Assembled water Fock: shape {tuple(water_matrix.shape)} "
    f"(expected ({expected_ao}, {expected_ao}): O 14 + H 5 + H 5)"
)
print(f"max |H - H^T|: {matrix_symmetry:.2e}")

# %% [markdown]
"""
## Equivariance Check

The defining property: under a random proper rotation `R` the assembled dense
matrix transforms as `H(R x) = D(R) H(x) D(R)^T`, where `D(R)` is block-diagonal
over atoms — each atom's block is the `(14, 14)` Wigner-D `D_14(R)` of
`BLOCK_IRREPS` restricted to that atom's populated AO slots (whole shells, so the
restriction is exact). This holds for *any* weights, so the untrained predictor
already satisfies it to matmul precision.
"""


# %%
def block_wigner(rotation: jax.Array) -> jax.Array:
    """Build the ``(14, 14)`` block-diagonal Wigner-D of ``BLOCK_IRREPS``."""
    matrices: list[jax.Array] = []
    for mul, irrep in BLOCK_IRREPS.blocks:
        wigner = wigner_d(irrep.l, rotation)
        matrices.extend([wigner] * mul)
    return jax.scipy.linalg.block_diag(*matrices)


def assembled_ao_wigner(atomic_numbers: jax.Array, rotation: jax.Array) -> jax.Array:
    """Build the ``(n_ao, n_ao)`` AO-basis Wigner-D matching ``assemble_matrix``.

    Args:
        atomic_numbers: Nuclear charges of the single molecule [Shape: (n_atoms,)].
        rotation: A ``3x3`` proper rotation matrix.

    Returns:
        The block-diagonal-over-atoms rotation ``D(R)`` of the assembled matrix.
    """
    full = block_wigner(rotation)
    diag_mask = block_validity_mask(atomic_numbers)
    per_atom: list[jax.Array] = []
    for atom in range(int(atomic_numbers.shape[0])):
        valid = jnp.where(jnp.diagonal(diag_mask[atom]))[0]
        per_atom.append(full[jnp.ix_(valid, valid)])
    return jax.scipy.linalg.block_diag(*per_atom)


def random_rotation(seed: int) -> jax.Array:
    """Return a uniformly random proper rotation matrix (det = +1)."""
    key = jax.random.PRNGKey(seed)
    gaussian = jax.random.normal(key, (3, 3), dtype=jnp.float64)
    orthogonal, _ = jnp.linalg.qr(gaussian)
    return orthogonal * jnp.sign(jnp.linalg.det(orthogonal))


print()
print("Assembled-matrix equivariance H(R x) = D(R) H(x) D(R)^T under random rotations:")
equivariance_errors: list[float] = []
for seed in range(5):
    rotation = random_rotation(seed)
    rotated_out = predict_blocks(
        predictor, water_z, water_pos @ rotation.T, water_edges, jnp.zeros((n_water,), jnp.int32)
    )
    rotated_matrix = predictor.assemble_matrix(
        rotated_out["diagonal_blocks"], rotated_out["off_diagonal_blocks"], water_z, water_edges
    )
    wigner = assembled_ao_wigner(WATER_ATOMIC_NUMBERS, rotation)
    expected = wigner @ water_matrix @ wigner.T
    error = float(jnp.max(jnp.abs(rotated_matrix - expected)))
    equivariance_errors.append(error)
    print(f"  rotation {seed}: max |H(Rx) - D H D^T| = {error:.3e}")

max_equivariance_error = max(equivariance_errors)
print(f"Worst-case equivariance error: {max_equivariance_error:.3e}")

# %% [markdown]
"""
## Visualization

Two diagnostics: the assembled water Fock heatmap (its symmetric structure and
the masked H-block sparsity), and the per-rotation equivariance error.
"""

# %%
matrix_np = np.asarray(water_matrix)
vmax = float(np.abs(matrix_np).max())
fig, ax = plt.subplots(figsize=(6.4, 5.4))
image = ax.imshow(matrix_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("Assembled water Fock H (untrained, symmetric)")
ax.set_xlabel("AO index")
ax.set_ylabel("AO index")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fock_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(equivariance_errors)), equivariance_errors, color="#2ca02c")
ax.set_yscale("log")
ax.set_xlabel("Random rotation")
ax.set_ylabel("max |H(Rx) - D(R) H(x) D(R)^T|")
ax.set_title("Assembled-matrix SE(3) equivariance error")
ax.grid(True, which="both", axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equivariance_error.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print()
print(f"Saved plots to {OUTPUT_DIR}/")
print("  fock_heatmap.png, equivariance_error.png")

# %% [markdown]
"""
## Summary

A thin composition of opifex's electronic-structure stack — the
`BlockHamiltonianPredictor`, its per-atom / per-edge `(14, 14)` blocks, and the
`assemble_matrix` helper — predicts the dense DFT/Hartree-Fock Fock matrix `H` of
a molecule from its geometry in QHNet block form. Two different molecules
concatenate into one flat batch and run through a single compiled forward, with
the batch blocks reproducing the per-molecule blocks exactly — the segment design
that makes heterogeneous QH9 training tractable.

The defining guarantee is **SE(3) equivariance by construction**: each block
transforms as `B(R x) = D_14(R) B(x) D_14(R)^T`, so the assembled dense matrix
obeys `H(R x) = D(R) H(x) D(R)^T` to matmul precision for *any* weights, because
every block is a Clebsch-Gordan (Wigner-Eckart) contraction of a steerable
feature and the matrix is symmetrised `H = H~ + H~^T`.

To *train* this predictor against the QH9 benchmark (converged B3LYP/def2-SVP
Fock matrices for QM9 molecules), run `scripts/train_qh9_blocks.py`, which streams
QH9-Stable through the block-form data pipeline
(`opifex.data.sources.qh9_blocks` / `qh9_block_stream`) and the masked block loss
(`qh9_block_loss`, `make_block_train_step`). The method, the block expansion and
the QH9 benchmark are documented in
[Hamiltonian Prediction](../../methods/hamiltonian-prediction.md).
"""
