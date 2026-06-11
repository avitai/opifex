# Hamiltonian Prediction

Equivariant prediction of the electronic-structure matrices of a molecule — the
DFT/Hartree-Fock Hamiltonian (Fock) matrix `H` and the atomic-orbital overlap `S`
— directly from its geometry. Mean-field electronic structure solves the
generalised eigenvalue problem `H C = S C eps` in an AO basis; predicting the
converged `H` in one shot removes the self-consistent-field iteration that is the
bottleneck. opifex builds the predictor from the native SE(3)-equivariant kit in
`opifex.neural.equivariant` and the NequIP steerable trunk, following QHNet (Yu et
al. 2023, [arXiv:2306.04922](https://arxiv.org/abs/2306.04922)).

## The matrix as an equivariant operator

`H` and `S` are **equivariant operators**, not invariant scalars. Each atomic
orbital has a definite angular momentum `l`, so a rigid rotation `R` of the
molecule rotates the AO components by the Wigner-D matrix `D^l(R)` of their shell.
The whole matrix therefore transforms as

```
H(R x) = D(R) H(x) D(R)^T,
```

with `D(R)` block-diagonal over shells (one `wigner_d(l, R)` block per shell). A
model that bakes this law in never has to learn the symmetry from data and needs
far fewer training examples — a single converged geometry is enough geometric
supervision to test that the architecture can represent a real matrix.

## Design: trunk → fixed-size blocks → assemble → symmetrize

`opifex.neural.quantum.hamiltonian.BlockHamiltonianPredictor` builds the matrix
in fixed-size **blocks**, every primitive reused from the equivariant kit:

1. **Steerable trunk.** The NequIP tensor-product message passing
   (`opifex.neural.atomistic.backbones.nequip`) produces a *full steerable*
   per-atom feature (an `IrrepsArray`, not just the scalar readout the energy head
   uses). The trunk is shared with the
   [Atomistic Potentials](atomistic-potentials.md) family.
2. **Node head — diagonal blocks `H_ii`.** A shared `HamiltonianBlockExpansion`
   turns each atom's node feature and invariant embedding into the full
   `(14, 14)` def2-SVP on-site block (`BLOCK_IRREPS = 3x0e + 2x1e + 1x2e`), which
   the forward symmetrizes (`D + D^T`).
3. **Edge head — off-diagonal blocks `H_ij`.** For every directed atom pair a
   mixed sender/receiver edge feature (a radially-modulated tensor product of the
   sender's features with the edge spherical harmonics, combined with the
   receiver's features) is expanded by the *same* head into the `(14, 14)`
   off-site block.
4. **Mask, scatter and symmetrize.** `assemble_matrix` masks each `(14, 14)`
   block to its element's valid AO slots (`block_validity_mask` — hydrogen keeps
   `2s + 1p`, C/N/O/F all 14), scatters it to the per-atom AO offsets
   (`atom_orbital_counts`), writes off-diagonal blocks at both `(i, j)` and
   `(j, i)`, then `H = H~ + H~^T` makes the matrix Hermitian (QHNet's
   `transpose_edge_index` symmetrization).

Because every stage is equivariant, the assembled matrix obeys
`H(R x) = D(R) H(x) D(R)^T` **for any weights**, so the symmetry is structural,
not learned. The blocks are a *fixed* `(14, 14)` per atom / per edge and the
NequIP convolution scatters only over within-molecule edges, so **any
concatenation of heterogeneous molecules runs through one compiled forward** — no
per-composition assembly plan and no per-molecule recompile. The predictor is
`jit`/`grad`/`vmap` clean over geometry.

## The block expansion (`HamiltonianBlockExpansion`)

The heart of the predictor is the *inverse* use of the Clebsch-Gordan tensor that
drives the tensor product. A tensor product **couples** two irreps `l_i ⊗ l_j`
down to a single output irrep `L`; the expansion **decouples** a steerable feature
— a direct sum over the triangle-rule degrees `|l_i - l_j| ≤ L ≤ l_i + l_j` — back
into the `(2 l_i + 1) x (2 l_j + 1)` matrix block that couples to those `L`. For a
shell pair `(l_i, l_j)` the block is

```
B[a, b] = sum_L sum_M  C^{l_i l_j L}_{a b M}  f^L_M,
```

the contraction of the **last** index `M` of the real Clebsch-Gordan tensor
`clebsch_gordan(l_i, l_j, L)` with the `L`-chunk `f^L` of the input feature
(QHNet's `einsum("ijk, ...k -> ...ij")`). Because `C` is the intertwiner between
`D^{l_i} ⊗ D^{l_j}` and `D^L`, and the feature transforms as `f^L → D^L f^L`, the
block satisfies the Wigner-Eckart block law
`B(R x) = D^{l_i}(R) B(x) D^{l_j}(R)^T` — the block-wise statement of the matrix
equivariance.

The public surface lives in `opifex.neural.quantum.hamiltonian`:

| Symbol | Role |
|--------|------|
| `BLOCK_IRREPS` | the 14-dim row/col representation of a Fock block (`3x0e + 2x1e + 1x2e`) |
| `HamiltonianBlockExpansion` | the shared block head: last-index Clebsch-Gordan contraction of a steerable feature into a `(14, 14)` block, driven by an invariant embedding |
| `block_validity_mask` / `atom_orbital_counts` | the per-element AO mask (hydrogen `2s + 1p`, C/N/O/F all 14) and populated-AO counts assembly uses |
| `BlockHamiltonianPredictor` | the heterogeneous-batchable predictor: per-atom diagonal + per-edge off-diagonal blocks, with `assemble_matrix` building the symmetric dense matrix |

## Example

```python
import jax.numpy as jnp
from flax import nnx

from opifex.neural.quantum.hamiltonian import (
    BlockHamiltonianConfig, BlockHamiltonianPredictor,
)

predictor = BlockHamiltonianPredictor(
    config=BlockHamiltonianConfig(
        hidden_irreps="32x0e + 16x1o + 8x2e",  # must carry every degree the s/p/d blocks reach
        sh_lmax=2,
        num_interactions=2,
        cutoff=20.0,                            # Bohr
    ),
    rngs=nnx.Rngs(0),
)

# A flat concatenated batch: (A,) atomic numbers, (A, 3) positions (Bohr) and a
# (2, E) within-molecule directed (senders, receivers) edge index. Water (O,H,H).
atomic_numbers = jnp.array([8, 1, 1])
positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
edge_index = jnp.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=jnp.int32)

blocks = predictor(atomic_numbers, positions, edge_index)
# blocks["diagonal_blocks"]: (3, 14, 14), blocks["off_diagonal_blocks"]: (6, 14, 14)
matrix = predictor.assemble_matrix(
    blocks["diagonal_blocks"], blocks["off_diagonal_blocks"], atomic_numbers, edge_index
)   # (24, 24), symmetric and equivariant (O 14 + H 5 + H 5)
```

A single trained predictor generalises across molecules with **no rebinding**:
the same weights and the same compiled forward run over any concatenation of
molecules (only the padded atom/edge counts are static), so heterogeneous QH9
batches need no per-composition recompile.

## Training against QH9

The training target is the QH9 benchmark: converged B3LYP/def2-SVP Fock matrices
for the QM9 molecules (Yu et al. 2023, QH9). `opifex.data.sources.qh9_blocks` /
`qh9_block_stream` read the QH9-Stable SQLite database directly (no `torch`),
apply the def2-SVP convention transform into opifex's spherical AO ordering, cut
each Fock matrix into the per-atom / per-edge `(14, 14)` blocks the predictor
emits, and collate heterogeneous molecules into one flat batch. The masked block
loss `qh9_block_loss` (with `make_block_train_step` / `make_block_eval_step`)
compares only the valid AO slots. `scripts/train_qh9_blocks.py` wires the data
pipeline, predictor and loss into an end-to-end training run.

Both QH9 benchmarks are supported. QH9-Stable (one equilibrium geometry per
molecule, `--dataset stable --split random`) and QH9-Dynamic (~100 molecular-
dynamics geometries per molecule, `--dataset dynamic-300k`/`dynamic-100k`) share
the predictor, loss and out-of-core padded source. The Dynamic loader
(`opifex.data.sources.qh9_dynamic`) reproduces the two reference splits exactly:
`--split geometry` (every molecule in all splits at disjoint timesteps) and
`--split mol` (whole molecules held out — the harder generalisation test). Because
a Dynamic molecule's geometries share a non-unique `id`, the source keys rows by
`rowid`; the splits, decode and padding are otherwise identical to Stable.

For a thin, untrained demo of the block mechanics — building the predictor,
running a concatenated batch of two molecules, assembling a symmetric dense Fock,
and verifying the assembled-matrix equivariance `H(R x) = D(R) H(x) D(R)^T` with
`wigner_d` — see
[Equivariant DFT Hamiltonian Prediction](../examples/quantum-chemistry/hamiltonian-prediction.md).

## Accelerating SCF with a predicted Fock

A predicted Fock matrix close to the self-consistent one is a high-quality SCF
initial guess: seeding the iteration with a near-converged density lets the
Anderson/DIIS solver reach the fixed point in fewer steps than the default
core-Hamiltonian guess. The SCF solver exposes this seam directly —
`SCFSolver.solve(initial_density=...)` accepts a closed-shell density in the
solver's AO basis, and `density_from_fock(fock, overlap, n_occupied)`
reconstructs that density from a Fock matrix by solving `FC = SCe`. The converged
result is independent of the seed; only the iteration count changes.

`measure_scf_acceleration(solver, initial_density)` runs the solve from both the
default guess and the supplied density and reports the iteration counts
(`baseline_iterations`, `guided_iterations`, `iteration_reduction`), after
checking both reach the same energy — a guard that a mismatched basis would trip.

The guess must live in the *solver's* AO basis. The native `SCFSolver` runs LDA
on a Cartesian def2-SVP (or STO-3G) basis, whereas the QH9 predictor emits
spherical def2-SVP B3LYP Fock blocks, so wiring the trained predictor end-to-end
additionally requires a spherical-def2-SVP solver path (a Cartesian↔spherical `d`
transform); that basis bridge is tracked separately and is not assumed by the
acceleration utilities.

## Extending

- **SO(2)-frame convolution (QHNetV2).** Yu et al. 2025
  ([arXiv:2506.09398](https://arxiv.org/abs/2506.09398)), building on eSCN
  (Passaro & Zitnick 2023, [arXiv:2302.03655](https://arxiv.org/abs/2302.03655)),
  rotate each edge into a local frame aligned with its direction, reducing the
  `SO(3)` Clebsch-Gordan tensor product to cheaper `SO(2)` operations — the main
  scalability lever for large bases. The present trunk uses the full `SO(3)`
  tensor product; the SO(2)-frame upgrade is a drop-in replacement for the edge
  tensor product.
- **Overlap matrix `S` and larger bases.** The block head emits the def2-SVP
  Fock blocks; a second head with the same mechanism predicts the AO overlap `S`
  (which obeys the identical transformation law), and the `BLOCK_IRREPS` layout
  extends to bases beyond def2-SVP by widening the per-atom block irreps.

## See also

- [Atomistic Potentials](atomistic-potentials.md) — the NequIP steerable trunk
  the predictor reuses, and the shared equivariant primitives.
- Yu et al. 2023, *Efficient and Equivariant Graph Networks for Predicting Quantum
  Hamiltonian* (QHNet), ICML 2023
  ([arXiv:2306.04922](https://arxiv.org/abs/2306.04922)).
- Yu et al. 2025, *QHNetV2: A Fully Equivariant Network for Quantum Hamiltonian
  Prediction* ([arXiv:2506.09398](https://arxiv.org/abs/2506.09398)).
- Unke et al. 2021, *SE(3)-equivariant prediction of molecular wavefunctions and
  electronic densities* (PhiSNet), NeurIPS 2021
  ([arXiv:2106.02347](https://arxiv.org/abs/2106.02347)).
- Yu et al. 2023, *QH9: A Quantum Hamiltonian Prediction Benchmark for QM9
  Molecules*, NeurIPS 2023
  ([arXiv:2306.09549](https://arxiv.org/abs/2306.09549)).
