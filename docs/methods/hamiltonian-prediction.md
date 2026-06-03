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

## Design: trunk → blocks → assemble → symmetrize

`opifex.neural.quantum.hamiltonian.HamiltonianPredictor` (registered
`@register_property_head("hamiltonian")`) builds the dense matrix in four stages,
every primitive reused from the equivariant kit:

1. **Steerable trunk.** The NequIP tensor-product message passing
   (`opifex.neural.atomistic.backbones.nequip`) produces a *full steerable*
   per-atom feature (an `IrrepsArray`, not just the scalar readout the energy head
   uses). The trunk is shared with the
   [Atomistic Potentials](atomistic-potentials.md) family.
2. **Node head — diagonal blocks `H_ii`.** For every intra-atom shell pair
   `(l_i, l_j)` an equivariant `PairExpansion` turns the atom's node feature into
   the `(2 l_i + 1) x (2 l_j + 1)` on-site block.
3. **Edge head — off-diagonal blocks `H_ij`.** For every directed atom pair a
   mixed sender/receiver edge feature (a radially-modulated tensor product of the
   sender's features with the edge spherical harmonics, combined with the
   receiver's features) is expanded per shell-pair type into the off-site block.
4. **Scatter and symmetrize.** Blocks are written into the dense matrix at their
   static `(row_offset, col_offset)` AO positions, then `H = H~ + H~^T` makes the
   matrix Hermitian. The off-diagonal contribution naturally realises
   `H_ij = H_ji^T` because the directed graph contains both `(i, j)` and `(j, i)`
   and the transpose of the `(j, i)` block lands on the `(i, j)` sub-matrix
   (QHNet's `transpose_edge_index` symmetrization).

Because every stage is equivariant, the assembled matrix obeys
`H(R x) = D(R) H(x) D(R)^T` **for any weights**, so the symmetry is structural,
not learned. The per-block shapes are static (one shared `PairExpansion` per
angular-momentum pair type), so the predictor is `jit`/`grad`/`vmap` clean over
geometry.

## The block expansion (`block_from_irreps`)

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
| `pair_feature_irreps(l_i, l_j)` | the steerable layout `sum_L 1 x L` a block expands from (triangle rule, parity `(-1)^L`) |
| `block_from_irreps(feature, l_i, l_j)` | the last-index Clebsch-Gordan contraction: steerable feature → dense `(2l_i+1, 2l_j+1)` block |
| `PairExpansion` | learnable expansion into `mul_i x mul_j` blocks; the multiplicity axes distinguish same-`l` shells (e.g. oxygen `1s`/`2s`) |
| `HamiltonianPredictor` | the registered `"hamiltonian"` head assembling node + edge blocks into a symmetric matrix |

## Example

```python
import jax.numpy as jnp
from flax import nnx

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.hamiltonian import (
    HamiltonianPredictor, HamiltonianPredictorConfig,
)

water = MolecularSystem(
    atomic_numbers=jnp.array([8, 1, 1]),
    positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
    basis_set="sto-3g",
)
basis = AtomicOrbitalBasis.from_molecular_system(water, basis_name="sto-3g")
predictor = HamiltonianPredictor(
    basis=basis,
    config=HamiltonianPredictorConfig(
        hidden_irreps="32x0e + 24x1o + 16x2e",  # must carry every degree the s/p blocks reach
        sh_lmax=2,
        num_interactions=2,
        cutoff=8.0,                               # Bohr
        property_name="hamiltonian",              # or "overlap" for S
    ),
    rngs=nnx.Rngs(0),
)
prediction = predictor(water)["hamiltonian"]   # (7, 7), symmetric and equivariant
```

A single trained predictor generalises across molecules: `predictor.rebind(basis)`
returns a copy bound to another molecule's AO basis, sharing the trunk and
per-pair-type expansion weights and swapping only the static block plan (a jit
recompile for the new atom/orbital count, as for any static-shape change).

## Validation against PySCF

The reference target is the converged restricted-Hartree-Fock Fock matrix and AO
overlap from [PySCF](https://pyscf.org/), run with `cart=True` so its Cartesian-AO
ordering matches opifex's STO-3G shell/AO layout exactly (atom-major; `s` shells
then `p` in `(x, y, z)`). Fitting the predictor to a single water geometry
reproduces the Fock matrix to a fraction of a milli-Hartree, and the
molecular-orbital energies of the predicted `H` (with the true `S`) match the RHF
eigenvalues — a physics-level check that the matrix is usable for the downstream
generalised eigenproblem, not just close in norm.

For an end-to-end run — PySCF ground truth, building and fitting the predictor,
reporting `H`/`S` MAE and orbital energies, and verifying the block-wise
equivariance under random rotations with `wigner_d` — see
[Equivariant DFT Hamiltonian Prediction](../examples/quantum-chemistry/hamiltonian-prediction.md).

## STO-3G needs no Cartesian-to-spherical transform

opifex's STO-3G covers H/C/N/O, whose shells are `s` and `p` only (`l ∈ {0, 1}`).
For `s` and `p` the Cartesian AO components coincide with the real
spherical-harmonic (irrep) components in the identical `(x, y, z)` order used by
`wigner_d`, so the predicted blocks land directly in opifex's AO ordering and no
Cartesian-to-spherical transform is needed. This is what makes the STO-3G example
fully self-contained.

## Extending

- **Larger bases with `d` orbitals (def2-SVP and up).** For `l ≥ 2` the number of
  Cartesian components (`GaussianShell.n_cartesian`) exceeds `2l + 1`, so the
  predicted spherical blocks must be mapped to the basis's Cartesian AOs through a
  Cartesian↔solid-harmonic transform before comparison with the integral engine.
  That transform belongs with the basis module and is the main step to reach the
  QH9 benchmark.
- **SO(2)-frame convolution (QHNetV2).** Yu et al. 2025
  ([arXiv:2506.09398](https://arxiv.org/abs/2506.09398)), building on eSCN
  (Passaro & Zitnick 2023, [arXiv:2302.03655](https://arxiv.org/abs/2302.03655)),
  rotate each edge into a local frame aligned with its direction, reducing the
  `SO(3)` Clebsch-Gordan tensor product to cheaper `SO(2)` operations — the main
  scalability lever for large bases. The present trunk uses the full `SO(3)`
  tensor product; the SO(2)-frame upgrade is a drop-in replacement for the edge
  tensor product.
- **The QH9 benchmark.** QH9 (Yu et al. 2023,
  [arXiv:2306.09549](https://arxiv.org/abs/2306.09549)) provides converged DFT
  Hamiltonians for QM9 molecules at the B3LYP/def2-SVP level — the standard
  training/evaluation target once the def2-SVP transform is in place.

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
