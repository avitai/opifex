# Higher-body-order NequIP: the symmetric contraction (`correlation > 1`)

## Motivation

opifex's `NequIP` backbone is a faithful **two-body** (`correlation = 1`)
tensor-product potential: each interaction tensors a node's features with the
spherical harmonics of one edge, so a single message carries two-body
(center + one neighbour) information and body order grows only with depth. On the
rMD17 aspirin benchmark this plateaus at a force MAE of ~15 meV·Å⁻¹ — within ~1.9×
of the published NequIP (~8 meV·Å⁻¹) — and that gap is **robust** across precision
(float32/float64), `l_max`, the `linear_up`/`linear_down` mixings, the
spherical-harmonic / gate / energy-scale conditioning fixes, the per-element
self-connection, and the full per-atom-energy + `ReduceLROnPlateau` training recipe.
Every one of those levers was verified *not* to close it.

The remaining axis is **body order**. MACE (Batatia et al. 2022, *MACE: Higher Order
Equivariant Message Passing*, NeurIPS) raises each message to a high body order with
the Atomic Cluster Expansion symmetric contraction (Drautz 2019), reaching
~6.6 meV·Å⁻¹ on the same benchmark. `NequIPConfig.correlation` already reserves this
knob; today only `correlation = 1` is implemented and `> 1` raises a `ValueError`.
This document specifies how to build `correlation > 1`.

## Background: the product basis

After the two-body convolution + neighbour pooling, each node holds a
**single-particle basis** `A_i` — an equivariant feature of shape
`(channels, irreps_in.dim)` summarising the atom's environment. A body order of
`ν + 1` is reached by taking the **symmetric `ν`-th tensor power** of `A_i` and
projecting it back onto each output irrep:

```
B_i = Linear( concat_ν [ A_i,  sym(A_i ⊗ A_i),  sym(A_i ⊗ A_i ⊗ A_i),  … ] )
```

The projection onto irrep `ir_out` of the `ν`-fold symmetric power uses a
precomputed **generalized Clebsch-Gordan tensor** `U^(ν)` (an orthonormal basis of
the symmetric coupling paths), contracted with learnable per-path weights `w^(ν)`:

```
B_i^(ν)[ir_out] = Σ_k  U^(ν)[…, k] · w^(ν)_k · (A_i ⊗ … ⊗ A_i)     (ν factors)
```

`correlation = ν_max` is the maximum order; MACE's default `correlation = 3` gives up
to 4-body messages.

## Algorithm (faithful to MACE / e3nn-jax)

### U tensors (precomputed constants)
`U^(ν)` for a given `(irreps_in, irreps_out, ν)` is an orthonormal change of basis
from the symmetric `ν`-th power of `irreps_in` onto `irreps_out`, built by recursively
coupling `ν` copies with real Clebsch-Gordan and then imposing permutation symmetry
(Gram-Schmidt over the symmetric group). It is a **constant** — not learned — of shape
`[irreps_out.dim] + [irreps_in.dim] * ν + [num_paths]`, where `num_paths` is the number
of independent symmetric coupling paths. References: MACE `U_matrix_real` / `_wigner_nj`;
e3nn-jax `reduced_symmetric_tensor_product_basis` (`reduce_basis_product` then
`constrain_rotation_basis_by_permutation_basis`).

### Horner recursion (the runtime contraction)
The naive `ν`-fold outer product is `O(irreps_in.dim^ν)`; both MACE and e3nn-jax avoid
it with a Horner factorization that builds order `ν` from order `ν−1` by contracting in
one more copy of `A`:

```
out = U^(ν_max) · w^(ν_max) · A            # highest order
for ν from ν_max-1 down to 1:
    c   = U^(ν) · w^(ν)                     # fold weights into the U paths
    out = (c + out) · A                     # accumulate, then multiply in one more A
return out                                   # == ((w_ν·A + w_{ν-1})·A + w_{ν-2})·A …
```

e3nn-jax forward einsums (per output irrep, the cleanest reference):
`"...jki,kc,cj->c...i"` (highest), `"...ki,kc->c...i"` (weight fold),
`"c...ji,cj->c...i"` (multiply in `A`); `U` is normalized by `U / U.shape[-2]`.

### Learnable weights
One weight tensor per order `ν`, of shape `(num_elements, num_paths, num_channels)`,
initialised `randn / num_paths`. The **element axis** makes the contraction
chemistry-aware: a one-hot atom type selects the per-species weight slab inside the
einsum (`"ekc,be->bc…"`). This is MACE's extension over e3nn-jax (which has no element
axis). It reuses the species one-hot opifex already builds for the species-indexed
self-connection (`NequIPConfig.species`).

### Block wiring
Per layer MACE is **Interaction → Product(symmetric contraction) → Linear + residual**:

1. the two-body convolution forms the message `A_i` and a self-connection `sc`
   (opifex already has both: the edge tensor product + the species-indexed skip);
2. the **product block** raises `A_i` to body order `≤ ν_max + 1` via the symmetric
   contraction, then applies `EquivariantLinear` and adds `sc`.

So the symmetric contraction **augments** — does not replace — the existing two-body
interaction.

## What opifex already has vs. what is new

Reusable today (`opifex.neural.equivariant`): `Irreps`/`IrrepsArray`, `clebsch_gordan`,
`FullyConnectedTensorProduct` and `ChannelwiseTensorProduct` (both store CG as frozen
nested-tuple static aux-data and rebuild it as a compile-time constant in `__call__` —
the exact pattern the `U` tensors should follow), `EquivariantLinear`, `from_chunks`,
`spherical_harmonics`, `scatter_sum`, `gate`/`NormGate`, the radial bases, and the
NequIP interaction block with its species one-hot.

Genuinely new numerical machinery (only the U-matrix builder is novel; CG already
exists):

1. **`reduced_symmetric_tensor_product_basis(irreps_in, degree, keep_ir)`** — the U
   builder. Port the e3nn-jax recursion: couple `ν` copies via
   `sqrt(ir.dim) · clebsch_gordan(...)`, then `constrain_rotation_basis_by_permutation_basis`
   for the symmetric permutation symmetry. Build order `ν` from order `ν−1`.
2. **`SymmetricContraction` (`nnx.Module`)** — holds the `U^(ν)` as static constants
   (mirror the `_to_nested_tuple` pattern), `(num_elements, num_paths, num_channels)`
   `nnx.Param` weights per order, and the Horner-recursion forward.
3. **`EquivariantProductBasisBlock`** — `SymmetricContraction → EquivariantLinear + sc`.

## Integration into `NequIP`

- `_ConvolutionLayer`: when `config.correlation > 1`, reshape the aggregated message
  to the single-particle basis `(channels, irreps)` and run it through the product
  block before the gate/residual; the species one-hot already threaded for the skip
  feeds the element-indexed weights. `correlation = 1` keeps today's path unchanged.
- Lift the `correlation == 1` guard in `NequIP.__init__` once the block exists; keep
  `correlation` per-layer-configurable (MACE allows a per-interaction list).
- Requires uniform-multiplicity hidden irreps (the contraction is channel-wise), e.g.
  `"64x0e + 64x1o + 64x2e"`; document the constraint and validate it in the config.

## Testing (TDD)

1. **U-matrix unit tests**: orthonormality, the expected `num_paths` for small
   `(irreps_in, ν)`, and **invariance of the contraction under the symmetric group**
   (permuting the `ν` factors leaves `B` unchanged).
2. **Equivariance**: a `SymmetricContraction` and a `correlation>1` `NequIP` must pass
   the shared backbone contracts (`tests/.../backbones/_helpers.py`): energy invariance,
   force equivariance, `F = −∇E` vs finite differences, jit/grad/vmap cleanliness.
3. **Reference parity**: check the order-2 contraction against an explicit
   `sym(A⊗A)` + CG projection computed independently, and (optionally) against e3nn-jax
   on a fixed small input.
4. **Body-order sanity**: a `correlation=3` model fits a synthetic 3-body target a
   `correlation=1` model cannot.
5. **Example**: re-run `nequip_md17` at `correlation=3` and confirm the force MAE drops
   toward the published ~6.6–8 meV·Å⁻¹.

## Cost, risks, defaults

- **Precompute**: building `U^(ν)` is combinatorial in `ν` and `irreps_in.dim`; do it
  once at construction and cache as static constants. `ν = 4` typically restricts the
  intermediate irreps; `ν > 3` is rarely worth it.
- **Runtime**: the Horner recursion is the key optimization — order `ν` reuses order
  `ν−1`. Let XLA fold the constant `U` (as the existing TPs do with CG). Expect a
  per-layer cost increase over the two-body TP; budget a larger example runtime.
- **Memory**: the element axis multiplies *parameter* count by `num_elements` but not
  per-node FLOPs (one slab gathered per node).
- **Defaults** (match MACE): `correlation = 3`, uniform `num_channels`,
  `irrep_normalization="component"`, weight init `randn / num_paths`.

## Acceptance criteria

- `reduced_symmetric_tensor_product_basis` + `SymmetricContraction` +
  `EquivariantProductBasisBlock` land as tested `opifex.neural.equivariant` /
  `opifex.neural.atomistic` building blocks (the contraction is generic, reusable
  beyond NequIP).
- `NequIP(correlation=3)` passes every backbone physics contract.
- `nequip_md17` at `correlation=3` measurably closes the force-MAE gap toward the
  published MACE/NequIP numbers; the example documents the body-order/runtime tradeoff
  honestly.

## References

- Batatia, Kovács, Simm, Ortner, Csányi, *MACE: Higher Order Equivariant Message
  Passing Interatomic Potentials*, NeurIPS 2022 (arXiv:2206.07697).
- Drautz, *Atomic cluster expansion for accurate and transferable interatomic
  potentials*, Phys. Rev. B 99, 014104 (2019).
- Batzner et al., *E(3)-equivariant graph neural networks…*, Nat. Commun. 13, 2453
  (2022) (arXiv:2101.03164) — the two-body baseline.
- The `e3nn-jax` `reduced_symmetric_tensor_product_basis` and `SymmetricTensorProduct`
  provide the canonical JAX reference for the U-matrix builder and the contraction
  forward.
