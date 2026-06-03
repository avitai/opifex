# NequIP on rMD17 (Aspirin)

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~57 min (GPU) |
| **Prerequisites** | JAX, Flax NNX, E(3)-equivariant networks, MLIPs |
| **Format** | Python + Jupyter |
| **Memory** | ~3 GB RAM |

## Overview

This example trains a [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
machine-learning interatomic potential (MLIP) on the aspirin molecule from the
revised MD17 (rMD17) benchmark, fitting **energies and conservative forces**
jointly. NequIP (Batzner et al. 2022, arXiv:2101.03164) is an E(3)-equivariant
message-passing network: node features are steerable spherical tensors updated by
Clebsch-Gordan tensor-product convolutions on spherical-harmonic edge embeddings.
The total energy is an invariant scalar and the forces are its conservative
gradient `F = -dE/dx`, so they are energy-consistent by construction.

The example is deliberately **thin** — it composes opifex's committed atomistic
stack and changes no library internals:

- `create_rmd17_loader` downloads and caches aspirin, builds the canonical
  1000/1000 train/validation split, and yields stacked `{positions, energy,
  forces}` batches.
- `fit_atomic_scale_shift` fits the per-atom energy scale-shift on the training
  split, so the network learns only the small, well-conditioned interaction
  energy.
- `NequIP` + `EnergyHead` + `ForcesHead` assemble into an `AtomisticModel`.
- `make_atomistic_train_step` builds the jitted energy+forces training step
  (`nnx.value_and_grad` + `optimizer.update`); the force term trains the model
  through grad-of-grad autodiff.
- `create_optimizer` supplies AdamW with a cosine learning-rate schedule.
- `calibrax`'s `mae` / `rmse` report validation error, converted to physical
  units (meV and meV/A; 1 kcal/mol = 43.364 meV).

## What You'll Learn

1. **Load** a real MLIP benchmark with `create_rmd17_loader`
2. **Normalize** total energies with `fit_atomic_scale_shift`
3. **Assemble** a NequIP `AtomisticModel` with energy and conservative-force heads
4. **Train** the joint energy+forces objective with the jitted atomistic step
5. **Evaluate** energy/force MAE and RMSE in physical units
6. **Visualize** energy parity, force-component parity, and the loss curve

## Background: equivariant interatomic potentials

An interatomic potential maps atomic positions and species to a potential energy.
For the dynamics to conserve energy, the forces must be the exact gradient of that
energy. NequIP enforces both physical symmetries directly:

- **Energy is E(3)-invariant** — unchanged under rotation, translation, and
  reflection of the molecule, and under permutation of identical atoms.
- **Forces are E(3)-equivariant** — they rotate with the molecule, because they
  are obtained by differentiating the invariant energy through `ForcesHead`.

opifex builds NequIP from the native equivariant kit in
`opifex.neural.equivariant` (irreps, tensor products, spherical harmonics, gated
nonlinearities, Bessel radial bases). See
[Atomistic Potentials](../../methods/atomistic-potentials.md) for the
backbone → heads design.

## Data and normalization

```python
from opifex.data.loaders import create_rmd17_loader
from opifex.neural.atomistic import AtomisticBatch, fit_atomic_scale_shift
import jax.numpy as jnp

loaders = create_rmd17_loader(
    molecule="aspirin", n_train=1000, n_val=1000, batch_size=5, seed=0
)
atomic_numbers = jnp.asarray(loaders.atomic_numbers)  # (21,) for aspirin (C9H8O4)
```

Iterating a datarax pipeline once yields the split's `{positions, energy,
forces}` batches as stacked arrays sharing one `atomic_numbers` vector;
`AtomisticBatch.from_arrays` packs each into the JAX PyTree the jitted step
consumes — no per-configuration `MolecularSystem` round-trip:

```python
def collect_batches(pipeline):
    return [
        AtomisticBatch.from_arrays(
            jnp.asarray(r["positions"]), atomic_numbers,
            jnp.asarray(r["energy"]), jnp.asarray(r["forces"]),
        )
        for r in pipeline
    ]

train_batches = collect_batches(loaders.train)
val_batches = collect_batches(loaders.val)
```

The total energy of aspirin is dominated by a near-constant sum of per-atom
reference energies (~ -400000 kcal/mol), so fitting it directly is badly
conditioned. `fit_atomic_scale_shift` fits the MACE/NequIP affine readout
`E = scale * E_raw + n_atoms * shift` on the **training** split — `shift` is the
mean per-atom energy and `scale` is the spread of the residual interaction energy
— and is passed into the `EnergyHead`:

```python
train_energies = jnp.concatenate([b.energies for b in train_batches])
atom_counts = jnp.full(train_energies.shape, float(atomic_numbers.shape[0]))
scale_shift = fit_atomic_scale_shift(train_energies, atom_counts)
```

## Model assembly

The `EnergyHead`'s `feature_dim` equals the number of `0e` scalar channels in the
hidden irreps (64 here). The hyper-parameters follow the NequIP rMD17 recipe
(Batzner et al. 2022, SI; the reference Flax implementation
`jax_md/_nn/nequip.py`, which uses ~64-128 feature channels, `l_max = 2` and 5
interaction layers): steerable features up to `l_max = 2`, **five**
tensor-product convolution layers with **64** scalar channels, an 8-function
Bessel radial basis, and a 5 A cutoff.

```python
from flax import nnx
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.backbones import NequIP, NequIPConfig
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead

rngs = nnx.Rngs(0)
backbone = NequIP(
    config=NequIPConfig(
        hidden_irreps="64x0e + 32x1o + 16x2e",
        sh_lmax=2,
        num_interactions=5,
        num_radial_basis=8,
        radial_hidden_dim=64,
        cutoff=5.0,
        average_num_neighbors=14.4,
    ),
    rngs=rngs,
)
model = AtomisticModel(
    backbone=backbone,
    heads={
        "energy": EnergyHead(feature_dim=64, scale_shift=scale_shift, rngs=rngs),
        "forces": ForcesHead(),
    },
    neighbor_list=RadiusNeighborList(cutoff=5.0),
    max_edges=atomic_numbers.shape[0] ** 2,
)
```

## Training

`make_atomistic_train_step` returns an `nnx.jit` step that computes the weighted
energy+forces loss, its gradient with `nnx.value_and_grad`, and applies the AdamW
update in place. Forces carry `3 * n_atoms` labels per structure and dominate the
chemistry of the potential-energy surface, so NequIP / MACE weight the force term
heavily — but too large a force weight *starves the absolute-energy term*: the
relative energies and forces converge while the constant energy offset never does.
The example therefore trains in **two phases**, both built from the same thin
`make_atomistic_train_step` and sharing one optimizer (so the cosine schedule
advances continuously across the boundary):

1. an **energy warm-up** — the first **80 epochs** at a moderate force weight
   (`force_weight = 5`) so the absolute per-structure energy offset converges while
   the forces already start improving, then
2. the **main phase** — the remaining **470 epochs** at a heavier force weight
   (`force_weight = 150`) that refines the forces without letting the energy term
   diverge again.

Switching the weight rebuilds the jitted step once (a single recompile). The force
error keeps falling through the low-learning-rate tail of the cosine schedule, so
AdamW uses a *deep* cosine decay (to ~`1e-3` of the peak rate) with global-norm
gradient clipping over the whole **550-epoch** run; the clip keeps the
force-weighted grad-of-grad objective stable.

```python
from opifex.core.training import OptimizerConfig
from opifex.core.training.optimizers import create_optimizer
from opifex.neural.atomistic import make_atomistic_train_step

WARMUP_EPOCHS, MAIN_EPOCHS = 80, 470
num_epochs = WARMUP_EPOCHS + MAIN_EPOCHS  # 550

steps_per_epoch = len(train_batches)
optimizer = nnx.Optimizer(
    model,
    create_optimizer(OptimizerConfig(
        optimizer_type="adamw", learning_rate=4e-3, weight_decay=1e-5,
        schedule_type="cosine", decay_steps=num_epochs * steps_per_epoch, alpha=1e-3,
        gradient_clip=1.0, clip_type="by_global_norm",
    )),
    wrt=nnx.Param,
)
# Two jitted steps from the same factory, one per phase weight, sharing the one
# optimizer so the single cosine schedule spans the whole two-phase run.
warmup_step = make_atomistic_train_step(
    model, optimizer, energy_weight=1.0, force_weight=5.0
)
main_step = make_atomistic_train_step(
    model, optimizer, energy_weight=1.0, force_weight=150.0
)

for epoch in range(num_epochs):
    train_step = warmup_step if epoch < WARMUP_EPOCHS else main_step
    for batch in train_batches:
        train_step(model, optimizer, batch)
        ema.update(model)  # blend the live weights into the EMA shadow each step
```

Because the forces are the energy gradient, the force term differentiates a
gradient — the backbone is jit / grad / vmap clean for exactly this grad-of-grad
path.

Following NequIP/MACE, the example keeps an **exponential moving average of the
weights** (`ParamEMA`, decay `0.99` — the NequIP `ema_decay` default and the MACE
`--ema_decay` default) and evaluates against the *smoothed* weights, not the noisy
last-step ones. Seeded once from the initial model (`ema = ParamEMA(model,
decay=0.99)`), the library `ParamEMA` holds a shadow `nnx.Param` state, is updated
after every step (`ema.update(model)`, shown above), and exposes
`ema.swap_in(model)` (a context that loads the averaged weights for evaluation and
restores the live weights) and `ema.copy_to(model)` (load the averaged weights
permanently for the final report). All validation metrics and parity plots below
are reported against the EMA weights.

## Results

Validation error on the 1000-configuration aspirin test split, measured on a
single GPU run of this example (550 epochs, ~57 min on one RTX 4090) and evaluated
against the EMA (smoothed) weights. The published column is the canonical rMD17
@1000 benchmark — models trained on 950 configurations and validated on 50 — from
Batzner et al. 2022 (NequIP, Nat. Commun.) as tabulated in Batatia et al. 2022
(MACE, NeurIPS, Table 1), which reports NequIP and MACE on the same split:

| Metric | This example | NequIP @1000 | MACE @1000 |
|--------|-------------:|-------------:|-----------:|
| Energy MAE  | **7.29 meV**   | ~2.3 meV   | ~2.2 meV   |
| Energy RMSE | 10.68 meV      | —          | —          |
| Force MAE   | **20.19 meV/A** | ~8 meV/A  | ~6.6 meV/A |
| Force RMSE  | 31.47 meV/A    | —          | —          |

This opifex NequIP is a two-body (`correlation = 1`) tensor-product potential,
without MACE's higher-body-order symmetric contraction. The two-phase loss
(energy warm-up then heavier-force main phase) and the EMA evaluation weights
**converge the absolute energy offset**: the energy MAE of **7.29 meV** is stable
and single-digit-meV, ~3x the published ~2.3 meV NequIP reference rather than the
hundreds-of-meV offset a single very large force weight leaves behind. On the same
@1000 aspirin benchmark the force MAE of **20.19 meV/A** **approaches — but does
not yet match** — the published ~8 meV/A NequIP accuracy, within a small factor
(~2.5x). Both targets now land in the same ballpark instead of the energy term
being starved by the force term.

The remaining gap reflects the smaller model (`l_max = 2`, five layers, two-body
messages) and the finite training budget; the canonical NequIP/MACE numbers come
from larger models trained substantially longer. Closing it needs `l_max = 3`,
wider features, the MACE-style higher-body contraction, and a longer schedule.

### Energy and force parity

![Energy parity (predicted vs reference mean-centred total energy) and force-component parity (every Cartesian force component) on the aspirin validation split](../../assets/examples/nequip_rmd17/parity.png)

The energy parity is mean-centred so the residual interaction energy the network
actually learns is visible; the force parity shows all `3 * n_atoms` Cartesian
components across the validation split.

### Training loss

![Weighted energy+forces training loss per epoch on a log scale, with a marked step at the warm-up to main phase boundary where the force weight jumps from 5 to 150](../../assets/examples/nequip_rmd17/loss_curve.png)

## Running the example

```bash
uv run python examples/atomistic/nequip_md17.py
```

The first run downloads the aspirin npz from figshare (~50 MB) and caches it
under `~/.cache/opifex/rmd17`; subsequent runs reuse the cache.

## Key takeaways

- A NequIP MLIP is a **thin composition** of the opifex atomistic stack:
  loader → scale-shift → backbone + heads → jitted energy+forces step → metrics.
- **Energy normalization is essential** — fitting the raw total energy is badly
  conditioned; the per-atom scale-shift makes the network learn the small
  interaction energy.
- **Conservative forces come for free** — `ForcesHead` differentiates the energy,
  so forces are energy-consistent and the joint loss is a grad-of-grad objective
  the equivariant backbone handles under `jit` / `grad` / `vmap`.
- **A two-phase loss converges both targets** — an energy warm-up at a moderate
  force weight settles the absolute energy offset before the main phase raises the
  force weight, and EMA evaluation weights keep the reported metrics stable; the
  energy MAE lands at single-digit meV instead of being starved by the force term.
- The recipe **approaches** the published NequIP force accuracy on the rMD17
  @1000 aspirin benchmark with a two-body model, and the gap closes with
  `l_max = 3`, wider features, the MACE-style higher-body contraction, and a
  longer training schedule.

## See also

- [Atomistic Potentials](../../methods/atomistic-potentials.md) — the
  backbone → heads design and the available backbones (SchNet, PaiNN, NequIP).
- Batzner et al. 2022, *E(3)-equivariant graph neural networks for data-efficient
  and accurate interatomic potentials*, Nat. Commun. 13, 2453
  ([arXiv:2101.03164](https://arxiv.org/abs/2101.03164)).
- Batatia et al. 2022, *MACE: Higher Order Equivariant Message Passing Neural
  Networks for Fast and Accurate Force Fields*, NeurIPS 2022
  ([arXiv:2206.07697](https://arxiv.org/abs/2206.07697)) — Table 1 reports the
  canonical rMD17 @1000 errors for NequIP, MACE, Allegro, and others.
