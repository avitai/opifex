# Atomistic Potentials

Machine-learning interatomic potentials (MLIPs) predict the energy of a set of atoms —
and, by differentiation, the forces and stress — from their positions and chemical
species. opifex builds them from the native E(3)-equivariant kit in
`opifex.neural.equivariant`, assembled through a small set of protocols so new
architectures and output properties plug in without touching existing code.

## Design: backbone → heads

An `opifex.neural.atomistic.AtomisticModel` is composed of three swappable pieces:

- a **backbone** (the `Backbone` protocol) that turns a `MolecularSystem` and a
  neighbour graph into per-atom features;
- one or more **property heads** (the `PropertyHead` protocol) that map those features
  to physical outputs;
- a **neighbour list** (the `NeighborList` protocol) that builds the edge graph within
  a cutoff radius.

Energy is an invariant scalar; **forces are the conservative gradient** `F = −∂E/∂x`
and stress is the virial from the strain derivative — both provided by `ForcesHead` and
`StressHead`, so any backbone gets correct, energy-conserving forces for free.

## Backbones

| Backbone | Symmetry | Mechanism | Reference |
|----------|----------|-----------|-----------|
| `SchNet` | E(3)-invariant | continuous-filter convolutions on radial features | Schütt et al. 2018 (arXiv:1706.08566) |
| `PaiNN` | E(3)-equivariant (scalar + vector) | gated equivariant message passing | Schütt, Unke & Gastegger 2021 (arXiv:2102.03150) |
| `NequIP` | E(3)-equivariant (steerable tensors) | Clebsch–Gordan tensor-product convolutions on spherical-harmonic edges | Batzner et al. 2022 (arXiv:2101.03164) |

All three emit per-atom invariant scalar features for the energy head and are
registered under `opifex.core.quantum.registry` (`register_backbone`), so they can also
be built by name.

## Example

```python
import jax.numpy as jnp
import flax.nnx as nnx

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.backbones import NequIP, NequIPConfig
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead

rngs = nnx.Rngs(0)
backbone = NequIP(
    config=NequIPConfig(
        hidden_irreps="32x0e + 8x1o",
        sh_lmax=2,
        num_interactions=2,
        num_radial_basis=8,
        cutoff=5.0,
    ),
    rngs=rngs,
)
model = AtomisticModel(
    backbone=backbone,
    heads={"energy": EnergyHead(feature_dim=32, rngs=rngs), "forces": ForcesHead()},
    neighbor_list=RadiusNeighborList(cutoff=5.0),
    max_edges=64,
)

water = MolecularSystem(
    atomic_numbers=jnp.array([8, 1, 1]),
    positions=jnp.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
)
prediction = model(water)  # {"energy": (), "forces": (3, 3)}
```

The energy is invariant and the forces are equivariant under rotation, translation, and
permutation of identical atoms. Train the model with the energy+forces objective in
`opifex.neural.atomistic.training` (`make_atomistic_train_step` / `fit_atomistic`) against
reference energies and forces.

For an end-to-end example — downloading the rMD17 aspirin benchmark, normalizing energies,
and training NequIP toward the published rMD17 @1000 energy/force accuracy with
visualization — see
[NequIP on rMD17 (Aspirin)](../examples/atomistic/nequip-md17.md).

## Property heads

Each `PropertyHead` owns a single property family (`opifex.neural.atomistic.heads`):

- **`EnergyHead`** — invariant total energy `E = Σ_i ε_i`.
- **`ForcesHead`** — conservative forces `F = −∇_r E` (autodiff).
- **`StressHead`** — virial / stress via strain-displacement autodiff.
- **`ChargeHead`** — per-atom partial charges with total-charge conservation
  (`Σ_i q_i = Q`).
- **`DipoleHead`** — molecular dipole `μ = Σ_i q_i r_i` (rotationally equivariant).
- **`EvidentialEnergyHead`** — single-pass uncertainty (below).

## Single-pass uncertainty (evidential)

`EvidentialEnergyHead` predicts the Normal-Inverse-Gamma parameters of a Deep Evidential
Regression output (Amini et al. 2020; evidential interatomic potentials,
[arXiv:2407.13994](https://arxiv.org/abs/2407.13994)), giving **per-prediction epistemic and
aleatoric uncertainty in one forward pass** — no ensemble or dropout sampling. `EvidentialAdapter`
(or `opifex.uncertainty.nig_to_predictive_distribution`) maps its output to the platform
`PredictiveDistribution`, so it plugs into the existing `opifex.uncertainty` machinery
(calibration, active learning).

## Fine-tuning foundation models

`opifex.neural.atomistic.foundation` provides the transfer-learning mechanism:
`remap_element_table` remaps a pretrained checkpoint's per-element parameters onto a new element
set, `freeze_backbone` / `trainable_filter` keep the backbone fixed while training new heads, and
`LoRALinear` / `apply_lora` add low-rank equivariant adapters (`W_eff = W + (α/r)·B·A`, after
MACE's `lora.py`) for parameter-efficient fine-tuning.

## ASE calculator / MD

`opifex.deployment.ase_calculator.OpifexCalculator` wraps any `AtomisticModel` as an ASE
`Calculator`, driving ASE relaxations and molecular dynamics on the jitted forward path
(energies in eV, forces in eV/Å, stress as the Voigt 6-vector). Requires the optional `ase`
dependency.

## Extending

- **A new architecture** — implement the `Backbone` protocol and decorate it with
  `@register_backbone("name")`; it composes with the existing heads unchanged.
- **A new output property** (dipole, charges, …) — implement the `PropertyHead`
  protocol and add it to the model's `heads`.
- **Periodic systems** — supply a periodic `Space` (`opifex.core.quantum.space`); the
  conservative stress head uses the strain derivative.

The shared equivariant primitives — irreducible representations, Clebsch–Gordan tensor
products, spherical harmonics, gated nonlinearities, and radial bases — live in
`opifex.neural.equivariant` and are reused across the atomistic and
equivariant-property families.
