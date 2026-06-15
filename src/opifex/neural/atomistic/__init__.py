r"""Machine-learning interatomic potentials (MLIPs) -- backbone + named heads.

This package hosts the assembled atomistic models following the convergent
**backbone -> typed property heads** design of SchNetPack-2 / fairchem / MACE:

* :class:`~opifex.neural.atomistic.base.AtomisticModel` -- the reusable assembly
  (orchestrates a backbone + heads + neighbour list; owns the conservative
  force/stress autodiff closures);
* :mod:`opifex.neural.atomistic.heads` -- energy / forces / stress readouts;
* :mod:`opifex.neural.atomistic.backbones` -- the concrete embedding producers
  :class:`SchNet` (invariant), :class:`PaiNN` (equivariant ``l <= 1``) and
  :class:`NequIP` (E(3) tensor-product).

The backbones plug into :class:`AtomisticModel` via the
:class:`opifex.core.quantum.protocols.Backbone` protocol and the
``opifex.core.quantum.registry`` family registries, without editing the assembly
(Open-Closed). A MACE-style higher-body-order backbone is a documented future
wave. The package is domain-pure: it imports no infrastructure (ASE / training /
serving live in their own layers).
"""

from opifex.neural.atomistic.backbones import (
    NequIP,
    NequIPConfig,
    PaiNN,
    PaiNNConfig,
    SchNet,
    SchNetConfig,
)
from opifex.neural.atomistic.base import AtomisticModel
from opifex.neural.atomistic.conditioning import (
    ChargeSpinConditioning,
    ChargeSpinConditioningConfig,
)
from opifex.neural.atomistic.foundation import (
    FineTuneConfig,
    freeze_backbone,
    remap_element_table,
    trainable_filter,
)
from opifex.neural.atomistic.long_range import latent_ewald_energy, LatentEwaldHead
from opifex.neural.atomistic.lora import apply_lora, LoRALinear
from opifex.neural.atomistic.multitask import MultiTaskEnergyHead, TASK_NAME_KEY
from opifex.neural.atomistic.scale_shift import AtomicScaleShift, fit_atomic_scale_shift
from opifex.neural.atomistic.training import (
    AtomisticBatch,
    energy_forces_loss,
    fit_atomistic,
    make_atomistic_train_step,
    make_scanned_epoch,
    ParamEMA,
)


__all__ = [
    "TASK_NAME_KEY",
    "AtomicScaleShift",
    "AtomisticBatch",
    "AtomisticModel",
    "ChargeSpinConditioning",
    "ChargeSpinConditioningConfig",
    "FineTuneConfig",
    "LatentEwaldHead",
    "LoRALinear",
    "MultiTaskEnergyHead",
    "NequIP",
    "NequIPConfig",
    "PaiNN",
    "PaiNNConfig",
    "ParamEMA",
    "SchNet",
    "SchNetConfig",
    "apply_lora",
    "energy_forces_loss",
    "fit_atomic_scale_shift",
    "fit_atomistic",
    "freeze_backbone",
    "latent_ewald_energy",
    "make_atomistic_train_step",
    "make_scanned_epoch",
    "remap_element_table",
    "trainable_filter",
]
