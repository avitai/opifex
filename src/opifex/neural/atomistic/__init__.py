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
from opifex.neural.atomistic.scale_shift import AtomicScaleShift, fit_atomic_scale_shift
from opifex.neural.atomistic.training import (
    AtomisticBatch,
    energy_forces_loss,
    fit_atomistic,
    make_atomistic_train_step,
)


__all__ = [
    "AtomicScaleShift",
    "AtomisticBatch",
    "AtomisticModel",
    "NequIP",
    "NequIPConfig",
    "PaiNN",
    "PaiNNConfig",
    "SchNet",
    "SchNetConfig",
    "energy_forces_loss",
    "fit_atomic_scale_shift",
    "fit_atomistic",
    "make_atomistic_train_step",
]
