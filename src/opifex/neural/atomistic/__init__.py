r"""Machine-learning interatomic potentials (MLIPs) -- backbone + named heads.

This package hosts the assembled atomistic models following the convergent
**backbone -> typed property heads** design of SchNetPack-2 / fairchem / MACE:

* :class:`~opifex.neural.atomistic.base.AtomisticModel` -- the reusable assembly
  (orchestrates a backbone + heads + neighbour list; owns the conservative
  force/stress autodiff closures);
* :mod:`opifex.neural.atomistic.heads` -- energy / forces / stress readouts.

Concrete backbones (SchNet, PaiNN, NequIP, MACE) are a later wave: they plug into
:class:`AtomisticModel` via the
:class:`opifex.core.quantum.protocols.Backbone` protocol and the
``opifex.core.quantum.registry`` family registries, without editing this package
(Open-Closed). The package is domain-pure: it imports no infrastructure (ASE /
training / serving live in their own layers).
"""

from opifex.neural.atomistic.base import AtomisticModel


__all__ = ["AtomisticModel"]
