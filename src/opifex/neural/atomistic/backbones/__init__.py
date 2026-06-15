r"""Concrete interatomic-potential backbones (embedding producers).

Each backbone is an :class:`flax.nnx.Module` satisfying
:class:`opifex.core.quantum.protocols.Backbone` and self-registering in the
``opifex.core.quantum.registry`` backbone registry, so importing this package
makes ``"schnet"``, ``"painn"`` and ``"nequip"`` discoverable by name:

* :class:`SchNet` -- invariant continuous-filter convolutions (Schuett 2018);
* :class:`PaiNN` -- equivariant scalar/vector message passing, ``l <= 1``
  (Schuett 2021);
* :class:`NequIP` -- E(3)-equivariant Clebsch-Gordan tensor-product message
  passing (Batzner 2022).

All three **compose opifex's Q0 equivariant kit** (:mod:`opifex.neural.equivariant`)
via the shared :mod:`opifex.neural.atomistic.backbones._message_passing` helper,
and emit per-atom invariant ``"node_features"`` consumed by the property heads.
"""

from opifex.neural.atomistic.backbones.nequip import NequIP, NequIPConfig
from opifex.neural.atomistic.backbones.painn import PaiNN, PaiNNConfig
from opifex.neural.atomistic.backbones.schnet import SchNet, SchNetConfig


__all__ = [
    "NequIP",
    "NequIPConfig",
    "PaiNN",
    "PaiNNConfig",
    "SchNet",
    "SchNetConfig",
]
