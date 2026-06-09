r"""Typed property heads for atomistic models (backbone -> named outputs).

Each head owns exactly one property family (single responsibility) and satisfies
the :class:`opifex.core.quantum.protocols.PropertyHead` protocol:

* :class:`EnergyHead` -- sum of per-atom scalar energies (invariant total energy);
* :class:`ForcesHead` -- conservative forces ``-grad(E)`` (autodiff);
* :class:`StressHead` -- virial / stress via strain-displacement autodiff.

Conservative force/stress are the default strategies; direct-readout variants
plug into the same protocol later.
"""

from opifex.neural.atomistic.heads.charge import ChargeHead, conserve_total_charge
from opifex.neural.atomistic.heads.dipole import DipoleHead
from opifex.neural.atomistic.heads.energy import EnergyHead
from opifex.neural.atomistic.heads.evidential import EvidentialEnergyHead
from opifex.neural.atomistic.heads.forces import ENERGY_FN_KEY, ForcesHead
from opifex.neural.atomistic.heads.stress import STRAIN_ENERGY_FN_KEY, StressHead


__all__ = [
    "ENERGY_FN_KEY",
    "STRAIN_ENERGY_FN_KEY",
    "ChargeHead",
    "DipoleHead",
    "EnergyHead",
    "EvidentialEnergyHead",
    "ForcesHead",
    "StressHead",
    "conserve_total_charge",
]
