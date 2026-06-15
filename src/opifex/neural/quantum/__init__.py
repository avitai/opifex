"""Neural quantum chemistry modules for scientific machine learning.

The public surface spans three integral-independent families:

* the learned exchange-correlation functional
  (:class:`~opifex.neural.quantum.neural_xc.NeuralXCFunctional`) and the
  differentiable Kohn-Sham density-functional theory solver in
  :mod:`opifex.neural.quantum.dft` (:class:`~opifex.neural.quantum.dft.SCFSolver`);
* the neural-wavefunction / variational Monte Carlo stack in
  :mod:`opifex.neural.quantum.vmc` (:class:`~opifex.neural.quantum.vmc.FermiNet`).

The Kohn-Sham DFT names (:class:`SCFSolver`, :class:`SCFResult`,
:class:`Functional`, :class:`SolverMode`) are exposed **lazily** through
:pep:`562` ``__getattr__`` so that integral-free subpackages -- notably the VMC
family, which the task spec keeps free of any dependency on the Gaussian-integral
engine -- can be imported without pulling in the DFT grid/SCF machinery (and its
``opifex.core.quantum`` backend). The names remain importable exactly as before;
they are simply resolved on first access.
"""

from __future__ import annotations

from typing import Any

from opifex.neural.quantum._uq_capabilities import QUANTUM_CAPABILITIES
from opifex.neural.quantum.dft import Functional, SCFResult, SCFSolver, SolverMode
from opifex.neural.quantum.neural_xc import NeuralXCFunctional
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in QUANTUM_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


# Heavy DFT names resolved lazily so integral-free subpackages (e.g. ``vmc``)
# import without the Gaussian-integral backend.
_LAZY_DFT_NAMES = frozenset({"Functional", "SCFResult", "SCFSolver", "SolverMode"})


def __getattr__(name: str) -> Any:
    """Resolve the lazily-exposed Kohn-Sham DFT names on first access (:pep:`562`)."""
    if name in _LAZY_DFT_NAMES:
        from opifex.neural.quantum import dft

        return getattr(dft, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QUANTUM_CAPABILITIES",
    "Functional",
    "NeuralXCFunctional",
    "SCFResult",
    "SCFSolver",
    "SolverMode",
]
