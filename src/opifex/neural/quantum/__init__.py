"""Neural quantum chemistry modules for scientific machine learning.

The public surface is the learned exchange-correlation functional
(:class:`~opifex.neural.quantum.neural_xc.NeuralXCFunctional`) and the
differentiable Kohn-Sham density-functional theory solver in
:mod:`opifex.neural.quantum.dft` (:class:`~opifex.neural.quantum.dft.SCFSolver`).
"""

from opifex.neural.quantum._uq_capabilities import QUANTUM_CAPABILITIES
from opifex.neural.quantum.dft import (
    Functional,
    SCFResult,
    SCFSolver,
    SolverMode,
)
from opifex.neural.quantum.neural_xc import NeuralXCFunctional
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in QUANTUM_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "QUANTUM_CAPABILITIES",
    "Functional",
    "NeuralXCFunctional",
    "SCFResult",
    "SCFSolver",
    "SolverMode",
]
