"""Solvers package for Opifex.

Exports all unified solvers and wrappers.
"""

from opifex.solvers._uq_capabilities import SOLVER_CAPABILITIES
from opifex.solvers.hybrid import HybridSolver
from opifex.solvers.neural_operator import NeuralOperatorSolver
from opifex.solvers.pinn import (
    heat_residual,
    helmholtz_residual,
    PINNConfig,
    PINNResult,
    PINNSolver,
    poisson_residual,
)
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.2. The standard deterministic
# neural model family + the three solver entry points + the solver-side
# UQ aggregation utilities (per Phase 6 Task 6.2 deletion of the four
# wrapper classes) each get a capability declaration in the singleton
# :class:`UQRegistry`. Guarded against double-registration on repeat
# imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in SOLVER_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "SOLVER_CAPABILITIES",
    "HybridSolver",
    "NeuralOperatorSolver",
    "PINNConfig",
    "PINNResult",
    "PINNSolver",
    "heat_residual",
    "helmholtz_residual",
    "poisson_residual",
]
