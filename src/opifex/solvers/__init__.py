"""Solvers package for Opifex.

Exports all unified solvers and wrappers.
"""

from opifex.solvers._uq_capabilities import SOLVER_CAPABILITIES
from opifex.solvers.hybrid import additive_hybrid
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


# The solver entry points and the solver-side UQ aggregation utilities each
# declare their capabilities in the singleton :class:`UQRegistry`, guarded so a
# repeated import registers nothing twice.
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in SOLVER_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "SOLVER_CAPABILITIES",
    "NeuralOperatorSolver",
    "PINNConfig",
    "PINNResult",
    "PINNSolver",
    "additive_hybrid",
    "heat_residual",
    "helmholtz_residual",
    "poisson_residual",
]
