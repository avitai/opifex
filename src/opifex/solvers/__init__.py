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
