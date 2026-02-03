"""Solvers package for Opifex.

Exports all unified solvers and wrappers.
"""

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
from opifex.solvers.wrappers import BayesianWrapper, ConformalWrapper, L2OWrapper


__all__ = [
    "BayesianWrapper",
    "ConformalWrapper",
    "HybridSolver",
    "L2OWrapper",
    "NeuralOperatorSolver",
    "PINNConfig",
    "PINNResult",
    "PINNSolver",
    "heat_residual",
    "helmholtz_residual",
    "poisson_residual",
]
