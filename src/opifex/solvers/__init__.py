"""Solvers package for Opifex.

Exports all unified solvers and wrappers.
"""

from .hybrid import HybridSolver
from .neural_operator import NeuralOperatorSolver
from .pinn import (
    heat_residual,
    helmholtz_residual,
    PINNConfig,
    PINNResult,
    PINNSolver,
    poisson_residual,
)
from .wrappers import BayesianWrapper, ConformalWrapper, L2OWrapper


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
