"""Solvers package for Opifex.

Exports all unified solvers and wrappers.
"""

from opifex.solvers._uq_capabilities import SOLVER_CAPABILITIES
from opifex.solvers.hybrid import additive_hybrid
from opifex.solvers.neural_operator import NeuralOperatorSolver
from opifex.solvers.nonlinear_least_squares import (
    levenberg_marquardt,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
)
from opifex.solvers.pinn import (
    heat_residual,
    helmholtz_residual,
    PINNConfig,
    PINNResult,
    PINNSolver,
    poisson_residual,
)
from opifex.solvers.stiff_ode import (
    GridKind,
    InitialValueProblem,
    Kvaerno3,
    Rodas4,
    Ros34PW2,
    solve_stiff_ode,
)
from opifex.solvers.ude import (
    create_ude,
    NeuralODE,
    tsit5_solve,
    UDEConfig,
    UDESolver,
)


__all__ = [
    "SOLVER_CAPABILITIES",
    "GridKind",
    "InitialValueProblem",
    "Kvaerno3",
    "LevenbergMarquardt",
    "NeuralODE",
    "NeuralOperatorSolver",
    "NonlinearLeastSquaresProblem",
    "PINNConfig",
    "PINNResult",
    "PINNSolver",
    "Rodas4",
    "Ros34PW2",
    "UDEConfig",
    "UDESolver",
    "additive_hybrid",
    "create_ude",
    "heat_residual",
    "helmholtz_residual",
    "levenberg_marquardt",
    "poisson_residual",
    "solve_stiff_ode",
    "tsit5_solve",
]
