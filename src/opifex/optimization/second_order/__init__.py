"""Second-order optimization infrastructure for Opifex.

Provides pure-JAX second-order optimization methods including Optax L-BFGS,
Gauss-Newton and Levenberg-Marquardt nonlinear least-squares solvers
(:mod:`opifex.solvers.nonlinear_least_squares`), full-memory :class:`BFGSSolver`,
and hybrid Adam→L-BFGS strategies.

References:
    - Survey: arXiv:2601.10222v1 Section 7 (Second-Order Methods)
    - optax L-BFGS: https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lbfgs
"""

from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    HybridOptimizerConfig,
    LBFGSConfig,
    LinesearchType,
    SecondOrderConfig,
    SecondOrderMethod,
    SwitchCriterion,
)
from opifex.optimization.second_order.hybrid_optimizer import (
    HybridOptimizer,
    HybridOptimizerState,
)
from opifex.optimization.second_order.nnx_integration import (
    create_nnx_lbfgs_optimizer,
    NNXHybridOptimizer,
    NNXSecondOrderOptimizer,
)
from opifex.optimization.second_order.wrappers import (
    BFGSSolver,
    create_bfgs_solver,
    create_gauss_newton_solver,
    create_lbfgs_optimizer,
    create_levenberg_marquardt_solver,
    OptimizationProblem,
)


__all__ = [
    "BFGSSolver",
    "GaussNewtonConfig",
    "HybridOptimizer",
    "HybridOptimizerConfig",
    "HybridOptimizerState",
    "LBFGSConfig",
    "LinesearchType",
    "NNXHybridOptimizer",
    "NNXSecondOrderOptimizer",
    "OptimizationProblem",
    "SecondOrderConfig",
    "SecondOrderMethod",
    "SwitchCriterion",
    "create_bfgs_solver",
    "create_gauss_newton_solver",
    "create_lbfgs_optimizer",
    "create_levenberg_marquardt_solver",
    "create_nnx_lbfgs_optimizer",
]
