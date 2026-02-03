"""Second-order optimization infrastructure for Opifex.

This module provides wrappers and utilities for second-order optimization
methods including L-BFGS, Gauss-Newton, Levenberg-Marquardt, and hybrid
Adam→L-BFGS strategies.

The design philosophy is to wrap existing robust implementations (optax, optimistix)
rather than reimplementing from scratch. We only implement novel functionality
like the hybrid optimizer that doesn't exist in external libraries.

Key Components:
    - SecondOrderConfig: Unified configuration for all second-order methods
    - LBFGSConfig: L-BFGS specific configuration
    - GaussNewtonConfig: Gauss-Newton/Levenberg-Marquardt configuration
    - HybridOptimizerConfig: Hybrid Adam→L-BFGS configuration
    - create_lbfgs_optimizer: Create optax L-BFGS optimizer
    - create_gauss_newton_solver: Create optimistix Gauss-Newton solver
    - HybridOptimizer: Adam→L-BFGS switching optimizer

References:
    - Survey: arXiv:2601.10222v1 Section 7 (Second-Order Methods)
    - optax L-BFGS: https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lbfgs
    - optimistix: https://docs.kidger.site/optimistix/
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
    create_bfgs_solver,
    create_gauss_newton_solver,
    create_lbfgs_optimizer,
    create_levenberg_marquardt_solver,
)


__all__ = [
    # Config
    "GaussNewtonConfig",
    # Hybrid optimizer
    "HybridOptimizer",
    "HybridOptimizerConfig",
    "HybridOptimizerState",
    "LBFGSConfig",
    "LinesearchType",
    # NNX integration
    "NNXHybridOptimizer",
    "NNXSecondOrderOptimizer",
    "SecondOrderConfig",
    "SecondOrderMethod",
    "SwitchCriterion",
    # Wrappers
    "create_bfgs_solver",
    "create_gauss_newton_solver",
    "create_lbfgs_optimizer",
    "create_levenberg_marquardt_solver",
    "create_nnx_lbfgs_optimizer",
]
