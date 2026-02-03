"""Configuration classes for second-order optimization methods.

This module provides unified configuration dataclasses for all second-order
optimization methods supported by the Opifex framework.

Design Principles:
    - All configs are frozen dataclasses (immutable)
    - Validation happens at construction time via __post_init__
    - Sensible defaults based on literature recommendations
    - Clear separation between method-specific and shared configs

References:
    - Survey: arXiv:2601.10222v1 Section 7
    - L-BFGS memory size: typically 3-20 (Liu & Nocedal, 1989)
    - Hybrid switching: Section 7.4 of the survey
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SecondOrderMethod(Enum):
    """Available second-order optimization methods."""

    LBFGS = "lbfgs"
    GAUSS_NEWTON = "gauss_newton"
    LEVENBERG_MARQUARDT = "levenberg_marquardt"
    BFGS = "bfgs"


class LinesearchType(Enum):
    """Line search algorithms for L-BFGS."""

    ZOOM = "zoom"
    BACKTRACKING = "backtracking"


class SwitchCriterion(Enum):
    """Criterion for switching from first-order to second-order optimization."""

    EPOCH = "epoch"
    LOSS_VARIANCE = "loss_variance"
    GRADIENT_NORM = "gradient_norm"
    RELATIVE_IMPROVEMENT = "relative_improvement"


@dataclass(frozen=True)
class SecondOrderConfig:
    """Unified configuration for second-order optimization.

    This is the base configuration class that provides common parameters
    shared across all second-order methods.

    Attributes:
        method: The second-order optimization method to use
        max_iterations: Maximum number of optimization iterations
        tolerance: Convergence tolerance for the optimizer
    """

    method: SecondOrderMethod = SecondOrderMethod.LBFGS
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(frozen=True)
class LBFGSConfig:
    """Configuration for L-BFGS optimizer.

    L-BFGS (Limited-memory BFGS) approximates the inverse Hessian using
    a limited history of gradient differences. This makes it suitable for
    large-scale optimization where storing the full Hessian is infeasible.

    Attributes:
        memory_size: Number of gradient pairs to store (typically 3-20)
        scale_init_precond: Whether to scale initial preconditioner
        linesearch: Line search algorithm to use
        max_linesearch_steps: Maximum steps for line search
        max_iterations: Maximum L-BFGS iterations
        tolerance: Convergence tolerance

    References:
        - Liu & Nocedal (1989): On the limited memory BFGS method
        - optax.lbfgs documentation
    """

    memory_size: int = 10
    scale_init_precond: bool = True
    linesearch: LinesearchType = LinesearchType.ZOOM
    max_linesearch_steps: int = 20
    max_iterations: int = 100
    tolerance: float = 1e-6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if self.max_linesearch_steps <= 0:
            raise ValueError("max_linesearch_steps must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class GaussNewtonConfig:
    """Configuration for Gauss-Newton and Levenberg-Marquardt solvers.

    Gauss-Newton is effective for nonlinear least-squares problems where
    the residual Jacobian can be computed efficiently. Levenberg-Marquardt
    adds damping for improved robustness.

    Attributes:
        damping_factor: Initial damping factor (λ) for LM
        damping_increase_factor: Factor to increase damping on failure (> 1)
        damping_decrease_factor: Factor to decrease damping on success (< 1)
        min_damping: Minimum allowed damping value
        max_damping: Maximum allowed damping value
        max_iterations: Maximum solver iterations
        rtol: Relative tolerance for convergence
        atol: Absolute tolerance for convergence

    References:
        - optimistix.LevenbergMarquardt documentation
        - Survey Section 7.3
    """

    damping_factor: float = 1e-3
    damping_increase_factor: float = 10.0
    damping_decrease_factor: float = 0.1
    min_damping: float = 1e-10
    max_damping: float = 1e10
    max_iterations: int = 100
    rtol: float = 1e-6
    atol: float = 1e-6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.damping_increase_factor <= 1:
            raise ValueError("increase_factor must be > 1")
        if self.damping_decrease_factor >= 1:
            raise ValueError("decrease_factor must be < 1")
        if self.min_damping >= self.max_damping:
            raise ValueError("min_damping must be < max_damping")


@dataclass(frozen=True)
class HybridOptimizerConfig:
    """Configuration for hybrid Adam→L-BFGS optimizer.

    This optimizer starts with Adam for initial exploration and switches
    to L-BFGS for efficient convergence once the loss landscape becomes
    smooth. This follows recommendations from Survey Section 7.4.

    The switch can be triggered by various criteria:
        - EPOCH: Switch after fixed number of steps
        - LOSS_VARIANCE: Switch when loss variance drops below threshold
        - GRADIENT_NORM: Switch when gradient norm drops below threshold
        - RELATIVE_IMPROVEMENT: Switch when relative improvement slows

    Attributes:
        first_order_steps: Steps to run Adam before considering switch
        switch_criterion: Criterion for switching to L-BFGS
        loss_variance_threshold: Threshold for loss variance criterion
        loss_history_window: Window size for computing loss statistics
        gradient_norm_threshold: Threshold for gradient norm criterion
        relative_improvement_threshold: Threshold for relative improvement
        adam_learning_rate: Learning rate for Adam phase
        adam_b1: Adam beta1 parameter
        adam_b2: Adam beta2 parameter
        lbfgs_config: Configuration for L-BFGS phase

    References:
        - Survey Section 7.4: "L-BFGS is more effective in later stages
          when loss varies smoothly"
    """

    first_order_steps: int = 1000
    switch_criterion: SwitchCriterion = SwitchCriterion.LOSS_VARIANCE
    loss_variance_threshold: float = 1e-4
    loss_history_window: int = 50
    gradient_norm_threshold: float = 1e-3
    relative_improvement_threshold: float = 1e-4
    adam_learning_rate: float = 1e-3
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    lbfgs_config: LBFGSConfig = field(default_factory=LBFGSConfig)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.first_order_steps <= 0:
            raise ValueError("first_order_steps must be positive")
        if self.loss_history_window <= 0:
            raise ValueError("loss_history_window must be positive")
        if self.adam_learning_rate <= 0:
            raise ValueError("adam_learning_rate must be positive")
