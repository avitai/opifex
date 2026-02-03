"""Wrappers for external second-order optimization libraries.

This module provides thin wrappers around optax and optimistix to create
second-order optimizers with our unified configuration interface.

Design Philosophy:
    - Wrap existing robust implementations (optax, optimistix)
    - Don't reinvent the wheel
    - Provide consistent interface through our config classes

References:
    - optax.lbfgs: Pure JAX L-BFGS with line search
    - optimistix: Gauss-Newton, Levenberg-Marquardt, BFGS
"""

from __future__ import annotations

import optax
import optimistix as optx

from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    LBFGSConfig,
    LinesearchType,
)


def create_lbfgs_optimizer(
    config: LBFGSConfig | None = None,
) -> optax.GradientTransformation:
    """Create L-BFGS optimizer using optax.

    L-BFGS is a quasi-Newton method that approximates the inverse Hessian
    using a limited history of gradient differences. This is the recommended
    second-order optimizer for large-scale optimization.

    Args:
        config: L-BFGS configuration. Uses defaults if None.

    Returns:
        Optax L-BFGS gradient transformation.

    Example:
        >>> config = LBFGSConfig(memory_size=20)
        >>> optimizer = create_lbfgs_optimizer(config)
        >>> # Use with optax training loop
    """
    if config is None:
        config = LBFGSConfig()

    # Select line search algorithm
    if config.linesearch == LinesearchType.ZOOM:
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=config.max_linesearch_steps,
        )
    else:
        linesearch = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=config.max_linesearch_steps,
        )

    return optax.lbfgs(
        memory_size=config.memory_size,
        scale_init_precond=config.scale_init_precond,
        linesearch=linesearch,
    )


def create_gauss_newton_solver(
    config: GaussNewtonConfig | None = None,
) -> optx.AbstractLeastSquaresSolver:
    """Create Gauss-Newton solver using optimistix.

    Gauss-Newton is effective for nonlinear least-squares problems.
    Note that this creates a solver for root-finding/minimization,
    not a gradient transformation like L-BFGS.

    Args:
        config: Gauss-Newton configuration. Uses defaults if None.

    Returns:
        Optimistix Gauss-Newton solver.

    Example:
        >>> solver = create_gauss_newton_solver()
        >>> # Use with optimistix.least_squares
    """
    if config is None:
        config = GaussNewtonConfig()

    return optx.GaussNewton(
        rtol=config.rtol,
        atol=config.atol,
    )


def create_levenberg_marquardt_solver(
    config: GaussNewtonConfig | None = None,
) -> optx.AbstractLeastSquaresSolver:
    """Create Levenberg-Marquardt solver using optimistix.

    Levenberg-Marquardt adds damping to Gauss-Newton for improved
    robustness, especially when far from the optimum. This is the
    recommended solver for ill-conditioned least-squares problems.

    Args:
        config: Gauss-Newton/LM configuration. Uses defaults if None.

    Returns:
        Optimistix Levenberg-Marquardt solver.

    Example:
        >>> solver = create_levenberg_marquardt_solver()
        >>> # Use with optimistix.least_squares
    """
    if config is None:
        config = GaussNewtonConfig()

    return optx.LevenbergMarquardt(
        rtol=config.rtol,
        atol=config.atol,
    )


def create_bfgs_solver(
    config: GaussNewtonConfig | None = None,
) -> optx.AbstractMinimiser:
    """Create BFGS solver using optimistix.

    Full-memory BFGS for smaller-scale problems where storing the
    full inverse Hessian approximation is feasible.

    Args:
        config: Configuration with tolerance settings. Uses defaults if None.

    Returns:
        Optimistix BFGS minimizer.

    Example:
        >>> solver = create_bfgs_solver()
        >>> # Use with optimistix.minimise
    """
    if config is None:
        config = GaussNewtonConfig()

    return optx.BFGS(
        rtol=config.rtol,
        atol=config.atol,
    )
