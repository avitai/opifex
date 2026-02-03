"""Adaptive method utilities for error estimation and mesh refinement.

This module provides utilities for:
1. Error estimation (residual-based, gradient-based, curvature-based)
2. Mesh refinement zone identification

All functions are JIT-compatible and designed for use with physics-informed
neural networks (PINNs) and adaptive mesh refinement (AMR) strategies.

All methods are based on well-established literature:
- Residual-based error: Direct PDE residual magnitude
- Gradient-based error: Gradient magnitude (Zienkiewicz-Zhu approach)
- Hessian-based error: Frobenius norm of Hessian (curvature indicator)
- Refinement zones: Threshold or percentile-based selection

Key Functions
-------------
- compute_residual_error: Residual-based error estimation
- compute_gradient_error: Gradient magnitude error indicators
- compute_hessian_error: Curvature-based error indicators
- identify_refinement_zones: Identify points needing refinement

Design Principles
-----------------
- All functions use AutoDiffEngine for differentiation (DRY principle)
- JIT-compatible (pure functions, no side effects)
- vmap-compatible for batch processing
- Type-annotated with jaxtyping for clarity

Example:
--------
>>> import jax.numpy as jnp
>>> from opifex.core.physics.autodiff_engine import AutoDiffEngine
>>> from opifex.core.physics.adaptive_methods import (
...     compute_gradient_error,
...     identify_refinement_zones,
... )
>>>
>>> # Define test function
>>> def u(x): return jnp.sum(x**2, axis=-1)
>>> x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
>>>
>>> # Compute gradient-based error
>>> error = compute_gradient_error(u, x, AutoDiffEngine)
>>>
>>> # Identify refinement zones
>>> needs_refinement = identify_refinement_zones(error, threshold=0.5)
"""
# ruff: noqa: F821
# F821 disabled: Ruff incorrectly flags jaxtyping symbolic dimensions ("batch", "dim")
# as undefined names. These are valid jaxtyping string literal dimension annotations.

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float


__all__ = [
    "compute_gradient_error",
    "compute_hessian_error",
    "compute_residual_error",
    "identify_refinement_zones",
]


# =============================================================================
# Error Estimation Functions
# =============================================================================


def compute_residual_error(
    model: Callable[[Float[Array, "... dim"]], Float[Array, "..."]],
    x: Float[Array, "batch dim"],
    residual: Float[Array, "batch"],
) -> Float[Array, "batch"]:
    """Compute residual-based error estimation.

    For a PDE residual R(u), the error is estimated as ||R(u)||. This is the
    simplest error indicator, directly measuring how well the solution satisfies
    the PDE.

    Mathematical Background
    -----------------------
    For a PDE operator L and solution u, the residual is:
        R(u) = L(u) - f

    The error estimate is:
        error = |R(u)|

    A perfect solution has R(u) = 0 everywhere.

    Parameters
    ----------
    model : Callable
        Neural network model u(x). Not used directly, but maintained for
        API consistency with other error estimation functions.
    x : Float[Array, "batch dim"]
        Spatial points where error is estimated.
    residual : Float[Array, "batch"]
        Pre-computed PDE residual at points x.

    Returns
    -------
    Float[Array, "batch"]
        Error estimate at each point. Equal to |residual|.

    Notes
    -----
    - Error scales linearly with residual magnitude
    - Zero residual implies zero error (perfect solution)
    - JIT-compatible and vmap-compatible

    Examples
    --------
    >>> residual = jnp.array([0.1, 0.5, 0.01, 0.3])
    >>> x = jnp.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [0.25, 0.75]])
    >>> error = compute_residual_error(lambda x: x, x, residual)
    >>> assert jnp.allclose(error, jnp.abs(residual))
    """
    # Error is simply the absolute value of the residual
    # This measures how far the solution is from satisfying the PDE
    return jnp.abs(residual)


def compute_gradient_error(
    model: Callable[[Float[Array, "... dim"]], Float[Array, "..."]],
    x: Float[Array, "batch dim"],
    autodiff_engine: Any,
) -> Float[Array, "batch"]:
    """Compute gradient-based error estimation.

    Uses the L2 norm of the gradient ||∇u|| as an error indicator. High gradient
    magnitude indicates rapid variation, suggesting the solution may need higher
    resolution in those regions.

    Mathematical Background
    -----------------------
    For solution u(x), the gradient-based error is:
        error = ||∇u|| = sqrt(Σᵢ (∂u/∂xᵢ)²)

    This is useful for:
    - Detecting sharp features (shocks, boundary layers)
    - Identifying regions needing refinement
    - Monitoring solution smoothness

    Parameters
    ----------
    model : Callable
        Neural network model u(x) mapping spatial coordinates to scalar output.
    x : Float[Array, "batch dim"]
        Spatial points where error is estimated. Shape (batch, spatial_dim).
    autodiff_engine : Any
        AutoDiffEngine class providing compute_gradient method.

    Returns
    -------
    Float[Array, "batch"]
        Gradient magnitude ||∇u|| at each point.

    Notes
    -----
    - Constant functions have zero gradient error
    - Linear functions have constant gradient error
    - JIT-compatible (uses AutoDiffEngine exclusively)
    - Works for any spatial dimension

    Examples
    --------
    >>> from opifex.core.physics.autodiff_engine import AutoDiffEngine
    >>> def u(x): return 2.0 * x[..., 0] + 3.0 * x[..., 1]  # Linear
    >>> x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    >>> error = compute_gradient_error(u, x, AutoDiffEngine)
    >>> expected = jnp.sqrt(2.0**2 + 3.0**2)  # √13 ≈ 3.606
    >>> assert jnp.allclose(error, expected)
    """
    # Compute gradient using AutoDiffEngine
    grad_u = autodiff_engine.compute_gradient(model, x)

    # Compute L2 norm of gradient at each point
    # grad_u shape: (batch, spatial_dim)
    # Result shape: (batch,)
    return jnp.linalg.norm(grad_u, axis=-1)


def compute_hessian_error(
    model: Callable[[Float[Array, "... dim"]], Float[Array, "..."]],
    x: Float[Array, "batch dim"],
    autodiff_engine: Any,
) -> Float[Array, "batch"]:
    """Compute curvature-based error estimation using Hessian matrix.

    Uses the Frobenius norm of the Hessian matrix ||H||_F as an error indicator.
    High curvature indicates regions where the solution changes rapidly, requiring
    finer resolution.

    Mathematical Background
    -----------------------
    For solution u(x), the Hessian matrix is:
        H_ij = ∂²u/∂xᵢ∂xⱼ

    The Frobenius norm is:
        ||H||_F = sqrt(Σᵢⱼ H_ij²)

    This measures total curvature and is useful for:
    - Detecting highly curved regions
    - Identifying second-order features
    - Refining near inflection points

    Parameters
    ----------
    model : Callable
        Neural network model u(x) mapping spatial coordinates to scalar output.
    x : Float[Array, "batch dim"]
        Spatial points where error is estimated. Shape (batch, spatial_dim).
    autodiff_engine : Any
        AutoDiffEngine class providing compute_hessian method.

    Returns
    -------
    Float[Array, "batch"]
        Frobenius norm of Hessian ||H||_F at each point.

    Notes
    -----
    - Linear functions have zero Hessian error
    - Quadratic functions have constant Hessian error
    - JIT-compatible (uses AutoDiffEngine exclusively)
    - More expensive than gradient error (second derivatives)

    Examples
    --------
    >>> from opifex.core.physics.autodiff_engine import AutoDiffEngine
    >>> def u(x): return jnp.sum(x**2, axis=-1)  # u = x² + y²
    >>> x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    >>> error = compute_hessian_error(u, x, AutoDiffEngine)
    >>> # Hessian of u = x² + y² is [[2, 0], [0, 2]]
    >>> # Frobenius norm: sqrt(2² + 2²) = sqrt(8) ≈ 2.828
    >>> assert jnp.allclose(error, jnp.sqrt(8.0), atol=1e-4)
    """
    # Compute Hessian using AutoDiffEngine
    hessian = autodiff_engine.compute_hessian(model, x)

    # Compute Frobenius norm of Hessian at each point
    # hessian shape: (batch, spatial_dim, spatial_dim)
    # Frobenius norm: sqrt(sum of squared elements)
    # Result shape: (batch,)

    # Reshape to (batch, spatial_dim * spatial_dim) and compute L2 norm
    batch_size = hessian.shape[0]
    hessian_flat = hessian.reshape(batch_size, -1)
    return jnp.linalg.norm(hessian_flat, axis=-1)


# =============================================================================
# Mesh Refinement Zone Identification
# =============================================================================


def identify_refinement_zones(
    error_indicator: Float[Array, "batch"],
    threshold: float = 0.1,
    percentile: float | None = None,
) -> Float[Array, "batch"]:
    """Identify points needing mesh refinement based on error indicators.

    Creates a boolean mask indicating which points should be refined based on
    either a fixed threshold or a percentile-based selection.

    Refinement Strategy
    -------------------
    - **Fixed Threshold**: Refine points where error > threshold
    - **Percentile-Based**: Refine top (100 - percentile)% of points

    Percentile-based selection is more adaptive and ensures a consistent
    fraction of points are refined regardless of absolute error magnitude.

    Parameters
    ----------
    error_indicator : Float[Array, "batch"]
        Error estimate at each point (from any error estimation function).
    threshold : float, default=0.1
        Fixed threshold for refinement. Points with error > threshold are refined.
        Ignored if percentile is provided.
    percentile : float | None, default=None
        If provided, refine points with error above this percentile.
        For example, percentile=75 means refine top 25% of points.

    Returns
    -------
    Float[Array, "batch"]
        Boolean mask indicating points needing refinement (True = refine).

    Notes
    -----
    - If percentile is provided, threshold is ignored
    - Percentile must be in [0, 100] if provided
    - Returns all False if no points exceed threshold
    - JIT-compatible

    Examples
    --------
    >>> # Fixed threshold
    >>> error = jnp.array([0.05, 0.15, 0.03, 0.25])
    >>> needs_refinement = identify_refinement_zones(error, threshold=0.1)
    >>> assert jnp.array_equal(needs_refinement, [False, True, False, True])
    >>>
    >>> # Percentile-based
    >>> error = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    >>> needs_refinement = identify_refinement_zones(
    ...     error, percentile=75.0
    ... )
    >>> assert jnp.sum(needs_refinement) == 2  # Top 25% = 2 points
    """
    if percentile is not None:
        # Percentile-based refinement
        # Compute threshold as the percentile value
        threshold_value = jnp.percentile(error_indicator, percentile)
        return error_indicator > threshold_value
    # Fixed threshold refinement
    return error_indicator > threshold
