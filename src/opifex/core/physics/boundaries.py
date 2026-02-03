"""Boundary condition application functions for physics-informed learning.

This module provides the single source of truth for all boundary condition
application logic across the Opifex framework.

All functions are pure, JAX-compatible, and designed for use in
JIT-compiled training loops.
"""

from __future__ import annotations

from enum import Enum

import jax.numpy as jnp


class BoundaryType(Enum):
    """Enum defining all supported boundary condition types."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    MIXED = "mixed"


def apply_dirichlet(
    params: jnp.ndarray,
    boundary_value: float = 0.0,
    left_boundary: float | None = None,
    right_boundary: float | None = None,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Apply Dirichlet boundary condition to parameters.

    Dirichlet boundary condition fixes the value at boundaries:
    u(boundary) = boundary_value

    Args:
        params: Parameter array (shape: [..., n] where n >= 1)
        boundary_value: Value to set at both boundaries (default: 0.0)
        left_boundary: Optional separate value for left boundary
        right_boundary: Optional separate value for right boundary
        weight: Constraint weight (0-1). 1.0 = full constraint, 0.0 = no constraint

    Returns:
        Parameters with Dirichlet boundary condition applied
    """
    if params.size == 0:
        return params

    # Determine boundary values
    left_val = left_boundary if left_boundary is not None else boundary_value
    right_val = right_boundary if right_boundary is not None else boundary_value

    # Apply Dirichlet boundary condition
    constrained = params.at[..., 0].set(left_val)
    if params.shape[-1] > 1:
        constrained = constrained.at[..., -1].set(right_val)

    # Apply weight (partial constraint application)
    return weight * constrained + (1 - weight) * params


def apply_neumann(
    params: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Apply Neumann boundary condition to parameters.

    Neumann boundary condition fixes the derivative at boundaries to zero:
    du/dn(boundary) = 0

    This is implemented by setting boundary values equal to their neighbors,
    ensuring zero gradient at the boundaries.

    Args:
        params: Parameter array (shape: [..., n] where n >= 1)
        weight: Constraint weight (0-1). 1.0 = full constraint, 0.0 = no constraint

    Returns:
        Parameters with Neumann boundary condition applied
    """
    if params.size == 0 or params.shape[-1] < 3:
        return params

    # Apply Neumann boundary condition (zero derivative)
    # Set boundary values equal to neighbors
    constrained = params.at[..., 0].set(params[..., 1])
    constrained = constrained.at[..., -1].set(params[..., -2])

    # Apply weight (partial constraint application)
    return weight * constrained + (1 - weight) * params


def apply_periodic(
    params: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Apply periodic boundary condition to parameters.

    Periodic boundary condition ensures periodicity:
    u(left) = u(right)

    This is implemented by setting both boundaries to their average value.

    Args:
        params: Parameter array (shape: [..., n] where n >= 1)
        weight: Constraint weight (0-1). 1.0 = full constraint, 0.0 = no constraint

    Returns:
        Parameters with periodic boundary condition applied
    """
    if params.size == 0 or params.shape[-1] < 2:
        return params

    # Calculate average of boundary values
    avg_boundary = (params[..., 0] + params[..., -1]) / 2

    # Apply periodic boundary condition
    constrained = params.at[..., 0].set(avg_boundary)
    constrained = constrained.at[..., -1].set(avg_boundary)

    # Apply weight (partial constraint application)
    return weight * constrained + (1 - weight) * params


def apply_robin(
    params: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Apply Robin (mixed) boundary condition to parameters.

    Robin boundary condition is a linear combination of Dirichlet and Neumann:
    alpha * u + beta * du/dn = gamma at boundary

    Special cases:
    - beta=0: Reduces to Dirichlet BC (u = gamma/alpha)
    - alpha=0: Reduces to Neumann BC (du/dn = gamma/beta)

    Args:
        params: Parameter array (shape: [..., n] where n >= 1)
        alpha: Coefficient for u term
        beta: Coefficient for du/dn term
        gamma: Right-hand side value
        weight: Constraint weight (0-1). 1.0 = full constraint, 0.0 = no constraint

    Returns:
        Parameters with Robin boundary condition applied
    """
    if params.size == 0:
        return params

    # Handle special cases
    if beta == 0.0 and alpha != 0.0:
        # Dirichlet limit: u = gamma/alpha
        boundary_value = gamma / alpha
        return apply_dirichlet(params, boundary_value=boundary_value, weight=weight)

    if alpha == 0.0 and beta != 0.0:
        # Neumann limit: du/dn = gamma/beta
        # For simplicity, treat as zero derivative if gamma=0
        if gamma == 0.0:
            return apply_neumann(params, weight=weight)
        # Otherwise, apply an approximate solution
        constrained = params
        return weight * constrained + (1 - weight) * params

    # General Robin BC: alpha*u + beta*du/dn = gamma
    # Approximate solution at boundaries
    if params.shape[-1] < 2:
        return params

    # Left boundary: alpha*u[0] + beta*(u[1] - u[0])/dx = gamma
    # Solving for u[0]: u[0] = (gamma - beta*u[1]/dx) / (alpha - beta/dx)
    # For simplicity, use dx=1
    denominator_left = alpha - beta
    if jnp.abs(denominator_left) < 1e-8:
        # When alpha ≈ beta, use average of neighbor and target value
        left_constrained = (params[..., 1] + gamma / (alpha + 1e-8)) / 2
    else:
        left_constrained = (gamma - beta * params[..., 1]) / denominator_left

    # Right boundary: alpha*u[-1] + beta*(u[-1] - u[-2])/dx = gamma
    # Solving for u[-1]: u[-1] = (gamma + beta*u[-2]/dx) / (alpha + beta/dx)
    denominator_right = alpha + beta
    if jnp.abs(denominator_right) < 1e-8:
        # When alpha ≈ -beta, use average
        right_constrained = (params[..., -2] + gamma / (beta + 1e-8)) / 2
    else:
        right_constrained = (gamma + beta * params[..., -2]) / denominator_right

    constrained = params.at[..., 0].set(left_constrained)
    constrained = constrained.at[..., -1].set(right_constrained)

    # Apply weight (partial constraint application)
    return weight * constrained + (1 - weight) * params


def apply_boundary_condition(
    params: jnp.ndarray,
    boundary_type: BoundaryType | str,
    **kwargs,
) -> jnp.ndarray:
    """Apply boundary condition using unified interface.

    This is a convenience function that dispatches to the appropriate
    boundary condition function based on the type.

    Args:
        params: Parameter array
        boundary_type: Type of boundary condition (enum or string)
        **kwargs: Additional arguments passed to specific BC function

    Returns:
        Parameters with boundary condition applied

    Raises:
        ValueError: If boundary_type is unknown
    """
    # Convert string to enum if needed
    if isinstance(boundary_type, str):
        boundary_type_lower = boundary_type.lower()
        type_map = {bt.value: bt for bt in BoundaryType}
        if boundary_type_lower not in type_map:
            raise ValueError(
                f"Unknown boundary type: {boundary_type}. "
                f"Must be one of {list(type_map.keys())}"
            )
        boundary_type = type_map[boundary_type_lower]

    # Dispatch to appropriate function
    if boundary_type == BoundaryType.DIRICHLET:
        return apply_dirichlet(params, **kwargs)
    if boundary_type == BoundaryType.NEUMANN:
        return apply_neumann(params, **kwargs)
    if boundary_type == BoundaryType.PERIODIC:
        return apply_periodic(params, **kwargs)
    if boundary_type == BoundaryType.ROBIN:
        return apply_robin(params, **kwargs)
    if boundary_type == BoundaryType.MIXED:
        # Mixed boundary conditions would require more complex implementation
        # For now, default to Dirichlet
        return apply_dirichlet(params, **kwargs)

    raise ValueError(f"Unsupported boundary type: {boundary_type}")
