"""Boundary condition application functions for physics-informed learning.

This module provides the single source of truth for all boundary condition
application logic across the Opifex framework.

All functions are pure, JAX-compatible, and designed for use in
JIT-compiled training loops.

Reference
---------
The per-boundary-type residual conventions follow DeepXDE's ``icbc`` module
(``deepxde/icbc/boundary_conditions.py``):

- ``DirichletBC``: residual ``u - g`` (value mismatch).
- ``NeumannBC``: residual ``du/dn - g`` (outward normal-derivative mismatch).
- ``RobinBC``: residual ``du/dn - func(x, u)``, i.e. the general linear form
  ``alpha*u + beta*du/dn - gamma``.
- Mixed boundaries dispatch each boundary segment by its declared type
  (see ``PointSetOperatorBC`` / per-segment ``error`` handling).

The outward-normal convention is: the left edge normal points in ``-x`` and the
right edge normal points in ``+x``.  These projection helpers set boundary
*values* so that the discrete one-sided derivative matches the prescribed
normal-derivative target; they never silently substitute Dirichlet for another
declared type.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp


class BoundaryType(Enum):
    """Enum defining all supported boundary condition types."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    MIXED = "mixed"


def _normalize_boundary_type(boundary_type: BoundaryType | str) -> BoundaryType:
    """Resolve a string or enum token to a :class:`BoundaryType`.

    Args:
        boundary_type: Boundary condition type as an enum member or its string
            value (case-insensitive).

    Returns:
        The corresponding :class:`BoundaryType` enum member.

    Raises:
        ValueError: If the token does not name a known boundary type.
    """
    if isinstance(boundary_type, BoundaryType):
        return boundary_type

    type_map = {bt.value: bt for bt in BoundaryType}
    key = boundary_type.lower()
    if key not in type_map:
        raise ValueError(
            f"Unknown boundary type: {boundary_type}. Must be one of {list(type_map.keys())}"
        )
    return type_map[key]


def apply_dirichlet(
    params: jnp.ndarray,
    boundary_value: float = 0.0,
    left_boundary: float | None = None,
    right_boundary: float | None = None,
    weight: jax.Array | float = 1.0,
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
    normal_derivative: jax.Array | float = 0.0,
    dx: float = 1.0,
    weight: jax.Array | float = 1.0,
) -> jnp.ndarray:
    """Apply Neumann boundary condition to parameters.

    Neumann boundary condition fixes the *outward normal derivative* at the
    boundaries to a prescribed value ``g`` (DeepXDE ``NeumannBC`` residual
    ``du/dn - g``)::

        du/dn(boundary) = normal_derivative

    Using the outward-normal convention (left normal -> -x, right normal -> +x)
    and a one-sided finite difference with spacing ``dx``:

    - Left edge:  ``du/dn|_left  = -(u[1] - u[0]) / dx == g``
      => ``u[0]  = u[1]  + g * dx``
    - Right edge: ``du/dn|_right =  (u[-1] - u[-2]) / dx == g``
      => ``u[-1] = u[-2] + g * dx``

    With ``g = 0`` this reduces to copying the neighbour value (zero gradient).

    Args:
        params: Parameter array (shape: [..., n] where n >= 3).
        normal_derivative: Prescribed outward normal derivative ``g`` at both
            boundaries (default: 0.0, i.e. zero-gradient Neumann).
        dx: Grid spacing used by the one-sided finite difference (default: 1.0).
        weight: Constraint weight (0-1). 1.0 = full constraint, 0.0 = no constraint.

    Returns:
        Parameters with the Neumann boundary condition applied.
    """
    if params.size == 0 or params.shape[-1] < 3:
        return params

    # Set boundary values so the one-sided outward normal derivative equals g.
    left_value = params[..., 1] + normal_derivative * dx
    right_value = params[..., -2] + normal_derivative * dx

    constrained = params.at[..., 0].set(left_value)
    constrained = constrained.at[..., -1].set(right_value)

    # Apply weight (partial constraint application)
    return weight * constrained + (1 - weight) * params


def apply_periodic(
    params: jnp.ndarray,
    weight: jax.Array | float = 1.0,
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
    weight: jax.Array | float = 1.0,
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
        # Neumann limit: du/dn = gamma / beta. Honour a non-zero prescribed
        # flux instead of silently returning the input unchanged.
        return apply_neumann(params, normal_derivative=gamma / beta, weight=weight)

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


# Per-boundary types that constrain a single edge. PERIODIC couples both edges
# and MIXED is itself a composite, so neither is a valid single-edge constraint.
_PER_BOUNDARY_TYPES: frozenset[BoundaryType] = frozenset(
    {BoundaryType.DIRICHLET, BoundaryType.NEUMANN, BoundaryType.ROBIN}
)


def _constrain_single_edge(
    params: jnp.ndarray,
    edge: int,
    boundary_type: BoundaryType,
    edge_kwargs: dict[str, Any],
    weight: jax.Array | float,
) -> jnp.ndarray:
    """Apply one boundary type to a single edge, preserving the opposite edge.

    The relevant per-type helper is applied to ``params`` (which constrains both
    ends), then the untouched edge is restored so only ``edge`` is modified.

    Args:
        params: Parameter array (shape: [..., n] where n >= 1).
        edge: ``0`` for the left edge or ``-1`` for the right edge.
        boundary_type: One of DIRICHLET, NEUMANN, or ROBIN.
        edge_kwargs: Keyword arguments for the per-type helper for this edge.
        weight: Constraint weight (0-1) applied to this edge.

    Returns:
        Parameters with only the requested edge constrained.

    Raises:
        ValueError: If ``boundary_type`` is not a valid per-boundary type.
    """
    if boundary_type not in _PER_BOUNDARY_TYPES:
        valid = sorted(bt.value for bt in _PER_BOUNDARY_TYPES)
        raise ValueError(
            f"{boundary_type.value!r} is not a valid per-boundary edge type. "
            f"Per-edge types must be one of {valid}."
        )

    if boundary_type == BoundaryType.DIRICHLET:
        constrained = apply_dirichlet(params, weight=weight, **edge_kwargs)
    elif boundary_type == BoundaryType.NEUMANN:
        constrained = apply_neumann(params, weight=weight, **edge_kwargs)
    else:  # BoundaryType.ROBIN
        constrained = apply_robin(params, weight=weight, **edge_kwargs)

    # Restore the opposite edge so only `edge` is altered.
    other = -1 if edge == 0 else 0
    return constrained.at[..., other].set(params[..., other])


def apply_mixed(
    params: jnp.ndarray,
    left_type: BoundaryType | str,
    right_type: BoundaryType | str,
    left_kwargs: dict[str, Any] | None = None,
    right_kwargs: dict[str, Any] | None = None,
    weight: jax.Array | float = 1.0,
) -> jnp.ndarray:
    """Apply mixed boundary conditions with per-edge type dispatch.

    Each edge is constrained by its own declared boundary type, mirroring
    DeepXDE's per-segment ``error`` dispatch where every boundary segment uses
    its declared BC residual.  Unknown types raise (fail-fast); a non-Dirichlet
    edge is never silently treated as Dirichlet.

    Args:
        params: Parameter array (shape: [..., n] where n >= 1).
        left_type: Boundary type for the left edge (enum or string). Must be one
            of DIRICHLET, NEUMANN, or ROBIN.
        right_type: Boundary type for the right edge (enum or string). Same
            constraint as ``left_type``.
        left_kwargs: Keyword arguments for the left-edge helper (e.g.
            ``{"boundary_value": 1.0}`` for Dirichlet, ``{"alpha":..., "beta":...,
            "gamma":...}`` for Robin). Defaults to no extra arguments.
        right_kwargs: Keyword arguments for the right-edge helper.
        weight: Constraint weight (0-1) applied to both edges.

    Returns:
        Parameters with each edge constrained by its declared type.

    Raises:
        ValueError: If either per-edge type is unknown or is not a valid
            single-edge constraint (MIXED/PERIODIC).
    """
    if params.size == 0:
        return params

    resolved_left = _normalize_boundary_type(left_type)
    resolved_right = _normalize_boundary_type(right_type)
    left_args = left_kwargs if left_kwargs is not None else {}
    right_args = right_kwargs if right_kwargs is not None else {}

    constrained = _constrain_single_edge(params, 0, resolved_left, left_args, weight)
    if params.shape[-1] > 1:
        constrained = _constrain_single_edge(constrained, -1, resolved_right, right_args, weight)
    return constrained


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

        For ``BoundaryType.MIXED`` the keyword arguments are forwarded to
        :func:`apply_mixed` and must include ``left_type`` and ``right_type``.

    Returns:
        Parameters with boundary condition applied

    Raises:
        ValueError: If ``boundary_type`` (or, for MIXED, a per-edge type) is
            unknown. Unknown types fail fast and are never treated as Dirichlet.
    """
    resolved = _normalize_boundary_type(boundary_type)

    # Dispatch to appropriate function. MIXED routes to genuine per-edge
    # handling; there is no silent Dirichlet fallback for any type.
    dispatch = {
        BoundaryType.DIRICHLET: apply_dirichlet,
        BoundaryType.NEUMANN: apply_neumann,
        BoundaryType.PERIODIC: apply_periodic,
        BoundaryType.ROBIN: apply_robin,
        BoundaryType.MIXED: apply_mixed,
    }
    return dispatch[resolved](params, **kwargs)
