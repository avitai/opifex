"""Local conservation-law residual losses for physics neural operators.

This module provides the shared, array-based flux-divergence loss used by the
physics-informed operator and the physics cross-attention mechanism. It is the
single source of truth for the *local* conservation law in flux form

    ∂_t q + ∇·F = 0,

where ``q`` is a conserved density (mass ``ρ``, momentum components ``ρu``, or
energy ``E``) and ``F`` is the corresponding flux. For a field sampled on a
uniform spatial grid the law is enforced by penalising the squared divergence of
the predicted flux, optionally including a discrete time derivative.

The continuity / mass, momentum and energy conservation laws in this flux form
are standard (see e.g. Landau & Lifshitz, *Fluid Mechanics*, §1-§6):

    mass:     ∂_t ρ      + ∇·(ρu)                 = 0
    momentum: ∂_t (ρu)   + ∇·(ρu⊗u + pI − τ)      = 0
    energy:   ∂_t E      + ∇·((E + p)u − k∇T − …)  = 0

A correct loss penalises the divergence of the conserved flux (the discrete
cell balance / net boundary flux), *not* the standard deviation of a sum.

The spatial divergence is computed with second-order central finite differences
and periodic wrapping, matching the ``FiniteDiff.divergence`` reference in
``neuraloperator`` (``neuralop/losses/differentiation.py``) and the grid
operators in :mod:`opifex.fields.operations`.

All functions are pure and compatible with ``jax.jit``, ``jax.grad`` and
``jax.vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def central_difference(values: jax.Array, axis: int, spacing: float | jax.Array) -> jax.Array:
    """Second-order central finite difference along a periodic axis.

    Computes ``(f_{i+1} - f_{i-1}) / (2 * spacing)`` with periodic wrapping at
    the boundaries via :func:`jax.numpy.roll`, matching the periodic branch of
    ``neuraloperator``'s ``FiniteDiff``.

    Args:
        values: Field samples on a uniform grid.
        axis: Axis along which to differentiate.
        spacing: Grid spacing ``Δx`` along ``axis`` (must be positive).

    Returns:
        Derivative ``∂f/∂x`` with the same shape as ``values``.
    """
    forward = jnp.roll(values, shift=-1, axis=axis)
    backward = jnp.roll(values, shift=1, axis=axis)
    return (forward - backward) / (2.0 * spacing)


def flux_divergence(flux: jax.Array, *, spatial_axis: int, spacing: float | jax.Array) -> jax.Array:
    """Discrete divergence ``∇·F`` of a vector flux along one spatial axis.

    The flux components are taken from the trailing channel axis and the
    divergence is accumulated as ``Σ_d ∂F_d/∂x_d``. Because the operator fields
    are sampled along a single flattened spatial axis, every flux component is
    differentiated with respect to that axis, giving the net cell balance
    ``Σ_d ∂F_d/∂x``.

    Args:
        flux: Flux field of shape ``(..., n_points, n_components)``.
        spatial_axis: Axis indexing the spatial grid points.
        spacing: Grid spacing along ``spatial_axis``.

    Returns:
        Scalar divergence field of shape ``(..., n_points)``.
    """
    derivative = central_difference(flux, axis=spatial_axis, spacing=spacing)
    return jnp.sum(derivative, axis=-1)


def conservation_residual_loss(
    flux: jax.Array,
    *,
    spatial_axis: int,
    spacing: float | jax.Array = 1.0,
    time_derivative: jax.Array | None = None,
) -> jax.Array:
    """Mean-squared local conservation-law residual ``mean((∂_t q + ∇·F)²)``.

    Enforces the flux-form conservation law ``∂_t q + ∇·F = 0`` by penalising
    the squared residual. With ``time_derivative=None`` this reduces to the
    steady-state / divergence-free residual ``mean((∇·F)²)``: a constant
    (divergence-free) flux yields ~0 loss while a flux with non-zero divergence
    yields the corresponding positive value.

    Args:
        flux: Predicted flux ``F`` of shape ``(..., n_points, n_components)``.
        spatial_axis: Axis indexing the spatial grid points.
        spacing: Uniform grid spacing ``Δx`` along ``spatial_axis``.
        time_derivative: Optional density time derivative ``∂_t q`` broadcastable
            to the divergence shape ``(..., n_points)``.

    Returns:
        Scalar mean-squared conservation residual.
    """
    divergence = flux_divergence(flux, spatial_axis=spatial_axis, spacing=spacing)
    residual = divergence if time_derivative is None else divergence + time_derivative
    return jnp.mean(residual**2)


__all__ = [
    "central_difference",
    "conservation_residual_loss",
    "flux_divergence",
]
