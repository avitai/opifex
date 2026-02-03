"""
Clean JIT-compatible multi-scale PDE implementations.

All functions are:
- JIT-compatible (no try/except, no Python control flow)
- Simple (only use AutoDiffEngine methods)
- Literature-backed

References:
- Homogenization: Bensoussan, Lions, Papanicolaou (1978)
- Two-scale: Allaire (1992), asymptotic expansion theory
- AMR: Kelly error estimator, Zienkiewicz-Zhu gradient recovery
"""
# ruff: noqa: F821
# F821 disabled: Ruff incorrectly flags jaxtyping symbolic dimensions ("batch", "dim")
# as undefined names. These are valid jaxtyping string literal dimension annotations.

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float


def _homogenization_residual_clean(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    coefficient_fn: Callable | None = None,
    source_term: Float[Array, "batch"] | None = None,
) -> Float[Array, "batch"]:
    """
    Compute homogenization PDE residual: -∇·(a(x)∇u) = f.

    Mathematical formulation (homogenization theory):
        -∇·(a(x)∇u) = f

    Using product rule:
        ∇·(a∇u) = ∇a·∇u + a∇²u

    References:
        - Bensoussan, Lions, Papanicolaou (1978)
          "Asymptotic Analysis for Periodic Structures"
        - Allaire (1992) "Homogenization and Two-Scale Convergence"

    JIT-compatible: No try/except, no Python control flow.
    """
    # Compute gradient and Laplacian using AutoDiffEngine
    grad_u = autodiff_engine.compute_gradient(model, x)
    laplacian_u = autodiff_engine.compute_laplacian(model, x)

    # Default coefficient to 1.0
    coeff = (
        jnp.ones(x.shape[0])
        if coefficient_fn is None
        else jnp.squeeze(coefficient_fn(x))
    )

    # Compute gradient of coefficient
    # For now, assume constant coefficient (∇a = 0) for simplicity and JIT compatibility
    # TODO: Add proper varying coefficient support with literature-backed methods
    # See: Bank & Weiser (1985) "Some a posteriori error estimators"
    grad_coeff = jnp.zeros_like(grad_u)

    # Divergence: ∇·(a∇u) = ∇a·∇u + a∇²u
    div_term = jnp.sum(grad_coeff * grad_u, axis=-1) + coeff * laplacian_u

    # Handle source term - default to zero
    source = jnp.zeros(x.shape[0]) if source_term is None else source_term

    # Residual: -∇·(a∇u) - f
    return -div_term - source


def _two_scale_residual_clean(
    model_macro: Callable,
    model_micro: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    epsilon: float = 0.1,
    coupling_fn: Callable | None = None,
) -> tuple[Array, Array]:
    """
    Compute two-scale expansion PDE residuals.

    Mathematical formulation (two-scale asymptotic expansion):
        L₀(u₀) + ε L₁(u₀, u₁) + O(ε²) = f

    where:
        - u₀ is macroscale solution
        - u₁ is microscale correction
        - ε → 0 is scale separation parameter

    When ε=0, use safe division to avoid NaN.

    References:
        - Allaire (1992) "Homogenization and Two-Scale Convergence"
        - Bensoussan, Lions, Papanicolaou (1978)

    JIT-compatible: No try/except, no Python control flow,
                     uses jnp.where for safe division.
    """
    # Compute macroscale operators
    laplacian_macro = autodiff_engine.compute_laplacian(model_macro, x)
    grad_macro = autodiff_engine.compute_gradient(model_macro, x)

    # Compute microscale operators
    laplacian_micro = autodiff_engine.compute_laplacian(model_micro, x)
    grad_micro = autodiff_engine.compute_gradient(model_micro, x)

    # Macroscale residual: L₀(u₀) + ε * coupling
    macro_operator = -laplacian_macro

    # Coupling term
    has_custom_coupling = coupling_fn is not None
    coupling = (
        coupling_fn(grad_macro, grad_micro)
        if has_custom_coupling
        else epsilon * jnp.sum(grad_macro * grad_micro, axis=-1)
    )

    macro_residual = macro_operator + coupling

    # Microscale residual with safe division
    # When ε=0, use -∇²u instead of -1/ε² ∇²u to avoid division by zero
    # Use jnp.where for JIT compatibility
    epsilon_sq_safe = jnp.where(epsilon == 0.0, 1.0, epsilon**2)
    micro_scale_factor = jnp.where(epsilon == 0.0, 1.0, 1.0 / epsilon_sq_safe)
    micro_residual = -micro_scale_factor * laplacian_micro

    return (macro_residual, micro_residual)


def _amr_poisson_residual_clean(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    source_term: Float[Array, "batch"] | None = None,
    error_threshold: float = 0.1,
) -> tuple[Array, Array]:
    """
    Compute Poisson residual with AMR error indicators.

    Mathematical formulation:
        Residual: ∇²u - f = 0
        Error indicator: ||∇u|| + ||H||_F

    where ||H||_F is Frobenius norm of Hessian (curvature indicator).

    References:
        - Kelly et al. error estimator
        - Zienkiewicz-Zhu gradient recovery method
        - deal.II KellyErrorEstimator implementation

    JIT-compatible: No try/except, no Python control flow.
    """
    # Compute Poisson residual
    laplacian = autodiff_engine.compute_laplacian(model, x)
    source = jnp.zeros(x.shape[0]) if source_term is None else source_term
    residual = laplacian - source

    # Compute error indicator: ||∇u|| + ||H||_F
    grad = autodiff_engine.compute_gradient(model, x)
    grad_magnitude = jnp.linalg.norm(grad, axis=-1)

    hessian = autodiff_engine.compute_hessian(model, x)
    hessian_flat = hessian.reshape(hessian.shape[0], -1)
    hessian_norm = jnp.linalg.norm(hessian_flat, axis=-1)

    error_indicator = grad_magnitude + hessian_norm

    return (residual, error_indicator)


__all__ = [
    "_amr_poisson_residual_clean",
    "_homogenization_residual_clean",
    "_two_scale_residual_clean",
]
