"""Autodiff utilities for physics-informed computing following JAX/NNX best practices.

This module provides optimized utilities for computing spatial derivatives using JAX
autodiff. All functions are JIT-compiled and support both generic callables and
Flax NNX modules.

Key Design Principles:
1. Pure functional API with static_argnums for callables
2. Separate NNX-aware functions using split/merge pattern
3. All functions are batched via vmap
4. Full JIT compilation for performance
5. Type-safe with jaxtyping hints

Usage - Generic Callables:
    from opifex.core.physics.autodiff_engine import compute_gradient, compute_laplacian

    def my_solution(x):
        return x[..., 0]**2 + x[..., 1]**2

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    grad = compute_gradient(my_solution, x)
    laplacian = compute_laplacian(my_solution, x)

Usage - NNX Models:
    from opifex.core.physics.autodiff_engine import compute_gradient_nnx

    model = MyPINN(rngs=nnx.Rngs(0))
    x = jnp.array([[0.5, 0.5]])
    grad = compute_gradient_nnx(model, x)  # Gradient w.r.t. input!
"""
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# F821 disabled: Ruff incorrectly flags jaxtyping symbolic dimensions ("batch", "dim")
# as undefined names. These are valid jaxtyping string literal dimension annotations.
# Pyright also disabled for the same reason - these are symbolic jaxtyping dimensions.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float


@partial(jax.jit, static_argnums=(0,))
def compute_gradient(
    f: Callable[[Float[Array, "... dim"]], Float[Array, ...]],
    x: Float[Array, "batch dim"],
) -> Float[Array, "batch dim"]:
    """Compute gradient ∇f with respect to input x.

    Uses JAX autodiff to compute the gradient of a scalar function with respect
    to its input. Batched computation via vmap.

    Args:
        f: Scalar function f: R^n -> R. Must return scalar per batch item.
           Example: lambda x: x[..., 0]**2 + x[..., 1]**2
        x: Input points, shape (batch, dim)

    Returns:
        Gradient ∇f at each point, shape (batch, dim)

    Example:
        >>> def f(x):
        ...     return x[..., 0]**2 + x[..., 1]**2
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> grad = compute_gradient(f, x)
        >>> # Expected: [[2.0, 4.0], [6.0, 8.0]]

    Note:
        Function f is marked as static (static_argnums=0), so changing f
        triggers recompilation. For frequently changing functions, consider
        using compute_gradient_nnx for NNX modules.
    """

    def grad_single(x_single: Float[Array, dim]) -> Float[Array, dim]:
        """Compute gradient for a single point."""
        # Use .ravel()[0] to handle both (batch,) and (batch, 1) shapes
        return jax.grad(lambda x: f(x[None, ...]).ravel()[0])(x_single)

    return jax.vmap(grad_single)(x)


@partial(jax.jit, static_argnums=(0,))
def compute_laplacian(
    f: Callable[[Float[Array, "... dim"]], Float[Array, ...]],
    x: Float[Array, "batch dim"],
) -> Float[Array, batch]:
    """Compute Laplacian ∇²f = trace(Hessian(f)).

    The Laplacian is the sum of all second partial derivatives:
    ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²

    For complex-valued functions f = u + iv, computes ∇²f = ∇²u + i∇²v
    by computing Laplacians of real and imaginary parts separately.

    Args:
        f: Scalar function f: R^n -> R (or C for complex outputs)
        x: Input points, shape (batch, dim)

    Returns:
        Laplacian ∇²f at each point, shape (batch,)

    Example:
        >>> def f(x):
        ...     return x[..., 0]**2 + x[..., 1]**2
        >>> x = jnp.array([[1.0, 1.0], [0.5, 0.5]])
        >>> lap = compute_laplacian(f, x)
        >>> # Expected: [4.0, 4.0] (constant for quadratic)
    """

    def laplacian_single(x_single: Float[Array, dim]) -> Float[Array, ""]:
        """Compute Laplacian for a single point via Hessian trace.

        For real-valued functions, returns real dtype. For complex functions,
        computes Laplacians of real and imaginary parts separately.
        Uses .ravel()[0] to support both (batch,) and (batch, 1) output shapes.
        """

        def f_scalar(x):
            return f(x[None, ...]).ravel()[0]

        # Compute Laplacians of real and imaginary parts separately
        # This handles both real and complex functions without try/except
        hess_real = jax.hessian(lambda x: jnp.real(f_scalar(x)))(x_single)
        lap_real = jnp.trace(hess_real)

        hess_imag = jax.hessian(lambda x: jnp.imag(f_scalar(x)))(x_single)
        lap_imag = jnp.trace(hess_imag)

        # For real functions, lap_imag will be 0, so this returns effectively real
        # For complex functions, returns complex Laplacian
        return lap_real + 1j * lap_imag

    return jax.vmap(laplacian_single)(x)


@partial(jax.jit, static_argnums=(0,))
def compute_divergence(
    F: Callable[[Float[Array, "... dim"]], Float[Array, "... dim"]],
    x: Float[Array, "batch dim"],
) -> Float[Array, batch]:
    """Compute divergence ∇·F = Σᵢ ∂Fᵢ/∂xᵢ for vector field F.

    The divergence measures the "outflow" of a vector field at each point.

    Args:
        F: Vector field F: R^n -> R^n (returns vector of same dimension as input)
        x: Input points, shape (batch, dim)

    Returns:
        Divergence ∇·F at each point, shape (batch,)

    Example:
        >>> def F(x):
        ...     return x  # Radial field [x, y]
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> div = compute_divergence(F, x)
        >>> # Expected: [2.0, 2.0] (∂x/∂x + ∂y/∂y = 1 + 1)
    """

    def divergence_single(x_single: Float[Array, dim]) -> Float[Array, ""]:
        """Compute divergence for a single point via Jacobian trace."""
        jac = jax.jacobian(lambda x: F(x[None, ...])[0])(x_single)
        return jnp.trace(jac)  # Sum of diagonal elements

    return jax.vmap(divergence_single)(x)


@partial(jax.jit, static_argnums=(0,))
def compute_hessian(
    f: Callable[[Float[Array, "... dim"]], Float[Array, ...]],
    x: Float[Array, "batch dim"],
) -> Float[Array, "batch dim dim"]:
    """Compute Hessian matrix of second partial derivatives.

    The Hessian H[i,j] = ∂²f/∂xᵢ∂xⱼ captures the curvature of function f.

    Args:
        f: Scalar function f: R^n -> R
        x: Input points, shape (batch, dim)

    Returns:
        Hessian matrix ∇²f at each point, shape (batch, dim, dim)

    Example:
        >>> def f(x):
        ...     return x[..., 0]**2 + x[..., 1]**2
        >>> x = jnp.array([[1.0, 2.0]])
        >>> hess = compute_hessian(f, x)
        >>> # Expected: [[[2.0, 0.0], [0.0, 2.0]]] (diagonal for separable function)
    """

    def hessian_single(x_single: Float[Array, dim]) -> Float[Array, "dim dim"]:
        """Compute Hessian for a single point."""
        # Use .ravel()[0] to handle both (batch,) and (batch, 1) shapes
        return jax.hessian(lambda x: f(x[None, ...]).ravel()[0])(x_single)

    return jax.vmap(hessian_single)(x)


def compute_gradient_nnx(
    model: nnx.Module,
    x: Float[Array, "batch dim"],
    output_idx: int = 0,
) -> Float[Array, "batch dim"]:
    """Compute gradient of NNX model output with respect to input.

    Uses the Flax NNX functional API (split/merge pattern) to enable pure JAX
    transformations on mutable NNX modules.

    Args:
        model: Flax NNX module (e.g., neural network)
        x: Input points, shape (batch, dim)
        output_idx: Which output dimension to differentiate (for multi-output models)

    Returns:
        Gradient of model(x)[output_idx] w.r.t. input x, shape (batch, dim)

    Example:
        >>> class MyPINN(nnx.Module):
        ...     def __init__(self, rngs):
        ...         self.dense = nnx.Linear(2, 1, rngs=rngs)
        ...     def __call__(self, x):
        ...         return self.dense(x)
        >>>
        >>> model = MyPINN(rngs=nnx.Rngs(0))
        >>> x = jnp.array([[0.5, 0.5]])
        >>> grad = compute_gradient_nnx(model, x)

    Note:
        This function splits the model into (graphdef, state) for functional
        transformation. The graphdef is marked static for JIT compilation.
    """
    # Split model into immutable graphdef and mutable state
    graphdef, state = nnx.split(model)

    @partial(jax.jit, static_argnums=(0,))
    def _grad_fn(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        x_batch: Float[Array, "batch dim"],
    ) -> Float[Array, "batch dim"]:
        """Inner function with graphdef marked static."""
        # Reconstruct model inside JIT-compiled function
        model_reconstructed = nnx.merge(graphdef, state)

        def compute_single_grad(x_single: Float[Array, dim]) -> Float[Array, dim]:
            """Compute gradient for single input point."""

            def model_output_scalar(x_input: Float[Array, dim]) -> Float[Array, ""]:
                """Extract scalar output from model for gradient."""
                out = model_reconstructed(x_input[None, ...])  # Add batch dimension
                return out[0, output_idx]  # Extract scalar at output_idx

            return jax.grad(model_output_scalar)(x_single)

        return jax.vmap(compute_single_grad)(x_batch)

    return _grad_fn(graphdef, state, x)


def compute_laplacian_nnx(
    model: nnx.Module,
    x: Float[Array, "batch dim"],
    output_idx: int = 0,
) -> Float[Array, batch]:
    """Compute Laplacian of NNX model output with respect to input.

    Computes ∇²u where u = model(x)[output_idx].

    Args:
        model: Flax NNX module
        x: Input points, shape (batch, dim)
        output_idx: Which output dimension to use

    Returns:
        Laplacian ∇²u at each point, shape (batch,)

    Example:
        >>> model = MyPINN(rngs=nnx.Rngs(0))
        >>> x = jnp.array([[0.5, 0.5]])
        >>> lap = compute_laplacian_nnx(model, x)
    """
    graphdef, state = nnx.split(model)

    @partial(jax.jit, static_argnums=(0,))
    def _laplacian_fn(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        x_batch: Float[Array, "batch dim"],
    ) -> Float[Array, batch]:
        model_reconstructed = nnx.merge(graphdef, state)

        def compute_single_laplacian(x_single: Float[Array, dim]) -> Float[Array, ""]:
            def model_output_scalar(x_input: Float[Array, dim]) -> Float[Array, ""]:
                out = model_reconstructed(x_input[None, ...])
                return out[0, output_idx]

            hessian = jax.hessian(model_output_scalar)(x_single)
            return jnp.trace(hessian)

        return jax.vmap(compute_single_laplacian)(x_batch)

    return _laplacian_fn(graphdef, state, x)


# =============================================================================
# AutoDiffEngine Class - Namespace for convenience
# =============================================================================


class AutoDiffEngine:
    """
    Namespace class grouping all autodiff utility functions.

    This class provides a convenient way to pass all autodiff utilities
    to PDE residual functions. All methods are static - this is purely
    for organizational convenience.

    Examples:
        Pass to PDE residual:
        >>> def my_pde_residual(model, x, autodiff_engine):
        ...     lap = autodiff_engine.compute_laplacian(model, x)
        ...     return lap

        Use directly:
        >>> from opifex.core.physics import AutoDiffEngine
        >>> grad = AutoDiffEngine.compute_gradient(f, x)
    """

    # Assign functions as static attributes
    compute_gradient = staticmethod(compute_gradient)
    compute_laplacian = staticmethod(compute_laplacian)
    compute_divergence = staticmethod(compute_divergence)
    compute_hessian = staticmethod(compute_hessian)
    compute_gradient_nnx = staticmethod(compute_gradient_nnx)
    compute_laplacian_nnx = staticmethod(compute_laplacian_nnx)


__all__ = [
    "AutoDiffEngine",
    "compute_divergence",
    "compute_gradient",
    "compute_gradient_nnx",
    "compute_hessian",
    "compute_laplacian",
    "compute_laplacian_nnx",
]
