"""Numerical utilities for SINDy equation discovery.

Provides differentiation and data preprocessing functions in pure JAX.
"""

from __future__ import annotations

import jax.numpy as jnp


def finite_difference(
    x: jnp.ndarray,
    dt: float | jnp.ndarray,
    order: int = 2,
) -> jnp.ndarray:
    """Compute time derivatives via centered finite differences.

    Uses second-order centered differences for interior points and
    first-order forward/backward differences at boundaries.

    Args:
        x: State data, shape (n_samples, n_features).
        dt: Time step between samples.
        order: Derivative order (only 1 supported currently).

    Returns:
        Estimated derivatives, shape (n_samples, n_features).
    """
    # Centered differences for interior points
    x_dot = jnp.zeros_like(x)

    # Interior: (x[i+1] - x[i-1]) / (2*dt)
    x_dot = x_dot.at[1:-1].set((x[2:] - x[:-2]) / (2.0 * dt))

    # Forward difference at left boundary
    x_dot = x_dot.at[0].set((x[1] - x[0]) / dt)

    # Backward difference at right boundary
    return x_dot.at[-1].set((x[-1] - x[-2]) / dt)


def smooth_data(
    x: jnp.ndarray,
    window_size: int = 5,
) -> jnp.ndarray:
    """Smooth data using a moving average filter.

    Applies a uniform moving average along the time axis (axis 0).
    Boundary values are handled by truncating the kernel.

    Args:
        x: Data matrix, shape (n_samples, n_features).
        window_size: Number of points in the averaging window.

    Returns:
        Smoothed data, shape (n_samples, n_features).
    """
    kernel = jnp.ones(window_size) / window_size

    # Apply 1D convolution along time axis for each feature
    smoothed_cols = []
    for col_idx in range(x.shape[1]):
        col = x[:, col_idx]
        # Pad for "same" convolution
        pad_left = window_size // 2
        pad_right = window_size - 1 - pad_left
        padded = jnp.pad(col, (pad_left, pad_right), mode="edge")
        smoothed = jnp.convolve(padded, kernel, mode="valid")
        smoothed_cols.append(smoothed)

    return jnp.column_stack(smoothed_cols)


__all__ = ["finite_difference", "smooth_data"]
