"""Utility functions for training infrastructure."""

from __future__ import annotations

from typing import Any

import jax


def safe_model_call(model: Any, x: jax.Array, **kwargs) -> jax.Array:
    """Safely call model with proper error handling."""
    try:
        # Try calling with kwargs first
        return model(x, **kwargs)
    except (TypeError, AttributeError):
        # Fallback: try calling without kwargs
        try:
            return model(x)
        except (TypeError, AttributeError):
            # Final fallback: try to find a forward method
            if hasattr(model, "forward"):
                return model.forward(x, **kwargs)
            if callable(model):
                return model.__call__(x, **kwargs)
            raise ValueError(f"Model {type(model)} is not callable") from None


def safe_compute_energy(model: Any, positions: jax.Array, **kwargs) -> jax.Array:
    """Safely compute energy with fallback for non-quantum models."""
    # Remove any problematic kwargs to avoid JAX issues
    clean_kwargs = {}
    if "deterministic" in kwargs:
        clean_kwargs["deterministic"] = kwargs["deterministic"]

    if hasattr(model, "compute_energy"):
        try:
            return model.compute_energy(positions, **clean_kwargs)
        except Exception:
            # Fallback to standard model call if compute_energy fails
            pass

    # Fallback for standard models - flatten positions and pass through
    if positions.ndim == 3:  # batched
        batch_size = positions.shape[0]
        flat_positions = positions.reshape(batch_size, -1)
    else:
        flat_positions = positions.flatten()[None, :]

    # Use direct model call without problematic kwargs
    try:
        return model(
            flat_positions, deterministic=clean_kwargs.get("deterministic", True)
        )
    except Exception:
        # Final fallback - simplest possible call
        return model(flat_positions)


def safe_compute_forces(model: Any, positions: jax.Array, **kwargs) -> jax.Array:
    """Safely compute forces with fallback for non-quantum models."""
    if hasattr(model, "compute_forces"):
        # Handle rngs parameter correctly for JAX transformations
        safe_kwargs = kwargs.copy()
        if "rngs" in safe_kwargs:
            # Don't pass rngs during JIT compilation - it can cause segmentation faults
            safe_kwargs.pop("rngs")
            if "deterministic" not in safe_kwargs:
                safe_kwargs["deterministic"] = (
                    True  # Default to deterministic if rngs present
                )
        return model.compute_forces(positions, **safe_kwargs)

    # Basic gradient-based forces for standard models
    def energy_fn(pos):
        return safe_compute_energy(model, pos, **kwargs).sum()

    if positions.ndim == 2:
        return -jax.grad(energy_fn)(positions)
    if positions.ndim == 3:
        batched_grad = jax.vmap(jax.grad(energy_fn))
        return -batched_grad(positions)
    raise ValueError(f"Unsupported positions shape: {positions.shape}")
