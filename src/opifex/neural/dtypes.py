"""Shared dtype helpers for neural modules."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array  # noqa: TC002


def default_float_dtype() -> jnp.dtype:
    """Return the active default float dtype, following JAX's ``x64`` flag.

    Equals ``float32`` normally and ``float64`` when ``jax_enable_x64`` is set.
    Used as the ``param_dtype`` for parameter-holding layers (e.g. ``nnx.Linear``)
    so weights track the global precision uniformly with the rest of a module --
    matching how raw ``jax.random.normal`` weights already promote under ``x64`` --
    instead of being pinned to ``float32`` by Flax's hardcoded default.
    """
    return jnp.dtype(jnp.result_type(float))


def canonicalize_dtype(dtype: Any = jnp.float32) -> jnp.dtype:
    """Return a concrete JAX dtype for neural paths."""
    return jnp.dtype(dtype)


def as_compute_array(value: Array, dtype: Any = jnp.float32) -> Array:
    """Convert inputs to the module compute dtype."""
    return jnp.asarray(value, dtype=canonicalize_dtype(dtype))
