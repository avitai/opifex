"""Shared dtype helpers for neural modules."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array  # noqa: TC002


def canonicalize_dtype(dtype: Any = jnp.float32) -> jnp.dtype:
    """Return a concrete JAX dtype for neural paths."""
    return jnp.dtype(dtype)


def as_compute_array(value: Array, dtype: Any = jnp.float32) -> Array:
    """Convert inputs to the module compute dtype."""
    return jnp.asarray(value, dtype=canonicalize_dtype(dtype))
