"""Tests for the shared neural dtype helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.neural.dtypes import (
    as_compute_array,
    canonicalize_dtype,
    default_float_dtype,
)


def test_default_float_dtype_is_float32_without_x64() -> None:
    """Without the x64 flag the default float dtype is float32."""
    assert default_float_dtype() == jnp.dtype(jnp.float32)


def test_default_float_dtype_follows_x64() -> None:
    """Under enable_x64 the default float dtype promotes to float64."""
    with jax.enable_x64(True):
        assert default_float_dtype() == jnp.dtype(jnp.float64)
    # The flag is scoped to the context manager and restored afterwards.
    assert default_float_dtype() == jnp.dtype(jnp.float32)


def test_canonicalize_dtype_default() -> None:
    """The canonicaliser returns a concrete float32 dtype by default."""
    assert canonicalize_dtype() == jnp.dtype(jnp.float32)


def test_as_compute_array_casts_to_requested_dtype() -> None:
    """Inputs are cast to the requested compute dtype."""
    out = as_compute_array(jnp.ones((3,), dtype=jnp.float64), dtype=jnp.float32)
    assert out.dtype == jnp.dtype(jnp.float32)
