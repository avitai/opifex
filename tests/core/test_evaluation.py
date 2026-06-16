"""Tests for batched inference helpers (`opifex.core.evaluation`).

TDD: defines `predict_in_batches`, the single tested helper that replaces the
copy-pasted batched-evaluation loop in the neural-operator examples. The key
contract is that batching must not change the result versus a single forward
pass; chunk size is purely a memory-bounding knob.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.evaluation import predict_in_batches


def _scale(batch: jax.Array) -> jax.Array:
    return batch * 2.0 + 1.0


def test_matches_single_shot_application() -> None:
    """Batched application equals applying the function to the whole array."""
    x = jnp.arange(30.0).reshape(10, 3)
    batched = predict_in_batches(_scale, x, batch_size=4)
    assert jnp.allclose(batched, _scale(x))


def test_handles_non_divisible_batch_size() -> None:
    """A final short chunk is handled (10 not divisible by 4)."""
    x = jnp.arange(10.0).reshape(10, 1)
    out = predict_in_batches(_scale, x, batch_size=4)
    assert out.shape == (10, 1)
    assert jnp.allclose(out, _scale(x))


def test_batch_size_larger_than_n_is_single_chunk() -> None:
    """A batch size exceeding the sample count still returns all rows."""
    x = jnp.ones((3, 5))
    out = predict_in_batches(_scale, x, batch_size=128)
    assert out.shape == (3, 5)
    assert jnp.allclose(out, _scale(x))


def test_preserves_trailing_field_dims() -> None:
    """Multi-axis field outputs keep their trailing shape."""
    x = jnp.ones((7, 8, 8, 2))
    out = predict_in_batches(lambda b: b[..., :1], x, batch_size=3)
    assert out.shape == (7, 8, 8, 1)


def test_accepts_a_jitted_function() -> None:
    """The per-batch function is typically jitted by the caller."""
    x = jnp.arange(30.0).reshape(10, 3)
    jitted = jax.jit(lambda b: b * 2.0 + 1.0)
    out = predict_in_batches(jitted, x, batch_size=4)
    assert jnp.allclose(out, x * 2.0 + 1.0)


def test_rejects_non_positive_batch_size() -> None:
    """A non-positive batch size is a programming error, surfaced eagerly."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        predict_in_batches(_scale, jnp.ones((4, 2)), batch_size=0)
