"""Batched inference helpers shared by the examples and evaluation paths.

A single tested implementation of "apply a function over an array in chunks",
replacing the batched-evaluation loop that was copy-pasted across the
neural-operator examples. Callers compose anything model-specific (a
``deterministic=True`` flag, a fixed DeepONet trunk, output un-normalisation)
by passing an appropriate callable.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp


def predict_in_batches(
    fn: Callable[[jax.Array], jax.Array],
    inputs: jax.Array,
    *,
    batch_size: int = 128,
) -> jax.Array:
    """Apply ``fn`` over ``inputs`` in chunks along the leading axis.

    Batching bounds peak memory without changing the result: the outputs of
    each chunk are concatenated back along axis 0.

    Args:
        fn: Callable mapping a batch ``(b, *dims)`` to outputs ``(b, *out_dims)``.
        inputs: Array whose leading axis is the sample axis.
        batch_size: Maximum number of samples per call to ``fn``.

    Returns:
        Concatenation of ``fn`` applied to each chunk, shape ``(N, *out_dims)``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    n_samples = inputs.shape[0]
    outputs = [fn(inputs[start : start + batch_size]) for start in range(0, n_samples, batch_size)]
    return jnp.concatenate(outputs, axis=0)
