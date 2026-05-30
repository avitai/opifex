"""Shared mixed-precision primitive operations.

This leaf module holds mixed-precision helpers that are reused by more than
one mixed-precision implementation (the
:class:`opifex.core.training.strategies.mixed_precision.MixedPrecisionTrainer`
strategy and the
:class:`opifex.core.training.components.MixedPrecisionComponent` building
block). Keeping the single source of truth here avoids duplicating the
overflow-detection logic across both call sites (DRY).

The module deliberately has no dependencies beyond JAX so that either consumer
can import it without pulling in a trainer or a component graph.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def check_for_overflow(grads: Any) -> bool:
    """Return ``True`` if any gradient leaf contains a non-finite value.

    Detects ``NaN`` / ``Inf`` produced by overflow during low-precision
    computation. The check is performed leaf-wise over an arbitrary pytree of
    gradients and reduced to a single boolean.

    Args:
        grads: A pytree of gradient arrays.

    Returns:
        ``True`` if any leaf contains a ``NaN`` or ``Inf`` value, otherwise
        ``False``.
    """

    def is_finite(leaf: jax.Array) -> jax.Array:
        return jnp.all(jnp.isfinite(leaf))

    finite_checks = jax.tree.map(is_finite, grads)
    return not jax.tree.reduce(lambda a, b: a and b, finite_checks, True)


__all__ = ["check_for_overflow"]
