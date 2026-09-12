"""Data helpers shared by the Markov-GP tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def binary_labels(times: jax.Array) -> jax.Array:
    """Return Bernoulli labels ``+1`` where ``sin(2t) >= 0`` and ``-1`` elsewhere.

    The Bernoulli likelihoods take labels in ``{-1, +1}``. ``jnp.sign(jnp.sin(2 t))`` returns ``0``
    wherever the sine vanishes, which a grid starting at ``t = 0`` hits.

    Args:
        times: Time stamps.

    Returns:
        Labels in ``{-1, +1}`` with the shape of ``times``.
    """
    return jnp.where(jnp.sin(2.0 * times) >= 0.0, 1.0, -1.0)
