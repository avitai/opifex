"""Tests for the Bernoulli label helper shared by the Markov-GP tests.

The Bernoulli likelihoods take labels in ``{-1, +1}`` (``log sigma(y f)``). ``jnp.sign`` returns ``0``
wherever ``sin(2t) = 0``, and a grid that starts at ``t = 0`` hits that point.
"""

from __future__ import annotations

import jax.numpy as jnp

from tests.uncertainty.markov._helpers import binary_labels


def test_binary_labels_lie_in_plus_or_minus_one_where_the_sine_vanishes() -> None:
    """Grids that contain ``t = 0`` and ``t = pi / 2`` still produce labels in ``{-1, +1}``."""
    times = jnp.concatenate([jnp.linspace(0.0, 4.0, 18), jnp.asarray([jnp.pi / 2.0])])
    labels = binary_labels(times)
    assert bool(jnp.all((labels == 1.0) | (labels == -1.0)))


def test_binary_labels_follow_the_sign_of_the_sine_elsewhere() -> None:
    """Away from the zeros of ``sin(2t)``, the labels equal ``sign(sin(2t))``."""
    times = jnp.linspace(0.1, 4.0, 25)
    signal = jnp.sin(2.0 * times)
    away_from_zero = jnp.abs(signal) > 1e-3
    assert bool(jnp.all(jnp.where(away_from_zero, binary_labels(times) == jnp.sign(signal), True)))
