"""Rank histogram for ensemble calibration assessment.

Hamill 2001, "Interpretation of rank histograms" — for each observed
target, count how many ensemble members it exceeds; histogram those
ranks over a calibration set. A perfectly calibrated ensemble produces a
uniform histogram across ``num_members + 1`` bins.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rank_histogram(*, ensemble: jax.Array, targets: jax.Array) -> jax.Array:
    """Count per-sample target ranks in the sorted-ensemble order.

    Args:
        ensemble: Shape ``(n_samples, n_members)``.
        targets: Shape ``(n_samples,)``.

    Returns:
        Integer counts of shape ``(n_members + 1,)``.

    """
    # Per-sample rank: number of ensemble members strictly less than the target.
    less = (ensemble < targets[:, None]).astype(jnp.int32)
    ranks = jnp.sum(less, axis=1)
    n_members = ensemble.shape[1]
    return jnp.bincount(ranks, length=n_members + 1)
