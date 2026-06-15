r"""REMBO — Random Embedding Bayesian Optimisation (Wang+ 2013) — Slice 22.

For high-dimensional input spaces ``x ∈ R^D`` with low effective
dimensionality ``d_eff << D``, REMBO samples a random Gaussian
projection ``A ∈ R^{D × d_eff}`` and runs BO entirely in the low-dim
embedded space ``y ∈ R^{d_eff}``. Each low-dim candidate is lifted to
the full space via ``x = clip(A y, box)`` for objective evaluation.

API:

* :func:`random_embedding_matrix` — sample ``A``.
* :func:`rembo_lift` — lift one low-dim point ``y`` to a full-dim
  point ``x``.
* :func:`rembo_lift_batched` — vectorised version for a batch of
  candidates.

References
----------
* Wang, Hutter, Zoghi, Matheson, de Freitas 2013 — *Bayesian
  Optimization in High Dimensions via Random Embeddings*, IJCAI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def random_embedding_matrix(
    *,
    full_dim: int,
    low_dim: int,
    key: jax.Array,
) -> jax.Array:
    r"""Return a ``(full_dim, low_dim)`` standard-normal random embedding matrix.

    Per Wang+ 2013 §3.1, the entries ``A_{ij} ~ N(0, 1)`` are
    independent. The resulting random embedding satisfies the
    Johnson-Lindenstrauss-style guarantee that ``A y`` lies in the
    span of an arbitrary ``d_eff``-dim subspace of ``R^D`` with high
    probability.
    """
    if full_dim <= 0 or low_dim <= 0:
        raise ValueError(f"full_dim and low_dim must be positive; got {full_dim=} {low_dim=}.")
    return jax.random.normal(key, (full_dim, low_dim))


def rembo_lift(
    *,
    low_dim_point: jax.Array,
    embedding: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    r"""Lift a single low-dim point ``y`` to the full search-space box.

    Computes ``x = clip(A y, [lower, upper])`` per Wang+ 2013 §3.2.

    Args:
        low_dim_point: ``(d_eff,)`` low-dim point.
        embedding: ``(D, d_eff)`` random embedding matrix.
        lower: ``(D,)`` global lower bounds.
        upper: ``(D,)`` global upper bounds.

    Returns:
        ``(D,)`` full-dim point inside the search-space box.
    """
    raw = embedding @ low_dim_point
    return jnp.clip(raw, lower, upper)


def rembo_lift_batched(
    *,
    low_dim_candidates: jax.Array,
    embedding: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    r"""Vectorised :func:`rembo_lift` over a batch of low-dim candidates.

    Args:
        low_dim_candidates: ``(N, d_eff)`` batch.
        embedding: ``(D, d_eff)`` matrix.
        lower / upper: ``(D,)`` global bounds.

    Returns:
        ``(N, D)`` lifted candidates inside the search-space box.
    """
    raw = low_dim_candidates @ embedding.T
    return jnp.clip(raw, lower, upper)


__all__ = [
    "random_embedding_matrix",
    "rembo_lift",
    "rembo_lift_batched",
]
