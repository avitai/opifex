r"""Tests for ``active/rembo.py`` — Slice 22 (audit finding #4a).

Random EMbedding Bayesian Optimisation (Wang+ 2013, IJCAI). For
high-dimensional input spaces with low effective dimensionality
``d_eff << D``, REMBO:

1. Samples a random Gaussian projection ``A ∈ R^{D × d_eff}``.
2. Runs BO in the low-dim space ``y ∈ R^{d_eff}``.
3. Lifts each candidate via ``x = clip(A y, box)`` back to the full
   ``D``-dim space for objective evaluation.

References
----------
* Wang+ 2013 — *Bayesian Optimization in High Dimensions via Random
  Embeddings*, IJCAI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_random_embedding_matrix_has_correct_shape() -> None:
    """``random_embedding_matrix(...)`` returns a ``(D, d_eff)`` Gaussian matrix."""
    from opifex.uncertainty.active.rembo import random_embedding_matrix

    embedding = random_embedding_matrix(full_dim=20, low_dim=3, key=jax.random.PRNGKey(0))
    assert embedding.shape == (20, 3)
    assert jnp.all(jnp.isfinite(embedding))


def test_rembo_lift_maps_low_dim_to_full_dim_inside_box() -> None:
    """Lifting respects the search-space box via per-axis clipping."""
    from opifex.uncertainty.active.rembo import random_embedding_matrix, rembo_lift

    embedding = random_embedding_matrix(full_dim=10, low_dim=2, key=jax.random.PRNGKey(1))
    low_dim_point = jnp.array([0.5, -0.3])
    lower = jnp.full((10,), -1.0)
    upper = jnp.full((10,), 1.0)
    lifted = rembo_lift(
        low_dim_point=low_dim_point,
        embedding=embedding,
        lower=lower,
        upper=upper,
    )
    assert lifted.shape == (10,)
    assert jnp.all(lifted >= lower - 1e-6)
    assert jnp.all(lifted <= upper + 1e-6)


def test_rembo_lift_batched_handles_many_low_dim_candidates() -> None:
    """Lifting vectorises over a batch of low-dim candidates."""
    from opifex.uncertainty.active.rembo import random_embedding_matrix, rembo_lift_batched

    embedding = random_embedding_matrix(full_dim=8, low_dim=2, key=jax.random.PRNGKey(2))
    candidates = jnp.array([[0.2, 0.1], [-0.5, 0.4], [0.0, -0.3]])
    lower = jnp.full((8,), -1.0)
    upper = jnp.full((8,), 1.0)
    lifted = rembo_lift_batched(
        low_dim_candidates=candidates,
        embedding=embedding,
        lower=lower,
        upper=upper,
    )
    assert lifted.shape == (3, 8)
    assert jnp.all(jnp.isfinite(lifted))


def test_rembo_lift_is_jit_compatible() -> None:
    """Lifting compiles under ``jax.jit``."""
    from opifex.uncertainty.active.rembo import random_embedding_matrix, rembo_lift

    embedding = random_embedding_matrix(full_dim=6, low_dim=2, key=jax.random.PRNGKey(3))

    @jax.jit
    def lift(y: jax.Array) -> jax.Array:
        return rembo_lift(
            low_dim_point=y,
            embedding=embedding,
            lower=jnp.full((6,), -1.0),
            upper=jnp.full((6,), 1.0),
        )

    out = lift(jnp.array([0.1, 0.2]))
    assert out.shape == (6,)
