"""Contracts for the learned-optimiser per-parameter feature primitives.

Verifies the formulas match the reference (``learned_optimization`` common/mlp_lopt).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.optimization.l2o import features


def test_momentum_ema_matches_recurrence() -> None:
    """Multi-decay momentum EMA equals decay*m + (1-decay)*grad on the trailing axis."""
    grad = jnp.array([1.0, -2.0, 3.0])
    momentum = features.init_ema(grad, features.MOMENTUM_DECAYS.size)
    assert momentum.shape == (3, 6)
    momentum = features.update_momentum(momentum, grad)
    # After one step from zero: m = (1 - decay) * grad.
    expected = (1.0 - features.MOMENTUM_DECAYS) * grad[:, None]
    assert jnp.allclose(momentum, expected)


def test_rms_ema_is_nonnegative_and_tracks_squared_grad() -> None:
    """Second-moment EMA accumulates grad**2 (non-negative)."""
    grad = jnp.array([2.0, -4.0])
    rms = features.init_ema(grad, features.MOMENTUM_DECAYS.size)
    rms = features.update_rms(rms, grad)
    expected = (1.0 - features.MOMENTUM_DECAYS) * (grad[:, None] ** 2)
    assert jnp.all(rms >= 0.0)
    assert jnp.allclose(rms, expected)


def test_second_moment_normalize_gives_unit_scale() -> None:
    """Normalised features have ~unit second moment along the normalised axis."""
    x = jnp.array([[3.0], [4.0], [0.0], [-5.0]])  # (n_params, 1)
    normed = features.second_moment_normalize(x, axis=0)
    assert jnp.isclose(jnp.mean(jnp.square(normed)), 1.0, atol=1e-3)


def test_tanh_time_embedding_shape_and_range() -> None:
    """The tanh embedding is length-11 and bounded in (-1, 1)."""
    emb = features.tanh_time_embedding(jnp.asarray(100.0))
    assert emb.shape == (11,)
    # tanh is bounded in [-1, 1]; short timescales legitimately saturate to 1.0.
    assert jnp.all(jnp.abs(emb) <= 1.0)
    # Monotone increasing in iteration for each timescale.
    later = features.tanh_time_embedding(jnp.asarray(1000.0))
    assert jnp.all(later >= emb - 1e-6)


def test_features_are_jit_and_vmap_safe() -> None:
    """Feature primitives compose under jit and vmap (transform compatibility)."""
    grads = jnp.ones((5, 3))  # batch of gradients

    @jax.jit
    def feat(g: jax.Array) -> jax.Array:
        m = features.update_momentum(features.init_ema(g, 6), g)
        return features.second_moment_normalize(m, axis=0)

    out = jax.vmap(feat)(grads)
    assert out.shape == (5, 3, 6)
    assert jnp.all(jnp.isfinite(out))
