"""Per-parameter input features for learned optimisers.

Faithful re-implementation of the feature primitives in Google's
``learned_optimization`` (``learned_optimizers/common.py`` and
``learned_optimizers/mlp_lopt.py``). A coordinatewise learned optimiser consumes, per scalar
parameter: multi-timescale momentum and RMS EMAs of the gradient, the gradient and parameter
themselves (second-moment-normalised across the tensor), and a tanh embedding of the iteration
(training-fraction awareness). Feature design follows Metz et al. 2020 (``arXiv:2009.11243``).

These are pure functions on arrays; ``learned.py`` composes them into the per-parameter feature
vector fed to the optimiser MLP. Multi-decay EMAs carry the decay along a trailing axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# Momentum/RMS EMA decay rates (``learned_optimizers/mlp_lopt.py`` default ``decays``).
MOMENTUM_DECAYS: jax.Array = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
# Adafactor-MLP decay sets (``adafac_mlp_lopt.AdafacMLPLOpt`` defaults).
ADAFAC_MOMENTUM_DECAYS: jax.Array = jnp.asarray([0.9, 0.99, 0.999])
ADAFAC_RMS_DECAYS: jax.Array = jnp.asarray([0.999])
ADAFAC_DECAYS: jax.Array = jnp.asarray([0.9, 0.99, 0.999])
# Iteration tanh-embedding timescales (``mlp_lopt._tanh_embedding``).
TANH_TIMESCALES: jax.Array = jnp.asarray(
    [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000], dtype=jnp.float32
)


def init_ema(grad: jax.Array, num_decays: int) -> jax.Array:
    """Zero EMA buffer of shape ``grad.shape + (num_decays,)``."""
    return jnp.zeros((*grad.shape, num_decays), dtype=grad.dtype)


def update_momentum(
    momentum: jax.Array, grad: jax.Array, decays: jax.Array = MOMENTUM_DECAYS
) -> jax.Array:
    """Multi-decay momentum EMA: ``m = decay*m + (1-decay)*grad`` (decay on the last axis)."""
    return decays * momentum + (1.0 - decays) * grad[..., None]


def update_rms(rms: jax.Array, grad: jax.Array, decays: jax.Array = MOMENTUM_DECAYS) -> jax.Array:
    """Multi-decay second-moment EMA: ``rms = decay*rms + (1-decay)*grad**2``."""
    clipped = jnp.clip(decays, 0.0, 1.0)
    return clipped * rms + (1.0 - clipped) * (grad[..., None] ** 2)


def second_moment_normalize(x: jax.Array, axis: int = 0, eps: float = 1e-5) -> jax.Array:
    """Scale features to unit second moment along ``axis``.

    Matches ``mlp_lopt._second_moment_normalizer``.
    """
    return x * jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def tanh_time_embedding(iteration: jax.Array) -> jax.Array:
    """Tanh embedding of the iteration over 11 timescales (``mlp_lopt._tanh_embedding``).

    Returns a length-11 vector ``tanh(iteration / timescale - 1)`` — a smooth, bounded
    encoding of training progress shared across all parameters of a tensor.
    """
    return jnp.tanh(iteration / TANH_TIMESCALES - 1.0)


def safe_rsqrt(x: jax.Array) -> jax.Array:
    """Reciprocal square root with a floor (``common.safe_rsqrt``)."""
    return jax.lax.rsqrt(jnp.maximum(x, 1e-9))


def factored_dims(shape: tuple[int, ...]) -> tuple[int, int] | None:
    """Adafactor factoring dims: the two largest axes, or ``None`` if rank < 2.

    Matches ``learned_optimizers/common.factored_dims``: factored second-moment estimation only
    applies to tensors of rank >= 2; for those it factors the two largest dimensions. Returns
    ``(d1, d0)`` with ``d0`` the largest axis (reduced for the row estimate) and ``d1`` the
    second-largest (reduced for the column estimate).
    """
    if len(shape) < 2:
        return None
    sorted_dims = np.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def init_adafactor_accum(
    param: jax.Array, num_decays: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Zeroed factored accumulators ``(v_row, v_col, v_diag)`` for one parameter tensor.

    Decay is the leading axis. Rank-``>=2`` tensors use ``(v_row, v_col)`` (the unused ``v_diag``
    is an empty placeholder); rank-``<2`` tensors use a diagonal ``v_diag`` (RMSProp-style),
    mirroring ``common.factored_rolling``.
    """
    factored = factored_dims(param.shape)
    if factored is not None:
        d1, d0 = factored
        v_row = jnp.zeros((num_decays, *tuple(np.delete(param.shape, d0))))
        v_col = jnp.zeros((num_decays, *tuple(np.delete(param.shape, d1))))
        return v_row, v_col, jnp.zeros((num_decays, 0))
    empty = jnp.zeros((num_decays, 0))
    return empty, empty, jnp.zeros((num_decays, *param.shape))


def update_adafactor_accum(
    v_row: jax.Array,
    v_col: jax.Array,
    v_diag: jax.Array,
    grad: jax.Array,
    decays: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the factored accumulators and return Adafactor features for one tensor.

    Faithful to ``common.factored_rolling`` / ``adafac_mlp_lopt._mod``. Returns
    ``(new_v_row, new_v_col, new_v_diag, fac_g, row_feat, col_feat, factor)`` where the four
    feature arrays carry a trailing decay axis (``grad.shape + (num_decays,)``):
    ``fac_g`` is the Adafactor-preconditioned gradient, ``row_feat``/``col_feat`` the raw row/column
    second-moment estimates broadcast back to the tensor shape, and ``factor`` the
    ``row_factor * col_factor`` preconditioner (the diagonal ``rsqrt`` in the non-factored case).
    """
    factored = factored_dims(grad.shape)
    grad_sqr = grad * grad + 1e-30
    clipped = jnp.clip(decays, 0.0, 1.0)

    if factored is not None:
        d1, d0 = factored
        reduced_d1 = d1 - 1 if d1 > d0 else d1

        def update_one(decay: jax.Array, row: jax.Array, col: jax.Array):
            new_row = decay * row + (1.0 - decay) * jnp.mean(grad_sqr, axis=d0)
            new_col = decay * col + (1.0 - decay) * jnp.mean(grad_sqr, axis=d1)
            row_col_mean = jnp.mean(new_row, axis=reduced_d1, keepdims=True)
            row_factor = safe_rsqrt(new_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(new_col)
            row_full = jnp.broadcast_to(jnp.expand_dims(row_factor, d0), grad.shape)
            col_full = jnp.broadcast_to(jnp.expand_dims(col_factor, d1), grad.shape)
            preconditioned = grad * row_full * col_full
            row_raw = jnp.broadcast_to(jnp.expand_dims(new_row, d0), grad.shape)
            col_raw = jnp.broadcast_to(jnp.expand_dims(new_col, d1), grad.shape)
            return new_row, new_col, preconditioned, row_raw, col_raw, row_full * col_full

        new_row, new_col, fac_g, row_feat, col_feat, factor = jax.vmap(update_one)(
            clipped, v_row, v_col
        )
        new_v_row, new_v_col, new_v_diag = new_row, new_col, v_diag
    else:

        def update_one_diagonal(decay: jax.Array, diag: jax.Array):
            new_diag = decay * diag + (1.0 - decay) * grad_sqr
            factor = safe_rsqrt(new_diag + 1e-9)
            return new_diag, grad * factor, new_diag, factor

        new_diag, fac_g, raw, factor = jax.vmap(update_one_diagonal)(clipped, v_diag)
        new_v_row, new_v_col, new_v_diag = v_row, v_col, new_diag
        row_feat = col_feat = raw

    def to_trailing(array: jax.Array) -> jax.Array:
        """Move the leading decay axis to the trailing position (S + (num_decays,))."""
        return jnp.moveaxis(array, 0, -1)

    return (
        new_v_row,
        new_v_col,
        new_v_diag,
        to_trailing(fac_g),
        to_trailing(row_feat),
        to_trailing(col_feat),
        to_trailing(factor),
    )


__all__ = [
    "ADAFAC_DECAYS",
    "ADAFAC_MOMENTUM_DECAYS",
    "ADAFAC_RMS_DECAYS",
    "MOMENTUM_DECAYS",
    "TANH_TIMESCALES",
    "factored_dims",
    "init_adafactor_accum",
    "init_ema",
    "safe_rsqrt",
    "second_moment_normalize",
    "tanh_time_embedding",
    "update_adafactor_accum",
    "update_momentum",
    "update_rms",
]
