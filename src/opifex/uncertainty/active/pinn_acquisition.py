"""PDE-residual acquisition for PINN-style active learning.

Reference: ``../al4pde`` (pool-based PDE active learning). The Pool-Based
class drives candidate selection via the PDE residual evaluated at each
pool element. The opifex JAX-native rewrite:

* Accepts either an ``nnx.Module`` or a plain callable as the surrogate.
* Delegates the residual computation to a caller-supplied
  ``residual_fn(predictions, candidates) -> residual``. This keeps the
  PDE physics out of the acquisition kernel and makes the function reusable
  for any first-order or higher-order operator (heat, Burgers, KdV, ...).
* Ranks candidates by residual norm and returns the top-``batch_size`` as
  an :class:`AcquiredBatch` carrying both residual and predictive-
  uncertainty metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.uncertainty.active.acquisition import AcquiredBatch


def _invoke_model(model: nnx.Module | Callable[..., jax.Array], x: jax.Array) -> jax.Array:
    """Invoke either an ``nnx.Module`` or a callable on ``x``.

    For ``nnx.Module`` we round-trip through ``nnx.split``/``nnx.merge``
    to honour the canonical NNX state pattern (avoids capturing the
    module in a closure for downstream ``jit``).
    """
    if isinstance(model, nnx.Module):
        graphdef, state = nnx.split(model)
        merged = nnx.merge(graphdef, state)
        return merged(x)
    return model(x)


def pinn_residual_acquisition(
    *,
    model: nnx.Module | Callable[..., jax.Array],
    candidates: jax.Array,
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    batch_size: int,
    rngs: nnx.Rngs | jax.Array,
    uncertainty_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> AcquiredBatch:
    r"""Rank candidates by ``|residual_fn(predictions, candidates)|``.

    Returns the top-``batch_size`` candidates and exposes both the
    per-candidate residual norm and a (placeholder) predictive
    uncertainty signal as ``metadata`` items so downstream consumers can
    inspect *why* each point was selected. When ``uncertainty_fn`` is
    supplied it is invoked on the model predictions; otherwise the
    metadata field carries zeros (deterministic surrogate).

    The acquisition is a JAX-traceable kernel — both ``_invoke_model``
    and ``residual_fn`` must operate purely on ``jax.Array`` inputs.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")
    _ = extract_rng_key(
        rngs,
        streams=("active_acquire", "default"),
        context="pinn_residual_acquisition",
    )

    predictions = _invoke_model(model, candidates)
    residual = residual_fn(predictions, candidates)
    if residual.ndim > 1:
        residual_norm = jnp.linalg.norm(residual.reshape(residual.shape[0], -1), axis=-1)
    else:
        residual_norm = jnp.abs(residual)

    uncertainty = (
        uncertainty_fn(predictions)
        if uncertainty_fn is not None
        else jnp.zeros_like(residual_norm)
    )

    # Combine: rank primarily by residual norm. The convention follows
    # al4pde's PoolBased — high-residual points are most informative.
    scores = residual_norm
    k = min(int(batch_size), int(scores.shape[0]))
    top_indices = jnp.argsort(scores)[-k:][::-1]

    metadata: tuple[tuple[str, Any], ...] = (
        ("residual_norm", tuple(float(r) for r in residual_norm)),
        ("predictive_uncertainty", tuple(float(u) for u in uncertainty)),
    )
    return AcquiredBatch(
        indices=top_indices,
        scores=scores,
        strategy="pinn_residual",
        metadata=metadata,
    )
