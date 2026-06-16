"""Operator-learning error metrics, built on calibrax's metric building blocks.

calibrax is the ecosystem's metric library. Its regression metrics include a
*global* ``relative_error`` (``sqrt(sum diff^2) / sqrt(sum target^2)``), but
operator learning conventionally reports the *per-sample-mean* relative L2 — the
mean over the batch of each sample's ``||diff|| / ||target||`` (the PDEBench
convention) — which calibrax does not provide. That metric is built here on
calibrax's own building blocks (``_prepare_arrays`` for shape validation and
``safe_divide`` for the shared ``1e-8`` epsilon guard), following how calibrax's
regression metrics are implemented, so the numerics stay consistent with the
rest of the ecosystem. It is the single source of truth used by both the
:class:`~opifex.core.training.trainer.Trainer` ``relative_l2`` loss and the
examples.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Reuse calibrax's metric building blocks (note 1: build on calibrax, do not
# reinvent). These are calibrax's shared numerical-stability helpers.
from calibrax.metrics._utils import _prepare_arrays, safe_divide


def per_sample_relative_l2(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Per-sample relative L2 error.

    Flattens the trailing dimensions of each leading-axis sample and returns the
    guarded ratio ``||prediction - target|| / ||target||`` for each sample.

    Args:
        prediction: Predicted field, shape ``(batch, *field_dims)``.
        target: Ground-truth field, same shape as ``prediction``.

    Returns:
        Per-sample relative L2 error, shape ``(batch,)``.
    """
    pred, tgt = _prepare_arrays(prediction, target)
    batch = pred.shape[0]
    numerator = jnp.linalg.norm((pred - tgt).reshape(batch, -1), axis=1)
    denominator = jnp.linalg.norm(tgt.reshape(batch, -1), axis=1)
    return safe_divide(numerator, denominator)


def relative_l2_error(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Mean per-sample relative L2 error (mean of :func:`per_sample_relative_l2`).

    Args:
        prediction: Predicted field, shape ``(batch, *field_dims)``.
        target: Ground-truth field, same shape as ``prediction``.

    Returns:
        Scalar mean relative L2 error.
    """
    return jnp.mean(per_sample_relative_l2(prediction, target))
