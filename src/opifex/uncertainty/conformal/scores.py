"""Conformal score functions and finite-sample quantile correction.

Score functions and the rank-based quantile rule are the bedrock of split
conformal prediction. The formulas here mirror the canonical JAX-native
reference at ``fortuna.conformal.regression`` (Apache-2.0):

* :func:`absolute_residual_score` — split-conformal score for a point
  predictor ``|y - ŷ|`` (Lei et al. 2018, JASA).
* :func:`cqr_score` — CQR score ``max(lo - y, y - hi)`` (Romano, Patterson,
  Candes 2019, arXiv:1905.03222).
* :func:`conformal_quantile` — finite-sample-corrected
  ``ceil((n + 1)(1 - alpha)) / n`` quantile.

Functions are pure ``jax.Array → jax.Array`` so they jit / vmap cleanly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def absolute_residual_score(*, predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Split-conformal score ``|y - ŷ|`` per Lei et al. 2018."""
    return jnp.abs(targets - predictions)


def cqr_score(*, lower: jax.Array, upper: jax.Array, targets: jax.Array) -> jax.Array:
    """CQR score ``max(lo - y, y - hi)`` per Romano, Patterson, Candes 2019."""
    return jnp.maximum(lower - targets, targets - upper)


def conformal_quantile(*, scores: jax.Array, alpha: float) -> jax.Array:
    """Finite-sample-corrected ``(1 - alpha)`` quantile of ``scores``.

    Matches the canonical Angelopoulos & Bates reference implementation
    (``aangelopoulos/conformal-prediction``,
    ``notebooks/imagenet-{smallest-sets,aps,raps}.ipynb`` and
    ``meps-cqr.ipynb``) which uses
    ``np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')``.
    The ``'higher'`` interpolation rule (a.k.a. ceil-based) preserves the
    finite-sample coverage bound; default linear interpolation does not.

    Args:
        scores: 1-D array of nonconformity scores.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar quantile threshold using the ``'higher'`` rule.

    """
    n = scores.shape[0]
    rank = jnp.minimum(jnp.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return jnp.quantile(scores, rank, method="higher")
