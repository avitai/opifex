"""Conformal score functions and finite-sample quantile correction.

Score functions and the rank-based quantile rule are the bedrock of split
conformal prediction:

* :func:`absolute_residual_score` — split-conformal score for a point
  predictor ``|y - ŷ|`` (Lei et al. 2018, JASA).
* :func:`cqr_score` — CQR score ``max(lo - y, y - hi)`` (Romano, Patterson,
  Candes 2019, arXiv:1905.03222).
* :func:`conformal_quantile` — finite-sample-corrected
  ``ceil((n + 1)(1 - alpha)) / n`` quantile, ``+inf`` when that rank exceeds
  the sample (:func:`minimum_calibration_size` gives the smallest ``n`` that
  attains ``1 - alpha``).

Functions are pure ``jax.Array → jax.Array`` so they jit / vmap cleanly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def absolute_residual_score(*, predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Split-conformal score ``|y - ŷ|`` per Lei et al. 2018."""
    return jnp.abs(targets - predictions)


def cqr_score(*, lower: jax.Array, upper: jax.Array, targets: jax.Array) -> jax.Array:
    """CQR score ``max(lo - y, y - hi)`` per Romano, Patterson, Candes 2019."""
    return jnp.maximum(lower - targets, targets - upper)


def minimum_calibration_size(*, alpha: float) -> int:
    """Smallest calibration size whose ``(1 - alpha)`` conformal quantile exists.

    The finite-sample rank is ``ceil((n + 1)(1 - alpha)) / n``. It is a rank of
    ``n`` scores only while ``ceil((n + 1)(1 - alpha)) <= n``, which first holds
    at ``n = ceil(1 / alpha) - 1``. Below it no threshold attains ``1 - alpha``
    and :func:`conformal_quantile` returns ``+inf``.

    Args:
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The smallest usable calibration size, e.g. 9 at ``alpha=0.1``.

    """
    return math.ceil(1.0 / alpha) - 1


def conformal_quantile(*, scores: jax.Array, alpha: float) -> jax.Array:
    """Finite-sample-corrected ``(1 - alpha)`` quantile of ``scores``.

    Uses the finite-sample rank ``ceil((n + 1)(1 - alpha)) / n`` with the
    ``'higher'`` interpolation rule (Angelopoulos & Bates 2021,
    arXiv:2107.07511). The ``'higher'`` interpolation rule (a.k.a. ceil-based) preserves the
    finite-sample coverage bound; default linear interpolation does not.

    When ``ceil((n + 1)(1 - alpha)) > n`` the required rank is not a rank of
    ``n`` scores: no threshold drawn from them attains ``1 - alpha``, and the
    conformal quantile is ``+inf``. Returning the largest score instead would
    give marginal coverage of exactly ``n / (n + 1)``, below the requested
    level, with nothing to distinguish it from a valid threshold. MAPIE takes
    the same branch (``signed * np.inf`` in ``mapie.utils``) and crepes emits
    maximum-size intervals for a calibration set too small for the confidence
    level. Use :func:`minimum_calibration_size` to check ``n`` up front.

    ``n`` and ``alpha`` are static, so the branch is resolved at trace time and
    this stays jit- and vmap-compatible.

    Args:
        scores: 1-D array of nonconformity scores.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar quantile threshold using the ``'higher'`` rule, or ``+inf`` when
        ``scores`` is too small to attain ``1 - alpha``.

    """
    n = scores.shape[0]
    needed_rank = math.ceil((n + 1) * (1.0 - alpha))
    if needed_rank > n:
        return jnp.asarray(jnp.inf, dtype=jnp.result_type(scores.dtype, float))
    return jnp.quantile(scores, needed_rank / n, method="higher")
