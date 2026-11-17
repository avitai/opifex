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


def signed_lower_residual_score(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    cutoff: float | jax.Array | None = None,
    scales: jax.Array | None = None,
    truncate_predictions: bool = False,
    scale_floor: float = 1e-6,
) -> jax.Array:
    """One-sided signed nonconformity score for lower conformal bounds.

    Computes ``(f_trunc - y_trunc) / sigma_+`` so positive scores penalize outcomes falling
    below the predictor (Candès, Lei & Ren 2023, JRSS-B 85(1):24-45).

    Args:
        predictions: Point or conditional lower-quantile predictions.
        targets: Observed outcomes (truncated at ``cutoff`` when ``cutoff`` is supplied).
        cutoff: Optional censoring cutoff ``c_0`` applied via ``min(targets, cutoff)``.
        scales: Optional positive local uncertainty scales ``sigma(X)``.
        truncate_predictions: When ``True``, replaces ``predictions`` by
            ``min(predictions, cutoff)`` for quantile-equivariant CQR scoring.
        scale_floor: Strictly positive lower clamp applied to ``scales``.

    Returns:
        1-D or batched array of signed nonconformity scores.

    Raises:
        ValueError: When ``scale_floor <= 0`` or concrete ``scales`` contain non-positive values.
    """
    if scale_floor <= 0.0:
        raise ValueError(f"scale_floor must be strictly positive; got {scale_floor!r}.")
    dtype = jnp.result_type(predictions, targets, jnp.float32)
    preds = jnp.asarray(predictions, dtype=dtype)
    targs = jnp.asarray(targets, dtype=dtype)

    if cutoff is not None:
        c0 = jnp.asarray(cutoff, dtype=dtype)
        safe_targs = jnp.where(jnp.isfinite(targs), targs, c0)
        y_eff = jnp.where(jnp.isneginf(targs), targs, jnp.minimum(safe_targs, c0))
        f_eff = (
            jnp.where(jnp.isfinite(preds), jnp.minimum(preds, c0), c0)
            if truncate_predictions
            else preds
        )
    else:
        y_eff = targs
        f_eff = preds

    residual = f_eff - y_eff
    if scales is None:
        return residual

    scale_arr = jnp.asarray(scales, dtype=dtype)
    try:
        all_pos = bool(jnp.all(scale_arr > 0))
    except jax.errors.TracerBoolConversionError:
        all_pos = True
    if not all_pos:
        raise ValueError("scales must be strictly positive.")
    return residual / jnp.maximum(scale_arr, jnp.asarray(scale_floor, dtype=dtype))


def effective_sample_size(weights: jax.Array) -> jax.Array:
    """Kish's effective sample size ``(sum w_i)^2 / sum w_i^2`` with zero-sum guard.

    Args:
        weights: Non-negative calibration weights.

    Returns:
        Scalar ``jax.Array`` in ``[0, n]`` (``0.0`` when all weights are zero).
    """
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w.dtype, jnp.float32)
    w_f = w.astype(dtype)
    sum_w = jnp.sum(w_f)
    sum_sq = jnp.sum(w_f * w_f)
    has_mass = sum_sq > 0.0
    safe_denom = jnp.where(has_mass, sum_sq, jnp.asarray(1.0, dtype=dtype))
    return jnp.where(has_mass, (sum_w * sum_w) / safe_denom, jnp.asarray(0.0, dtype=dtype))


def ipcw_conformal_quantile(
    *,
    scores: jax.Array,
    calibration_weights: jax.Array,
    test_weight: jax.Array | float,
    alpha: float,
) -> jax.Array:
    """Weighted ``(1 - alpha)`` conformal quantile with test mass at ``+inf``.

    Implements Eq. (5) of Tibshirani, Foygel Barber, Candès & Ramdas (2019, NeurIPS,
    arXiv:1904.06019) over ``n`` calibration atoms plus the test atom ``w_{n+1}(x)`` placed
    at ``+inf``. Excluded calibration entries (``calibration_weights <= 0``) are mapped to
    ``+inf`` before sorting so zero-weight points can never be selected as a finite quantile,
    and ``+inf`` is returned whenever ``sum_i p_i(x) < 1 - alpha``.

    Args:
        scores: 1-D array of calibration nonconformity scores of shape ``(n,)``.
        calibration_weights: 1-D array of non-negative IPCW weights of shape ``(n,)``.
        test_weight: Positive test-point weight ``w_{n+1}(x) = 1 / G(c_0 | x)``.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar ``jax.Array`` weighted conformal threshold (finite or ``+inf``).
    """
    dtype = jnp.result_type(scores, calibration_weights, test_weight, jnp.float32)
    scores_f = jnp.asarray(scores, dtype=dtype)
    weights_f = jnp.maximum(jnp.asarray(calibration_weights, dtype=dtype), 0.0)
    w_test = jnp.asarray(test_weight, dtype=dtype)

    active_scores = jnp.where(weights_f > 0.0, scores_f, jnp.asarray(jnp.inf, dtype=dtype))
    aug_scores = jnp.concatenate([active_scores, jnp.asarray([jnp.inf], dtype=dtype)], axis=0)
    aug_weights = jnp.concatenate([weights_f, w_test[None]], axis=0)

    total_weight = jnp.sum(aug_weights)
    safe_total = jnp.where(total_weight > 0.0, total_weight, jnp.asarray(1.0, dtype=dtype))

    order = jnp.argsort(aug_scores, stable=True)
    sorted_scores = aug_scores[order]
    sorted_weights = aug_weights[order]
    cumulative = jnp.cumsum(sorted_weights) / safe_total

    # Guard tiny float32 cumsum shortfall at the final +inf atom via explicit fallback to n.
    reached = cumulative >= jnp.asarray(1.0 - alpha, dtype=dtype)
    idx = jnp.where(jnp.any(reached), jnp.argmax(reached), sorted_scores.shape[0] - 1)
    return jnp.where(total_weight > 0.0, sorted_scores[idx], jnp.asarray(jnp.inf, dtype=dtype))
