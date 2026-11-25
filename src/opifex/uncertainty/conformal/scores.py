"""Conformal score functions and finite-sample quantile correction.

Score functions and the rank-based quantile rule are the bedrock of split
conformal prediction:

* :func:`absolute_residual_score` — split-conformal score for a point
  predictor ``|y - ŷ|`` (Lei et al. 2018, JASA).
* :func:`cqr_score` — CQR score ``max(lo - y, y - hi)`` (Romano, Patterson,
  Candes 2019, arXiv:1905.03222).
* :func:`conformal_quantile` — finite-sample-corrected
  ``ceil((n + 1)(1 - alpha))``-th smallest order statistic, ``+inf`` when that
  rank exceeds ``n`` (:func:`minimum_calibration_size` gives the smallest ``n``
  that attains ``1 - alpha``).

Functions are pure ``jax.Array → jax.Array`` so they jit / vmap cleanly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from calibrax.statistics import (
    conformal_quantile_rank,
    minimum_conformal_calibration_size,
)


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
        alpha: Static miscoverage level in ``(0, 1)`` (validated via
            :func:`calibrax.statistics.minimum_conformal_calibration_size`, raising
            ``ValueError`` outside ``(0, 1)`` and ``TypeError`` if traced or ``bfloat16``).

    Returns:
        The smallest usable calibration size, e.g. 9 at ``alpha=0.1``.
    """
    return minimum_conformal_calibration_size(alpha)


def conformal_quantile(*, scores: jax.Array, alpha: float) -> jax.Array:
    """Finite-sample-corrected ``(1 - alpha)`` order statistic of ``scores``.

    Selects the exact ``k = ceil((n + 1)(1 - alpha))``-th smallest element of
    ``scores`` (the ``inverted_cdf`` discrete quantile rule used in Lei et al.
    2018, Angelopoulos & Bates 2021 ``arXiv:2107.07511``, MAPIE, and puncc),
    which guarantees marginal coverage at least ``1 - alpha`` for exchangeable
    samples.

    When ``ceil((n + 1)(1 - alpha)) > n`` the required rank exceeds ``n``: no
    threshold drawn from ``scores`` attains ``1 - alpha``, and the conformal
    quantile is ``+inf``. Returning the largest score would give marginal
    coverage of ``n / (n + 1)``, below the requested level, with nothing to
    distinguish it from a valid threshold. MAPIE takes the same branch
    (``signed * np.inf`` in ``mapie.utils``) and crepes emits maximum-size
    intervals for a calibration set too small for the confidence level. Use
    :func:`minimum_calibration_size` to check ``n`` up front.

    ``n`` and ``alpha`` are static, so the branch is resolved at trace time and
    this stays jit- and vmap-compatible.

    Args:
        scores: 1-D array of nonconformity scores.
        alpha: Static miscoverage level in ``(0, 1)`` (validated via
            :func:`calibrax.statistics.conformal_quantile_rank`, raising ``TypeError``
            when traced or ``bfloat16``).

    Returns:
        Scalar ``ceil((n + 1)(1 - alpha))``-th order statistic, ``+inf`` when
        ``scores`` has fewer than :func:`minimum_calibration_size` elements, or
        ``NaN`` if ``scores`` contains ``NaN``.

    Raises:
        ValueError: When ``alpha`` lies outside ``(0, 1)`` or ``scores`` is not a
            non-empty 1-D array.
    """
    if scores.ndim != 1 or scores.shape[0] < 1:
        raise ValueError(f"scores must be a non-empty 1-D array; got shape {scores.shape}.")
    n = scores.shape[0]
    needed_rank = conformal_quantile_rank(n, alpha)
    dtype = jnp.result_type(scores, jnp.float32)
    scores_f = jnp.asarray(scores, dtype=dtype)
    has_nan = jnp.any(jnp.isnan(scores_f))
    nan_val = jnp.asarray(jnp.nan, dtype=dtype)
    if needed_rank > n:
        return jnp.where(has_nan, nan_val, jnp.asarray(jnp.inf, dtype=dtype))
    return jnp.where(has_nan, nan_val, jnp.sort(scores_f)[needed_rank - 1])


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


def _require_nonnegative_weights(calibration_weights: jax.Array, w_test_arr: jax.Array) -> None:
    """Reject concrete negative calibration or test weights."""
    try:
        has_negative = bool(jnp.any(calibration_weights < 0)) or bool(w_test_arr < 0)
    except jax.errors.TracerBoolConversionError:
        has_negative = False
    if has_negative:
        raise ValueError("calibration_weights and test_weight must be non-negative.")


def _validate_ipcw_quantile_inputs(
    *,
    scores: jax.Array,
    calibration_weights: jax.Array,
    test_weight: jax.Array | float,
    alpha: float,
) -> int:
    """Validate shapes, static ``alpha`` bounds, and concrete weight non-negativity."""
    if scores.ndim != 1 or calibration_weights.ndim != 1:
        raise ValueError(
            "scores and calibration_weights must be non-empty 1-D arrays of matching shape; "
            f"got {scores.shape} and {calibration_weights.shape}."
        )
    if scores.shape != calibration_weights.shape or scores.shape[0] < 1:
        raise ValueError(
            "scores and calibration_weights must be non-empty 1-D arrays of matching shape; "
            f"got {scores.shape} and {calibration_weights.shape}."
        )
    needed_rank = conformal_quantile_rank(scores.shape[0], alpha)
    w_test_arr = jnp.asarray(test_weight)
    if w_test_arr.ndim != 0:
        raise ValueError(f"test_weight must be a scalar (0-D); got shape {w_test_arr.shape}.")
    _require_nonnegative_weights(calibration_weights, w_test_arr)
    return needed_rank


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
    at ``+inf``. Excluded calibration entries (``calibration_weights == 0``) are mapped to
    ``+inf`` before sorting so zero-weight points can never be selected as a finite quantile,
    and ``+inf`` is returned whenever ``sum_i p_i(x) < 1 - alpha``. Under equal positive
    weights, selects the exact ``conformal_quantile_rank(n, alpha)`` order statistic without
    floating-point rounding drift.

    Args:
        scores: 1-D array of calibration nonconformity scores of shape ``(n,)``.
        calibration_weights: 1-D array of non-negative IPCW weights of shape ``(n,)``.
        test_weight: Non-negative scalar test-point weight ``w_{n+1}(x)``.
        alpha: Static miscoverage level in ``(0, 1)``.

    Returns:
        Scalar ``jax.Array`` weighted conformal threshold (finite, ``+inf``, or ``NaN``).
    """
    needed_rank = _validate_ipcw_quantile_inputs(
        scores=scores,
        calibration_weights=calibration_weights,
        test_weight=test_weight,
        alpha=alpha,
    )
    dtype = jnp.result_type(scores, calibration_weights, test_weight, jnp.float32)
    scores_f = jnp.asarray(scores, dtype=dtype)
    weights_f = jnp.asarray(calibration_weights, dtype=dtype)
    w_test = jnp.asarray(test_weight, dtype=dtype)

    has_nan = (
        jnp.any(jnp.isnan(scores_f))
        | jnp.any(jnp.isnan(weights_f))
        | jnp.isnan(w_test)
        | (jnp.any(weights_f < 0.0))
        | (w_test < 0.0)
    )
    active_scores = jnp.where(weights_f > 0.0, scores_f, jnp.asarray(jnp.inf, dtype=dtype))
    aug_scores = jnp.concatenate([active_scores, jnp.asarray([jnp.inf], dtype=dtype)], axis=0)
    aug_weights = jnp.concatenate([weights_f, w_test[None]], axis=0)

    clean_weights = jnp.where(has_nan, jnp.zeros_like(aug_weights), aug_weights)
    has_posinf_w = jnp.any(jnp.isposinf(clean_weights))
    finite_weights = jnp.where(
        has_posinf_w, jnp.isposinf(clean_weights).astype(dtype), clean_weights
    )
    w_max = jnp.max(finite_weights)
    all_equal = (w_max > 0.0) & jnp.all(finite_weights == w_max)
    safe_max = jnp.where(w_max > 0.0, w_max, jnp.ones_like(w_max))
    unit_weights = finite_weights / safe_max

    order = jnp.argsort(aug_scores, stable=True)
    sorted_scores = aug_scores[order]
    sorted_weights = unit_weights[order]
    prefix = jnp.cumsum(sorted_weights)
    total = prefix[-1]
    tail = total - prefix

    reached = tail <= jnp.asarray(alpha, dtype=dtype) * total
    idx = jnp.where(jnp.any(reached), jnp.argmax(reached), sorted_scores.shape[0] - 1)
    idx = jnp.where(all_equal, jnp.int32(needed_rank - 1), idx)
    quantile = jnp.where(total > 0.0, sorted_scores[idx], jnp.asarray(jnp.inf, dtype=dtype))
    return jnp.where(has_nan, jnp.asarray(jnp.nan, dtype=dtype), quantile)
