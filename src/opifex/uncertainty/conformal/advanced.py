"""Cross-conformal, Jackknife+, and weighted conformal (covariate shift).

References (canonical):
* Barber, Candes, Ramdas, Tibshirani 2021 — "Predictive Inference with the
  Jackknife+" (arXiv:1905.02928). J+ guarantees coverage >= 1 - 2*alpha;
  CV+ is the K-fold variant.
* Tibshirani, Foygel Barber, Candes, Ramdas 2019 — "Conformal Prediction
  Under Covariate Shift" (arXiv:1904.06019). Weighted quantile of
  calibration scores against likelihood-ratio weights.

Numerical core mirrors ``fortuna.conformal.regression.jackknifeplus`` and
``fortuna.conformal.regression.cvplus`` (Apache-2.0). Returned intervals
are :class:`opifex.uncertainty.types.PredictionInterval` value objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import PredictionInterval


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


# ---------------------------------------------------------------------------
# Jackknife+ / CV+
# ---------------------------------------------------------------------------


def jackknife_plus_intervals(
    *,
    loo_val_predictions: jax.Array,
    loo_val_targets: jax.Array,
    loo_test_predictions: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """Jackknife+ prediction intervals per Barber et al. 2021.

    Args:
        loo_val_predictions: 1-D array of shape ``(n_calib,)``; the i-th
            entry is the prediction of the leave-one-out model trained
            without sample i, evaluated at sample i.
        loo_val_targets: 1-D array of shape ``(n_calib,)``; ground-truth
            for each LOO held-out sample.
        loo_test_predictions: 2-D array of shape ``(n_calib, n_test)``;
            entry ``[i, j]`` is the prediction of the model that left out
            sample i, evaluated at test input j.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with ``method="jackknife_plus"`` and
        metadata recording ``method``, ``alpha``, ``loo_size``,
        ``coverage_guarantee="1-2*alpha"``.

    """
    if loo_val_predictions.shape[0] != loo_val_targets.shape[0]:
        raise ValueError(
            "loo_val_predictions and loo_val_targets must share leading dim; "
            f"got {loo_val_predictions.shape[0]} vs {loo_val_targets.shape[0]}."
        )
    if loo_test_predictions.shape[0] != loo_val_predictions.shape[0]:
        raise ValueError(
            "loo_test_predictions leading dim must equal n_calib; "
            f"got {loo_test_predictions.shape[0]} vs {loo_val_predictions.shape[0]}."
        )
    residuals = loo_val_targets - loo_val_predictions  # (n_calib,)
    lower_candidates = loo_test_predictions - jnp.abs(residuals)[:, None]
    upper_candidates = loo_test_predictions + jnp.abs(residuals)[:, None]
    lower = jnp.quantile(lower_candidates, alpha, axis=0)
    upper = jnp.quantile(upper_candidates, 1.0 - alpha, axis=0)
    metadata: MetadataItems = (
        ("method", "jackknife_plus"),
        ("alpha", float(alpha)),
        ("loo_size", int(loo_val_predictions.shape[0])),
        ("coverage_guarantee", "1-2*alpha"),
        ("assumption_status", "exchangeable_assumed"),
    )
    return PredictionInterval(
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        method="jackknife_plus",
        metadata=metadata,
    )


def cv_plus_intervals(
    *,
    fold_val_predictions: jax.Array,
    fold_val_targets: jax.Array,
    fold_test_predictions: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """CV+ prediction intervals per Barber et al. 2021 (K-fold extension of J+).

    Args:
        fold_val_predictions: ``(n_folds, n_per_fold)`` per-fold held-out
            predictions.
        fold_val_targets: ``(n_folds, n_per_fold)`` per-fold held-out targets.
        fold_test_predictions: ``(n_folds, n_test)`` per-fold predictions
            on the test set.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with ``method="cv_plus"`` and metadata
        recording ``method``, ``alpha``, ``num_folds``,
        ``calibration_size_per_fold``, ``coverage_guarantee="1-2*alpha"``.

    """
    if (
        fold_val_predictions.shape[0] != fold_val_targets.shape[0]
        or fold_val_predictions.shape[0] != fold_test_predictions.shape[0]
    ):
        raise ValueError(
            "fold leading dim must match across all three inputs; got "
            f"{fold_val_predictions.shape[0]}, {fold_val_targets.shape[0]}, "
            f"{fold_test_predictions.shape[0]}."
        )
    n_folds, n_per_fold = fold_val_predictions.shape
    # Flatten per-fold residuals.
    residuals = (fold_val_targets - fold_val_predictions).reshape((-1,))
    # For each test point, every (fold, calib-residual) combination produces a
    # candidate lower/upper bound. Test predictions are constant within fold.
    test_preds_repeated = jnp.repeat(fold_test_predictions, n_per_fold, axis=0)
    lower_candidates = test_preds_repeated - jnp.abs(residuals)[:, None]
    upper_candidates = test_preds_repeated + jnp.abs(residuals)[:, None]
    lower = jnp.quantile(lower_candidates, alpha, axis=0)
    upper = jnp.quantile(upper_candidates, 1.0 - alpha, axis=0)
    metadata: MetadataItems = (
        ("method", "cv_plus"),
        ("alpha", float(alpha)),
        ("num_folds", int(n_folds)),
        ("calibration_size_per_fold", int(n_per_fold)),
        ("coverage_guarantee", "1-2*alpha"),
        ("assumption_status", "exchangeable_assumed"),
    )
    return PredictionInterval(
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        method="cv_plus",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Weighted conformal (covariate shift)
# ---------------------------------------------------------------------------


def weighted_conformal_quantile(
    *,
    scores: jax.Array,
    weights: jax.Array,
    alpha: float,
) -> jax.Array:
    """Weighted step-CDF ``(1 - alpha)``-quantile per Tibshirani et al. 2019.

    Canonical formula (Tibshirani, Foygel Barber, Candes, Ramdas 2019,
    arXiv:1904.06019, eq. 5) reused exactly by Angelopoulos's
    ``conformal-prediction/notebooks/weather-time-series-distribution-shift.ipynb``::

        weighted_quantile = inf{ q : sum_i normalised_w_i * 1{s_i <= q} >= 1 - alpha }

    where ``normalised_w_i = w_i / sum_j w_j``. The reference notebook uses
    ``scipy.optimize.brentq`` over a continuous ``q`` axis; on a sorted
    discrete score grid this is equivalent to picking the smallest sorted
    score whose cumulative normalised weight reaches ``1 - alpha``.

    Args:
        scores: 1-D array of nonconformity scores.
        weights: 1-D array of non-negative likelihood-ratio weights;
            normalised internally to sum to 1.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar weighted quantile threshold (always a score value from
        ``scores``).

    """
    normalized = weights / jnp.sum(weights)
    order = jnp.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = normalized[order]
    cumulative = jnp.cumsum(sorted_weights)
    target = 1.0 - alpha
    above = cumulative >= target
    any_above = jnp.any(above)
    chosen_idx = jnp.where(
        any_above, jnp.argmax(above.astype(jnp.int32)), sorted_scores.shape[0] - 1
    )
    return sorted_scores[chosen_idx]


def weighted_split_conformal_intervals(
    *,
    calibration_scores: jax.Array,
    calibration_weights: jax.Array,
    predictions: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """Weighted split-conformal intervals using a weighted score quantile.

    Returned interval bounds are ``predictions ± weighted_quantile``.
    ``metadata`` records ``method="weighted_split_conformal"`` and
    ``assumption_status="weights_required"`` to flag that distribution-free
    coverage holds only when ``calibration_weights`` are correct likelihood
    ratios.
    """
    threshold = weighted_conformal_quantile(
        scores=calibration_scores, weights=calibration_weights, alpha=alpha
    )
    metadata: MetadataItems = (
        ("method", "weighted_split_conformal"),
        ("score_type", "absolute_residual"),
        ("alpha", float(alpha)),
        ("calibration_size", int(calibration_scores.shape[0])),
        ("assumption_status", "weights_required"),
    )
    return PredictionInterval(
        lower=predictions - threshold,
        upper=predictions + threshold,
        coverage=1.0 - alpha,
        method="weighted_split_conformal",
        metadata=metadata,
    )
