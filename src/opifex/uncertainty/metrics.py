"""Ensemble and interval scoring metrics for UQ.

This module hosts metrics that have NO canonical location in CalibraX and
that are specific to the Opifex predictive-distribution / interval surface:

* :func:`predictive_entropy` — ``H(mean_m p_m)`` for an ensemble of
  classification probabilities (Gal & Ghahramani 2016, "Dropout as a
  Bayesian Approximation").
* :func:`mutual_information` — ``H(mean_m p_m) - mean_m H(p_m)``, the
  epistemic-uncertainty component of the BALD decomposition
  (Houlsby et al. 2011, "Bayesian Active Learning by Disagreement").
* :func:`interval_score` / :func:`winkler_score` — Gneiting & Raftery
  2007 strictly proper interval score
  ``IS_α(l, u, y) = (u - l) + (2/α)·(l - y)_+ + (2/α)·(y - u)_+``.

For Gaussian NLL, Brier, ECE, pinball/quantile loss, regression
calibration error, PICP, MPIW, and the wrapped CalibraX classification
metrics, import from :mod:`opifex.uncertainty.calibration` directly — we
do not re-export them here to avoid forward shims (per Rule 1 / the
project's "don't shim" feedback).

All kernels are pure ``jax.Array → jax.Array`` functions; they trace
cleanly under ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` and contain no
NNX imports (a structural invariant for ``opifex.uncertainty`` metric
modules).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


_ENTROPY_EPS: float = 1e-12


def predictive_entropy(*, ensemble_probabilities: jax.Array) -> jax.Array:
    """Entropy of the ensemble-averaged categorical distribution.

    Computes ``H(mean_m p_m)`` where ``m`` is the ensemble-member axis.
    The leading axis of ``ensemble_probabilities`` is the ensemble
    dimension; remaining leading axes are batch; the last axis is the
    class axis.

    Args:
        ensemble_probabilities: Array of shape
            ``(num_members, *batch, num_classes)`` with non-negative
            entries summing to 1 along the last axis.

    Returns:
        Array of shape ``(*batch,)`` of per-sample predictive entropies.

    """
    mean_probs = jnp.mean(ensemble_probabilities, axis=0)
    safe_probs = mean_probs + _ENTROPY_EPS
    return -jnp.sum(mean_probs * jnp.log(safe_probs), axis=-1)


def mutual_information(*, ensemble_probabilities: jax.Array) -> jax.Array:
    """BALD-style epistemic mutual information.

    Computes ``H(mean_m p_m) - mean_m H(p_m)`` — the
    predictive-entropy minus the expected-entropy decomposition that
    Gal & Ghahramani 2016 use for epistemic uncertainty.

    Args:
        ensemble_probabilities: Array of shape
            ``(num_members, *batch, num_classes)``.

    Returns:
        Array of shape ``(*batch,)`` of per-sample mutual-information
        scores.

    """
    safe = ensemble_probabilities + _ENTROPY_EPS
    member_entropies = -jnp.sum(ensemble_probabilities * jnp.log(safe), axis=-1)
    return predictive_entropy(ensemble_probabilities=ensemble_probabilities) - jnp.mean(
        member_entropies, axis=0
    )


def interval_score(
    *,
    lower: jax.Array,
    upper: jax.Array,
    targets: jax.Array,
    alpha: float,
) -> jax.Array:
    """Strictly proper interval score per Gneiting & Raftery 2007.

    ``IS_α(l, u, y) = (u - l) + (2/α)·(l - y)_+ + (2/α)·(y - u)_+``

    Lower is better. Penalises width (``u - l``) plus a 2/α multiple of
    the one-sided miscoverage gap. The score is strictly proper for
    central ``(1 - α)`` prediction intervals.

    Args:
        lower: Lower interval bound of shape ``(batch, ...)``.
        upper: Upper interval bound, same shape.
        targets: Observed values, same shape.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Per-sample interval scores (same shape as ``targets``).

    """
    width = upper - lower
    below = jnp.maximum(0.0, lower - targets)
    above = jnp.maximum(0.0, targets - upper)
    return width + (2.0 / alpha) * (below + above)


def winkler_score(
    *,
    lower: jax.Array,
    upper: jax.Array,
    targets: jax.Array,
    alpha: float,
) -> jax.Array:
    """Alias for :func:`interval_score` — Winkler's 1972 original name."""
    return interval_score(lower=lower, upper=upper, targets=targets, alpha=alpha)


# ---------------------------------------------------------------------------
# Probnum-evaluation vendored calibration metrics
#
# Vendors the ANEES + NCI + chi-squared confidence-interval primitives from
# probnum-evaluation/src/probnumeval/timeseries/_calibration_measures.py.
# That reference is itself a stub (most bodies raise NotImplementedError);
# the formulas here come directly from the cited Bar-Shalom 2002 and
# Li+Zhao 2006 IFAC papers.
# ---------------------------------------------------------------------------


def _per_step_mahalanobis(errors: jax.Array, covariances: jax.Array) -> jax.Array:
    """Per-timestep Mahalanobis distance ``e_t^T P_t^{-1} e_t``."""
    solved = jax.vmap(jnp.linalg.solve)(covariances, errors)
    return jnp.einsum("ti,ti->t", errors, solved)


def anees(
    *,
    predicted_means: jax.Array,
    predicted_covariances: jax.Array,
    references: jax.Array,
) -> jax.Array:
    r"""Average Normalised Estimation Error Squared.

    For a sequence of predictions ``(m_t, P_t)`` and references ``r_t`` of
    dimension ``d``,

    .. math::

        \mathrm{ANEES} = \frac{1}{N d} \sum_t (r_t - m_t)^{\top} P_t^{-1} (r_t - m_t).

    Optimal value is ``1``: ``ANEES > 1`` indicates over-confidence (the
    predicted covariance is too small); ``ANEES < 1`` indicates
    under-confidence. Cite Bar-Shalom 2002 IFAC.

    Args:
        predicted_means: Shape ``(num_steps, d)``.
        predicted_covariances: Shape ``(num_steps, d, d)``.
        references: Shape ``(num_steps, d)``.

    Returns:
        Scalar ANEES.
    """
    errors = references - predicted_means
    mahalanobis = _per_step_mahalanobis(errors, predicted_covariances)
    return jnp.mean(mahalanobis) / errors.shape[-1]


def non_credibility_index(
    *,
    predicted_means: jax.Array,
    predicted_covariances: jax.Array,
    references: jax.Array,
    reference_covariances: jax.Array,
) -> jax.Array:
    r"""Non-Credibility Index of Li & Zhao 2006 IFAC.

    .. math::

        \mathrm{NCI} = \frac{10}{N} \sum_t \log_{10} \bigl(
            (r_t - m_t)^{\top} P_t^{-1} (r_t - m_t) /
            (r_t - m_t)^{\top} S_t^{-1} (r_t - m_t) \bigr),

    where ``P_t`` is the predicted covariance and ``S_t`` the reference
    error covariance. ``NCI = 0`` indicates a calibrated estimator;
    positive ``NCI`` means the predicted covariance underestimates the
    true error covariance.

    Args:
        predicted_means: Shape ``(num_steps, d)``.
        predicted_covariances: Shape ``(num_steps, d, d)``.
        references: Shape ``(num_steps, d)``.
        reference_covariances: True / reference error covariance, shape
            ``(num_steps, d, d)``.

    Returns:
        Scalar NCI in decibels.
    """
    errors = references - predicted_means
    predicted_mahalanobis = _per_step_mahalanobis(errors, predicted_covariances)
    reference_mahalanobis = _per_step_mahalanobis(errors, reference_covariances)
    ratio = predicted_mahalanobis / reference_mahalanobis
    return 10.0 * jnp.mean(jnp.log10(ratio))


def chi2_confidence_intervals(*, dim: int, percentile: float = 0.99) -> tuple[jax.Array, jax.Array]:
    """Symmetric ``percentile``-level confidence interval of ``chi^2_dim``.

    Returns the ``(lower, upper)`` quantiles at probability levels
    ``(1 - percentile) / 2`` and ``1 - (1 - percentile) / 2``. Ports
    :func:`probnumeval.timeseries._calibration_measures.chi2_confidence_intervals`.

    Args:
        dim: Degrees of freedom of the chi-squared distribution.
        percentile: Interval coverage in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` quantile pair.

    Raises:
        ValueError: If ``percentile`` is not strictly between 0 and 1.
    """
    if not 0.0 < percentile < 1.0:
        raise ValueError(f"percentile must be in (0, 1); got {percentile!r}.")
    import scipy.stats

    delta = (1.0 - percentile) / 2.0
    lower = jnp.asarray(scipy.stats.chi2(df=dim).ppf(delta))
    upper = jnp.asarray(scipy.stats.chi2(df=dim).ppf(1.0 - delta))
    return lower, upper
