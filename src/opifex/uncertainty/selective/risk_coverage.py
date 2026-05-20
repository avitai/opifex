"""Risk-coverage curve and area-under-risk-coverage (AURC).

Geifman & El-Yaniv 2017, "Selective Classification for Deep Neural
Networks" (NeurIPS, arXiv:1705.08500): sweep a confidence threshold from
highest to lowest; at each step ``k`` accept the top-``k`` highest-
confidence samples and compute the selective risk as the mean per-sample
error on the accepted set. AURC is the mean of those selective risks
over the coverage curve — equivalently, the rectangle-rule integral of
risk against coverage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def risk_coverage_curve(
    *,
    confidences: jax.Array,
    errors: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Risk-coverage curve at every per-sample accept threshold.

    Args:
        confidences: 1-D array of per-sample confidence scores; HIGHER →
            more likely to be accepted.
        errors: 1-D array of per-sample errors (HIGHER is WORSE).

    Returns:
        ``(coverages, risks)`` where ``coverages = [1/n, 2/n, ..., 1.0]``
        and ``risks[k-1]`` is the mean error on the top-``k``
        highest-confidence samples.

    """
    n = confidences.shape[0]
    order = jnp.argsort(-confidences)
    sorted_errors = errors[order]
    cumulative_errors = jnp.cumsum(sorted_errors)
    counts = jnp.arange(1, n + 1).astype(sorted_errors.dtype)
    risks = cumulative_errors / counts
    coverages = counts / n
    return coverages, risks


def area_under_risk_coverage(
    *,
    confidences: jax.Array,
    errors: jax.Array,
) -> jax.Array:
    """Area under the risk-coverage curve.

    Mean of the selective-risk values across the curve, evaluated at
    every per-sample threshold. Lower is better. Reduces to the
    unconditional mean error when confidence carries no information
    about error magnitude.
    """
    _, risks = risk_coverage_curve(confidences=confidences, errors=errors)
    return jnp.mean(risks)
