"""Risk-coverage curve, area-under-risk-coverage (AURC), and excess AURC (E-AURC).

References:
* Geifman & El-Yaniv 2017 — "Selective Classification for Deep Neural Networks"
  (NeurIPS, arXiv:1705.08500).
* Geifman, Uziel & El-Yaniv 2019 — "Bias-Reduced Uncertainty Estimation for Deep
  Neural Classifiers" (ICLR, arXiv:1805.08206): excess AURC subtracts the oracle
  AURC from the empirical AURC under the identical discrete Riemann sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def risk_coverage_curve(
    *,
    confidences: jax.Array,
    errors: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Tie-invariant risk-coverage curve at every per-sample accept threshold.

    Replaces tied confidence groups by their exact group-mean error (McSherry &
    Najork 2008) so constant or tied confidences yield exact ``mean(errors)``
    regardless of input permutation.

    Args:
        confidences: 1-D array of per-sample confidence scores; HIGHER →
            more likely to be accepted.
        errors: 1-D array of per-sample errors (HIGHER is WORSE).

    Returns:
        ``(coverages, risks)`` where ``coverages = [1/n, 2/n, ..., 1.0]``
        and ``risks[k-1]`` is the tie-averaged mean error on the top-``k``
        highest-confidence samples.

    Raises:
        ValueError: When ``confidences`` and ``errors`` are not non-empty 1-D arrays
            of identical shape.
    """
    if confidences.ndim != 1 or confidences.shape != errors.shape or confidences.shape[0] < 1:
        raise ValueError(
            "confidences and errors must be non-empty 1-D arrays of matching shape; "
            f"got {confidences.shape} and {errors.shape}."
        )
    dtype = jnp.result_type(confidences, errors, jnp.float32)
    conf_f = jnp.asarray(confidences, dtype=dtype)
    err_f = jnp.asarray(errors, dtype=dtype)
    n = conf_f.shape[0]

    order = jnp.argsort(-conf_f, stable=True)
    sorted_conf = conf_f[order]
    sorted_errors = err_f[order]

    group_starts = jnp.concatenate(
        [jnp.ones((1,), dtype=bool), sorted_conf[1:] != sorted_conf[:-1]],
        axis=0,
    )
    group_ids = jnp.cumsum(group_starts.astype(jnp.int32)) - 1
    group_sums = jax.ops.segment_sum(sorted_errors, group_ids, num_segments=n)
    group_counts = jax.ops.segment_sum(jnp.ones((n,), dtype=dtype), group_ids, num_segments=n)
    safe_counts = jnp.maximum(group_counts, jnp.asarray(1.0, dtype=dtype))
    tie_averaged_errors = group_sums[group_ids] / safe_counts[group_ids]

    cumulative_errors = jnp.cumsum(tie_averaged_errors)
    counts = jnp.arange(1, n + 1, dtype=dtype)
    risks = cumulative_errors / counts
    coverages = counts / jnp.asarray(n, dtype=dtype)
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

    Args:
        confidences: 1-D array of per-sample confidence scores.
        errors: 1-D array of per-sample errors.

    Returns:
        Scalar AURC value.
    """
    _, risks = risk_coverage_curve(confidences=confidences, errors=errors)
    return jnp.mean(risks)


def excess_aurc(
    *,
    confidences: jax.Array,
    errors: jax.Array,
) -> jax.Array:
    """Excess Area Under the Risk-Coverage curve (E-AURC; Geifman et al. 2019).

    Subtracts the oracle AURC (computed with ideal ranking ``confidences = -errors``
    under the identical discrete Riemann sum) from the empirical AURC, clamped
    below at ``0.0`` against floating-point rounding.

    Args:
        confidences: 1-D array of per-sample confidence scores.
        errors: 1-D array of per-sample errors.

    Returns:
        Non-negative scalar excess AURC (``0.0`` at an oracle confidence ranking).
    """
    aurc = area_under_risk_coverage(confidences=confidences, errors=errors)
    oracle_aurc = area_under_risk_coverage(confidences=-errors, errors=errors)
    return jnp.maximum(aurc - oracle_aurc, jnp.asarray(0.0, dtype=aurc.dtype))


__all__ = [
    "area_under_risk_coverage",
    "excess_aurc",
    "risk_coverage_curve",
]
