"""Exchangeability diagnostic for conformal calibration.

Conformal validity rests on the exchangeability assumption (Vovk et al.
2005, "Algorithmic Learning in a Random World", §2.4) that calibration and
evaluation scores share a permutation-invariant joint distribution. A
classical lightweight check is a two-sample Kolmogorov–Smirnov test on the
score distributions: under exchangeability the test's null hypothesis (same
distribution) holds and ``p > alpha``; under covariate / label shift the
score distributions diverge and ``p`` drops.

The KS statistic is computed in pure JAX via empirical CDFs on the union
support, and the asymptotic Kolmogorov p-value approximation is used:

    p ≈ 2 * sum_{k=1}^{∞} (-1)^{k-1} exp(-2 k² λ²)

with ``λ = (sqrt(n_eff) + 0.12 + 0.11 / sqrt(n_eff)) * D`` and
``n_eff = n1 n2 / (n1 + n2)``. We truncate the series at ``k <= 100``
(asymptotically exact for ``n_eff >= 10``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.types import MetadataItems  # noqa: TC001


@struct.dataclass(slots=True, kw_only=True)
class ExchangeabilityReport:
    """Outcome of a two-sample exchangeability check."""

    p_value: jax.Array
    passes: bool = struct.field(pytree_node=False)
    method: str = struct.field(pytree_node=False, default="ks_two_sample")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def ks_two_sample_pvalue(
    *,
    calibration_scores: jax.Array,
    evaluation_scores: jax.Array,
) -> jax.Array:
    """Two-sample Kolmogorov–Smirnov p-value (asymptotic).

    Pure JAX; traces under ``jax.jit`` and ``jax.vmap``.
    """
    n1 = calibration_scores.shape[0]
    n2 = evaluation_scores.shape[0]
    combined = jnp.concatenate([calibration_scores, evaluation_scores])
    cal_sorted = jnp.sort(calibration_scores)
    evl_sorted = jnp.sort(evaluation_scores)
    # Empirical CDFs evaluated on the union support.
    cdf_cal = jnp.searchsorted(cal_sorted, combined, side="right") / n1
    cdf_evl = jnp.searchsorted(evl_sorted, combined, side="right") / n2
    d_statistic = jnp.max(jnp.abs(cdf_cal - cdf_evl))
    n_eff = (n1 * n2) / (n1 + n2)
    sqrt_n = jnp.sqrt(n_eff)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_statistic
    k = jnp.arange(1, 101)
    series = 2.0 * jnp.sum(((-1.0) ** (k - 1)) * jnp.exp(-2.0 * (k * lam) ** 2))
    return jnp.clip(series, 0.0, 1.0)


def check_exchangeability(
    *,
    calibration_scores: jax.Array,
    evaluation_scores: jax.Array,
    alpha: float = 0.05,
) -> ExchangeabilityReport:
    """Run the KS exchangeability check and package the report.

    Args:
        calibration_scores: 1-D array of nonconformity scores on the
            calibration partition.
        evaluation_scores: 1-D array on the evaluation partition.
        alpha: Significance level. ``passes = p > alpha``.

    Returns:
        :class:`ExchangeabilityReport` with the p-value, pass flag, and
        metadata.

    """
    p_value = ks_two_sample_pvalue(
        calibration_scores=calibration_scores,
        evaluation_scores=evaluation_scores,
    )
    metadata: MetadataItems = (
        ("method", "ks_two_sample"),
        ("alpha", float(alpha)),
        ("calibration_size", int(calibration_scores.shape[0])),
        ("evaluation_size", int(evaluation_scores.shape[0])),
    )
    return ExchangeabilityReport(
        p_value=p_value,
        passes=bool(float(p_value) > alpha),
        metadata=metadata,
    )
