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

from typing import Final

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.types import MetadataItems  # noqa: TC001


@struct.dataclass(slots=True, kw_only=True)
class ExchangeabilityReport:
    """Outcome of a two-sample exchangeability check."""

    p_value: jax.Array
    passes: jax.Array
    method: str = struct.field(pytree_node=False, default="ks_two_sample")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def ks_two_sample_pvalue(
    *,
    calibration_scores: jax.Array,
    evaluation_scores: jax.Array,
) -> jax.Array:
    """Two-sample Kolmogorov-Smirnov p-value (asymptotic).

    Pure JAX; traces under ``jax.jit`` and ``jax.vmap``.

    Two factors scale the statistic, on very different evidence.

    ``n_eff = n1 n2 / (n1 + n2)`` is the asymptotic two-sample normalisation, and it is
    old and solid: Feller 1948 eqs. (1.6) to (1.8), and Smirnov 1948.

    The ``+ 0.12 + 0.11 / sqrt(n_eff)`` correction is on weaker ground. Stephens 1970
    Table 1 gives it for the **one-sample** statistic; Stephens does not treat two samples,
    and applying it with ``n_eff`` substituted has no source. Hodges 1958 eq. (5.3) gives
    the two-sample correction at this order as a function of
    ``(m + 2n) / sqrt(mn(m + n))``, which sample pairs of equal ``n_eff`` do not share, so
    no function of ``n_eff`` alone expresses it. Feller's condition (1.7), ``m / n -> a``,
    bounds the same thing: the limit law is ratio-free, the finite-sample correction is
    not. The factor reduces bias without being a calibrated quantity, and it was fitted in
    the upper tail.

    ``scipy.stats.ks_2samp`` deliberately does something else -- it evaluates the
    finite-``n`` distribution at ``round(n_eff)`` -- so a few percent of disagreement with
    it is by design. ``scipy.special.kolmogorov`` is the like-for-like comparison.

    Args:
        calibration_scores: 1-D array of scores on the calibration partition.
        evaluation_scores: 1-D array of scores on the evaluation partition.

    Returns:
        The asymptotic p-value, in ``[0, 1]``.

    References:
        * Kolmogorov 1933 -- *Sulla determinazione empirica di una legge di
          distribuzione*, G. Ist. Ital. Attuari 4, 83. The alternating series.
        * Feller 1948 -- *On the Kolmogorov-Smirnov limit theorems for empirical
          distributions*, Ann. Math. Statist. 19(2), 177. Eq. (1.4), p. 178 gives both
          series in one display, in this normalisation, and footnote 2 on the same page
          says of the theta form that "the second is more useful for small z", which
          is the branch rule below. The erratum belongs with the citation: the exponent of
          the *alternating* side is printed as ``-v^2 z^2`` and should be ``-2 v^2 z^2``
          (Ann. Math. Statist. 21(2), 301, 1950). The theta side is printed correctly.
        * Smirnov 1948 -- *Table for estimating the goodness of fit of empirical
          distributions*, Ann. Math. Statist. 19, 279. With Feller eqs. (1.6) to (1.8),
          the source for the two-sample normalisation.
        * van Mulbregt 2018 -- *Computing the cumulative distribution function and
          quantiles of the limit of the two-sided Kolmogorov-Smirnov statistic*,
          arXiv:1803.00426. Secs. 3.1, 3.3 and 4.1: the algorithm, the term bounds and
          the reason the cutover sits at the median.
        * Stephens 1970 -- *Use of the Kolmogorov-Smirnov, Cramer-von Mises and related
          statistics without extensive tables*, J. R. Stat. Soc. B 32, 115. Table 1, the
          one-sample modifier. (Table 2 of the same paper contains 0.82 for an unrelated
          quantity; the coincidence with the series cutover below is not a connection.)
        * Press et al. 1992 -- *Numerical Recipes in C*, 2nd ed., sec. 14.3, which applies
          the modifier to the two-sample case.
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
    return _kolmogorov_tail(lam)


# van Mulbregt 2018, sec. 4.1: the cutover sits at approximately the median of the
# distribution, so whichever branch runs computes the smaller of the tail and the body,
# and forming the complement never costs a cancellation. This is the value cephes uses as
# KOLMOG_CUTOVER, so `scipy.special.kolmogorov` switches at the same place.
_SERIES_CUTOVER: Final[float] = 0.82

# Terms per branch. van Mulbregt secs. 3.1 and 3.3 bound the requirement at the cutover by
# k > 4.29 / lambda for the alternating series and (2k - 1) > 5.46 * lambda for the theta
# series, so five and two suffice for double precision; these carry margin.
_ALTERNATING_TERMS: Final[int] = 6
_THETA_TERMS: Final[int] = 3

# Floor for the reciprocal in the theta branch. Both branches of a `where` are evaluated,
# so an unguarded 1 / lambda makes inf * 0 at the origin and poisons the gradient of the
# branch not taken with a NaN.
_SMALLEST_ARGUMENT: Final[float] = 1e-8


def _kolmogorov_tail(lam: jax.Array) -> jax.Array:
    """``Q(lam)``, the complementary CDF of the Kolmogorov limit distribution.

    Two series express the same function, each accurate where the other is not, and
    choosing between them is the whole of the implementation.

    The alternating form of Kolmogorov 1933,
    ``2 sum (-1)^(k-1) exp(-2 k^2 lam^2)``, has terms that do not decay until
    ``k > 4.29 / lam``, so it needs unboundedly many of them as ``lam`` falls: truncated,
    it returns zero at ``lam = 1e-8`` where the answer is one.

    The theta form, ``(sqrt(2 pi) / lam) sum exp(-((2k-1) pi)^2 / (8 lam^2))``, follows
    from Jacobi's functional equation for the theta function and converges faster the
    smaller the argument. It gives the body of the distribution, so the tail is its
    complement -- and that subtraction loses the answer once the tail is small.

    Splitting at the median is what makes both safe: each branch computes the smaller of
    the two quantities, so the complement is never formed from a number near one.

    Args:
        lam: The scaled Kolmogorov-Smirnov statistic, non-negative.

    Returns:
        The tail probability, in ``[0, 1]``.
    """
    alternating_k = jnp.arange(1, _ALTERNATING_TERMS + 1)
    tail = 2.0 * jnp.sum(
        ((-1.0) ** (alternating_k - 1)) * jnp.exp(-2.0 * (alternating_k * lam) ** 2)
    )

    guarded = jnp.maximum(lam, _SMALLEST_ARGUMENT)
    theta_k = jnp.arange(1, _THETA_TERMS + 1)
    body = (
        jnp.sqrt(2.0 * jnp.pi)
        / guarded
        * jnp.sum(jnp.exp(-(((2 * theta_k - 1) * jnp.pi) ** 2) / (8.0 * guarded**2)))
    )

    return jnp.clip(jnp.where(lam < _SERIES_CUTOVER, 1.0 - body, tail), 0.0, 1.0)


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
        passes=p_value > alpha,
        metadata=metadata,
    )
