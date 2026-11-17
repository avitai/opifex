"""Backend-neutral likelihood log-density helpers.

Pure JAX functions. No ``flax.nnx`` imports. All inputs are explicit; no model
internals are inspected. Likelihood outputs compose with
:attr:`opifex.uncertainty.objectives.UQLossComponents.negative_log_likelihood`.

:class:`LikelihoodSpec` is a frozen+slotted hashable capability declaration.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from opifex.core.capabilities import DefaultStrategy, UQCapability


_GL8_NODES_NP, _GL8_WEIGHTS_NP = np.polynomial.legendre.leggauss(8)
_GL8_HALF_WEIGHTS_NP = 0.5 * _GL8_WEIGHTS_NP
_INTERVAL_QUADRATURE_CUTOFF: float = 0.5
_MILLS_CF_CUTOFF: float = -4.0
_MILLS_CF_DEPTH: int = 28


def _mills_ratio_lower_tail(u: jax.Array) -> jax.Array:
    """Evaluate ``phi(-u) / Phi(-u)`` for ``u >= 4`` via Laplace's continued fraction.

    All terms in ``cf = u + k / cf`` are strictly positive on ``u >= 4``, eliminating
    alternating-series cancellation and reaching full float32/float64 precision.
    """
    cf = u
    for k in range(_MILLS_CF_DEPTH, 0, -1):
        cf = u + jnp.asarray(float(k), dtype=u.dtype) / cf
    return cf


@jax.custom_jvp
def _accurate_log_ndtr(x: jax.Array) -> jax.Array:
    """Compute ``log Phi(x)`` to machine precision across ``float32`` and ``float64``.

    Avoids ``jax.scipy.special.log_ndtr``'s default ``series_order=3`` truncation error in
    the left tail (``~1.6e-10`` at ``x = -30`` in ``float64``) and ``-ndtr(-x)`` right-tail
    linearization by combining:

    * ``x > 0``: ``log1p(-ndtr(-x))``
    * ``-4 < x <= 0``: ``log(ndtr(x))``
    * ``x <= -4``: ``-0.5 * log(2*pi) - 0.5 * x^2 - log(r(x))`` where ``r(x) = phi(x)/Phi(x)``
      is evaluated via Laplace's positive-term continued fraction.
    """
    dtype = x.dtype
    is_pos = x > 0.0
    is_lower = x <= _MILLS_CF_CUTOFF

    x_pos = jnp.where(is_pos, x, 1.0)
    upper_val = jnp.log1p(-jax.scipy.special.ndtr(-x_pos))

    x_mid = jnp.where((~is_pos) & (~is_lower), x, -1.0)
    mid_val = jnp.log(jax.scipy.special.ndtr(x_mid))

    u_low = jnp.where(is_lower, -x, 4.0)
    mills = _mills_ratio_lower_tail(u_low)
    lower_val = (
        -0.5 * jnp.log(jnp.asarray(2.0 * np.pi, dtype=dtype))
        - 2.0 * jnp.square(0.5 * u_low)
        - jnp.log(mills)
    )
    return jnp.where(is_pos, upper_val, jnp.where(is_lower, lower_val, mid_val))


def _accurate_log_ndtr_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Cancellation-free inverse Mills ratio JVP ``d/dx log Phi(x) = phi(x) / Phi(x)``."""
    (x,) = primals
    (x_dot,) = tangents
    dtype = x.dtype
    primal = _accurate_log_ndtr(x)

    is_lower = x <= _MILLS_CF_CUTOFF
    max_ax = 16.0 if dtype == jnp.float32 else 40.0
    x_central = jnp.where((~is_lower) & (x < max_ax), x, 0.0)
    ndtr_central = jax.scipy.special.ndtr(jnp.where(~is_lower, x, 0.0))
    grad_central = jnp.where(
        x < max_ax,
        jnp.exp(-0.5 * jnp.square(x_central))
        / (jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=dtype)) * ndtr_central),
        0.0,
    )

    u_low = jnp.where(is_lower, -x, 4.0)
    grad_lower = _mills_ratio_lower_tail(u_low)
    grad = jnp.where(is_lower, grad_lower, grad_central)
    return primal, grad * x_dot


_accurate_log_ndtr.defjvp(_accurate_log_ndtr_jvp)


def _require_positive_scale(scale: jax.Array, *, name: str) -> None:
    """Eager runtime check for non-traced scale arrays.

    Validation is skipped inside ``jax.jit`` / ``jax.grad`` tracers — the
    kernels are pure JAX, and the caller should validate at the model
    boundary before entering a transform. Concrete (eager) arrays still
    receive the check so direct calls surface bad inputs with ``ValueError``.

    Tracer detection: even when ``scale`` itself is concrete (e.g., a closure
    capture), ``scale > 0`` produces a tracer inside a jit context. We
    therefore detect tracing via the ``TracerBoolConversionError`` raised by
    ``bool(...)`` and skip the check.
    """
    try:
        is_positive = bool(jnp.all(scale > 0))
    except jax.errors.TracerBoolConversionError:
        return
    if not is_positive:
        raise ValueError(f"{name} must be strictly positive; got {scale!r}.")


def gaussian_log_likelihood(
    y: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Log-likelihood per observation of ``y`` under ``N(mean, scale^2)``."""
    _require_positive_scale(scale, name="scale")
    var = scale * scale
    return -0.5 * (jnp.log(2 * jnp.pi) + jnp.log(var) + (y - mean) ** 2 / var)


def heteroscedastic_gaussian_log_likelihood(
    y: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Heteroscedastic Gaussian: per-observation ``scale`` of same shape as ``y``."""
    if scale.shape != y.shape:
        raise ValueError(f"heteroscedastic scale shape {scale.shape} must match y shape {y.shape}.")
    return gaussian_log_likelihood(y, mean=mean, scale=scale)


def laplace_log_likelihood(
    y: jax.Array,
    *,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Log-likelihood per observation of ``y`` under ``Laplace(location, scale)``."""
    _require_positive_scale(scale, name="scale")
    return -jnp.log(2 * scale) - jnp.abs(y - location) / scale


def student_t_log_likelihood(
    y: jax.Array,
    *,
    df: float,
    location: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Log-likelihood per observation of ``y`` under ``StudentT(df, location, scale)``."""
    if df <= 0.0:
        raise ValueError(f"df must be strictly positive; got {df!r}.")
    _require_positive_scale(scale, name="scale")
    half_df = 0.5 * df
    log_gamma_term = jax.scipy.special.gammaln(half_df + 0.5) - jax.scipy.special.gammaln(half_df)
    log_norm = log_gamma_term - 0.5 * jnp.log(df * jnp.pi) - jnp.log(scale)
    z = (y - location) / scale
    return log_norm - (half_df + 0.5) * jnp.log1p(z * z / df)


def mixture_log_likelihood(
    y: jax.Array,
    *,
    weights: jax.Array,
    means: jax.Array,
    scales: jax.Array,
) -> jax.Array:
    """Log-likelihood of ``y`` under a 1D Gaussian mixture ``sum_k w_k N(mu_k, s_k^2)``.

    ``weights`` must sum to 1; ``means`` and ``scales`` have the same length.
    """
    try:
        sums_to_one = bool(jnp.isclose(jnp.sum(weights), 1.0))
        non_negative = bool(jnp.all(weights >= 0))
    except jax.errors.TracerBoolConversionError:
        sums_to_one = True
        non_negative = True
    if not sums_to_one:
        raise ValueError(f"Mixture weights must sum to 1; got sum={float(jnp.sum(weights))!r}.")
    if not non_negative:
        raise ValueError("Mixture weights must be non-negative.")
    _require_positive_scale(scales, name="scales")
    if weights.shape != means.shape or weights.shape != scales.shape:
        raise ValueError(
            "weights, means, and scales must share the same shape; got "
            f"{weights.shape}, {means.shape}, {scales.shape}."
        )

    # log sum_k w_k * N(y; mu_k, s_k^2) via logsumexp.
    y_expanded = y[..., None]
    components = gaussian_log_likelihood(y_expanded, mean=means, scale=scales) + jnp.log(weights)
    return jax.scipy.special.logsumexp(components, axis=-1)


def _require_valid_censored_bounds(
    lower: jax.Array,
    upper: jax.Array,
    *,
    exact_mask: jax.Array | None,
) -> None:
    """Validate censoring interval bounds eagerly outside JAX tracing."""
    if exact_mask is not None and jnp.asarray(exact_mask).dtype != jnp.bool_:
        raise ValueError(
            f"exact_mask must have boolean dtype; got {jnp.asarray(exact_mask).dtype}."
        )
    try:
        has_posinf_lower = bool(jnp.any(jnp.isposinf(lower)))
        has_neginf_upper = bool(jnp.any(jnp.isneginf(upper)))
        has_inverted = bool(jnp.any(lower > upper))
    except jax.errors.TracerBoolConversionError:
        return
    if has_posinf_lower:
        raise ValueError("lower censoring bound cannot be +inf.")
    if has_neginf_upper:
        raise ValueError("upper censoring bound cannot be -inf.")
    if has_inverted:
        raise ValueError("Censored bounds must satisfy lower <= upper elementwise.")


def _interval_gaussian_log_prob(
    lo_safe: jax.Array,
    hi_safe: jax.Array,
    z_lo: jax.Array,
    z_hi: jax.Array,
    scale: jax.Array,
    is_interval: jax.Array,
) -> jax.Array:
    """Compute ``log(Phi(z_hi) - Phi(z_lo))`` without tail or narrow-width cancellation.

    Splits ``is_interval`` across ``w * (1 + |z_mid|) <= 0.5``:

    * **Narrow / tail-narrow regime:** pulls ``-0.5 * z_mid^2`` and ``log(hi - lo) - log(scale)``
      outside the logarithm and evaluates the bounded factor
      ``0.5 * int_{-1}^1 exp(-0.5 * w * z_mid * t - 0.125 * w^2 * t^2) dt`` via 8-point
      Gauss-Legendre quadrature using ``log1p(sum(half_w * expm1(...)))``.
    * **Separated regime:** flips to whichever Gaussian tail is smaller around ``z_mid == 0``
      and evaluates ``log_major + jax.nn.log1mexp(log_major - log_minor)`` on masked inputs.
    """
    dtype = scale.dtype
    raw_width = jnp.where(is_interval, jnp.maximum(hi_safe - lo_safe, 0.0), scale)
    safe_raw_width = jnp.maximum(raw_width, jnp.finfo(dtype).tiny)
    w_std = jnp.where(is_interval, raw_width / scale, 1.0)
    z_mid = 0.5 * (z_lo + z_hi)

    use_quad = is_interval & (w_std * (1.0 + jnp.abs(z_mid)) <= _INTERVAL_QUADRATURE_CUTOFF)
    w_q = jnp.where(use_quad, w_std, 0.0)
    m_q = jnp.where(use_quad, z_mid, 0.0)
    nodes = jnp.asarray(_GL8_NODES_NP, dtype=dtype)
    half_weights = jnp.asarray(_GL8_HALF_WEIGHTS_NP, dtype=dtype)
    exponents = -0.5 * (w_q[..., None] * m_q[..., None]) * nodes - 0.125 * (w_q[..., None] ** 2) * (
        nodes * nodes
    )
    log_integral = jnp.log1p(jnp.sum(half_weights * jnp.expm1(exponents), axis=-1))
    quad_ll = (
        -0.5 * jnp.log(jnp.asarray(2.0 * np.pi, dtype=dtype))
        - 0.5 * z_mid * z_mid
        + jnp.log(safe_raw_width)
        - jnp.log(scale)
        + log_integral
    )

    use_tail_diff = is_interval & (~use_quad)
    z_lo_td = jnp.where(use_tail_diff, z_lo, -0.5)
    z_hi_td = jnp.where(use_tail_diff, z_hi, 0.5)
    use_upper_tail = (z_lo_td + z_hi_td) > 0.0
    arg_major = jnp.where(use_upper_tail, -z_lo_td, z_hi_td)
    arg_minor = jnp.where(use_upper_tail, -z_hi_td, z_lo_td)
    log_major = _accurate_log_ndtr(arg_major)
    log_minor = _accurate_log_ndtr(arg_minor)
    delta = log_major - log_minor
    log_two = jnp.log(jnp.asarray(2.0, dtype=dtype))
    delta_safe = jnp.where(use_tail_diff & (delta > 0.0), delta, log_two)
    tail_diff_ll = log_major + jax.nn.log1mexp(delta_safe)

    return jnp.where(use_quad, quad_ll, tail_diff_ll)


def _censored_branch_quantities(
    lower: jax.Array,
    upper: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
    exact_mask: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(is_exact, exact_log_density, censored_interval_log_prob)`` on sanitized inputs."""
    _require_positive_scale(scale, name="scale")
    _require_valid_censored_bounds(lower, upper, exact_mask=exact_mask)

    dtype = jnp.result_type(lower, upper, mean, scale, jnp.float32)
    lower_arr, upper_arr, mean_arr, scale_arr = jnp.broadcast_arrays(
        jnp.asarray(lower, dtype=dtype),
        jnp.asarray(upper, dtype=dtype),
        jnp.asarray(mean, dtype=dtype),
        jnp.asarray(scale, dtype=dtype),
    )

    lo_finite = jnp.isfinite(lower_arr)
    hi_finite = jnp.isfinite(upper_arr)
    if exact_mask is None:
        is_exact = lo_finite & hi_finite & (lower_arr == upper_arr)
    else:
        is_exact = jnp.broadcast_to(jnp.asarray(exact_mask, dtype=jnp.bool_), mean_arr.shape)

    # Sanitize +/-inf bounds BEFORE subtraction/division so unselected jnp.where arms
    # never form inf - inf, 0 * inf, or extreme z during reverse-mode differentiation.
    lo_safe = jnp.where(lo_finite, lower_arr, mean_arr)
    hi_safe = jnp.where(hi_finite, upper_arr, mean_arr)
    z_lo = (lo_safe - mean_arr) / scale_arr
    z_hi = (hi_safe - mean_arr) / scale_arr

    exact_ll = gaussian_log_likelihood(lo_safe, mean=mean_arr, scale=scale_arr)

    is_left = (~is_exact) & jnp.isneginf(lower_arr) & hi_finite
    is_right = (~is_exact) & lo_finite & jnp.isposinf(upper_arr)
    is_uninformative = (~is_exact) & jnp.isneginf(lower_arr) & jnp.isposinf(upper_arr)
    is_interval = (~is_exact) & lo_finite & hi_finite & (upper_arr >= lower_arr)

    left_ll = _accurate_log_ndtr(jnp.where(is_left, z_hi, 0.0))
    right_ll = _accurate_log_ndtr(jnp.where(is_right, -z_lo, 0.0))
    interval_ll = _interval_gaussian_log_prob(lo_safe, hi_safe, z_lo, z_hi, scale_arr, is_interval)
    cens_ll = jnp.where(
        is_left,
        left_ll,
        jnp.where(
            is_right,
            right_ll,
            jnp.where(
                is_interval,
                interval_ll,
                jnp.where(is_uninformative, 0.0, -jnp.inf),
            ),
        ),
    )
    return is_exact, exact_ll, cens_ll


def censored_gaussian_log_likelihood(
    lower: jax.Array,
    upper: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
    exact_mask: jax.Array | None = None,
) -> jax.Array:
    """Four-branch Gaussian log-likelihood over exact, one-sided, and interval-censored data.

    Evaluates exact Gaussian log-density on point observations and exact log-probability mass
    ``log P(lower <= Y <= upper)`` under ``Y ~ N(mean, scale^2)`` on left-censored
    ``(-inf, upper]``, right-censored ``[lower, +inf)``, two-sided ``[lower, upper]``, and
    uninformative ``(-inf, +inf)`` rows, with masked ``jnp.where`` inputs so values and
    reverse-mode gradients remain finite and cancellation-free into deep tails.

    Args:
        lower: Lower observation bound array (``-inf`` for left-censored rows).
        upper: Upper observation bound array (``+inf`` for right-censored rows).
        mean: Predictive Gaussian mean array broadcastable with ``lower`` and ``upper``.
        scale: Strictly positive Gaussian standard deviation array.
        exact_mask: Optional boolean mask identifying point observations. Defaults to strict
            equality ``isfinite(lower) & isfinite(upper) & (lower == upper)``.

    Returns:
        Per-observation log-likelihood array of the broadcast shape. Eagerly validates
        ``scale > 0``, ``lower <= upper``, finite-or-valid infinite endpoints, and boolean
        ``exact_mask`` when called outside ``jax.jit``.
    """
    is_exact, exact_ll, cens_ll = _censored_branch_quantities(
        lower, upper, mean=mean, scale=scale, exact_mask=exact_mask
    )
    return jnp.where(is_exact, exact_ll, cens_ll)


def tobit_nll(
    lower: jax.Array,
    upper: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
    exact_mask: jax.Array | None = None,
) -> jax.Array:
    """Interval-censored Tobit Gaussian negative log-likelihood per observation.

    Reference: Tobin (1958), *Estimation of Relationships for Limited Dependent Variables*,
    Econometrica 26(1):24-36.

    Args:
        lower: Lower observation bound array (``-inf`` for left-censored rows).
        upper: Upper observation bound array (``+inf`` for right-censored rows).
        mean: Predictive Gaussian mean array.
        scale: Strictly positive Gaussian standard deviation array.
        exact_mask: Optional boolean mask identifying point observations.

    Returns:
        Per-observation negative log-likelihood ``-censored_gaussian_log_likelihood(...)``.
    """
    return -censored_gaussian_log_likelihood(
        lower, upper, mean=mean, scale=scale, exact_mask=exact_mask
    )


def satisficing_censored_deficit(
    lower: jax.Array,
    upper: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
    tau: float = 0.9,
    exact_mask: jax.Array | None = None,
) -> jax.Array:
    """Squared probability-satisficing deficit ``max(tau - P(lower <= Y <= upper), 0)^2``.

    Once the model assigns probability at least ``tau`` to a censored interval, both the
    deficit and its gradient vanish so censored rows stop incentivizing unbounded mean
    extrapolation. Exact rows return ``0.0``.

    Reference: Mitropoulou et al. (2025), *CensorADMET*.

    Args:
        lower: Lower observation bound array.
        upper: Upper observation bound array.
        mean: Predictive Gaussian mean array.
        scale: Strictly positive Gaussian standard deviation array.
        tau: Target interval satisfaction probability in ``(0, 1]``.
        exact_mask: Optional boolean mask identifying point observations.

    Returns:
        Per-observation non-negative squared probability deficit (``0.0`` on exact rows).

    Raises:
        ValueError: When ``tau`` is outside ``(0, 1]`` or bounds/scales are invalid.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must lie in (0, 1]; got {tau!r}.")
    is_exact, _, cens_ll = _censored_branch_quantities(
        lower, upper, mean=mean, scale=scale, exact_mask=exact_mask
    )
    safe_cens_ll = jnp.where(is_exact, 0.0, jnp.minimum(cens_ll, 0.0))
    prob = jnp.exp(safe_cens_ll)
    active = (~is_exact) & (prob < tau)
    margin = jnp.where(active, tau - prob, 0.0)
    return margin * margin


def satisficing_censored_loss(
    lower: jax.Array,
    upper: jax.Array,
    *,
    mean: jax.Array,
    scale: jax.Array,
    tau: float = 0.9,
    censored_weight: float = 1.0,
    exact_mask: jax.Array | None = None,
) -> jax.Array:
    """Combined exact Gaussian NLL and weighted probability-satisficing censored loss.

    Evaluates ``-gaussian_log_likelihood`` on exact rows and
    ``censored_weight * max(tau - P(lower <= Y <= upper), 0)^2`` on censored rows.

    Reference: Mitropoulou et al. (2025), *CensorADMET*.

    Args:
        lower: Lower observation bound array.
        upper: Upper observation bound array.
        mean: Predictive Gaussian mean array.
        scale: Strictly positive Gaussian standard deviation array.
        tau: Target interval satisfaction probability in ``(0, 1]``.
        censored_weight: Non-negative multiplier on the censored satisficing deficit.
        exact_mask: Optional boolean mask identifying point observations.

    Returns:
        Per-observation loss array.

    Raises:
        ValueError: When ``tau`` is outside ``(0, 1]``, ``censored_weight < 0``, or
            bounds/scales are invalid.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must lie in (0, 1]; got {tau!r}.")
    if censored_weight < 0.0:
        raise ValueError(f"censored_weight must be non-negative; got {censored_weight!r}.")
    is_exact, exact_ll, cens_ll = _censored_branch_quantities(
        lower, upper, mean=mean, scale=scale, exact_mask=exact_mask
    )
    safe_cens_ll = jnp.where(is_exact, 0.0, jnp.minimum(cens_ll, 0.0))
    prob = jnp.exp(safe_cens_ll)
    active = (~is_exact) & (prob < tau)
    margin = jnp.where(active, tau - prob, 0.0)
    return jnp.where(is_exact, -exact_ll, censored_weight * (margin * margin))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LikelihoodSpec:
    """Capability declaration for a likelihood family.

    Frozen, slotted, hashable; sequence fields are tuples.
    """

    name: str
    family: str
    parameter_names: tuple[str, ...]
    supports_heteroscedastic: bool = False

    def __post_init__(self) -> None:
        """Validate ``parameter_names`` shape and a non-empty ``name``."""
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple.")
        if not self.name:
            raise ValueError("LikelihoodSpec.name must be non-empty.")


LIKELIHOOD_CAPABILITIES: dict[str, UQCapability] = {
    "likelihood:censored_gaussian": UQCapability(
        native_distributional=True,
        native_jax_kernel=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="opifex",
        notes=(
            "Four-branch exact/left/right/interval censored Gaussian log-likelihood and "
            "Tobit NLL (Tobin 1958) with Gauss-Legendre narrow-interval quadrature and "
            "log_ndtr + log1mexp tail symmetry."
        ),
    ),
    "likelihood:satisficing_censored": UQCapability(
        native_distributional=True,
        native_jax_kernel=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="opifex",
        notes=(
            "Probability-satisficing censored Gaussian deficit and combined exact-NLL + "
            "deficit loss (Mitropoulou et al. 2025, CensorADMET)."
        ),
    ),
}


__all__ = [
    "LIKELIHOOD_CAPABILITIES",
    "LikelihoodSpec",
    "censored_gaussian_log_likelihood",
    "gaussian_log_likelihood",
    "heteroscedastic_gaussian_log_likelihood",
    "laplace_log_likelihood",
    "mixture_log_likelihood",
    "satisficing_censored_deficit",
    "satisficing_censored_loss",
    "student_t_log_likelihood",
    "tobit_nll",
]
