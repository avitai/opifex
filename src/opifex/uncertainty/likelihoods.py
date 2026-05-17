"""Phase 1 Task 1.6 — backend-neutral likelihood log-density helpers.

Pure JAX functions. No ``flax.nnx`` imports. All inputs are explicit; no model
internals are inspected. Likelihood outputs compose with
:attr:`opifex.uncertainty.objectives.UQLossComponents.negative_log_likelihood`.

Container pattern: :class:`LikelihoodSpec` is pattern (A) per GUIDE_ALIGNMENT §5a.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LikelihoodSpec:
    """Pattern (A) capability declaration for a likelihood family.

    Sequence fields are tuples (GUIDE_ALIGNMENT item 22a).
    """

    name: str
    family: str
    parameter_names: tuple[str, ...]
    supports_heteroscedastic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple (GUIDE_ALIGNMENT item 22a).")
        if not self.name:
            raise ValueError("LikelihoodSpec.name must be non-empty.")


__all__ = [
    "LikelihoodSpec",
    "gaussian_log_likelihood",
    "heteroscedastic_gaussian_log_likelihood",
    "laplace_log_likelihood",
    "mixture_log_likelihood",
    "student_t_log_likelihood",
]
