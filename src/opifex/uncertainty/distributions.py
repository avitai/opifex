"""Distribution adapter layer for predictive-distribution interoperability.

Wraps a backend distribution into an Opifex
:class:`opifex.uncertainty.types.PredictiveDistribution`.

Adapter resolution order (mirrors ``DistributionAdapterSpec.resolution_order``):

1. :class:`artifex.generative_models.core.distributions.base.Distribution`
   (primary target).
2. Distrax-like objects exposing ``sample`` / ``log_prob`` / ``mean`` /
   ``variance``.

Future expansion (TFP-substrate / FlowJAX / bijx / GPJax / NumPyro) plugs
into the same protocol; their adapter classes follow the same shape and
live alongside :mod:`opifex.uncertainty.inference_backends.optional`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.distributions.base import Distribution

from opifex.uncertainty.types import PredictiveDistribution


class ArtifexDistributionAdapter:
    """Wrap an Artifex ``Distribution`` into a :class:`PredictiveDistribution`.

    Artifex's :class:`Distribution` base class delegates moment accessors to
    its internal wrapped distribution (e.g. ``distrax.Normal``) via
    ``_distribution()``. The adapter uses ``mean()`` / ``variance()`` on the
    wrapped distribution directly so subclasses that override these methods
    on the Artifex side (e.g. :class:`Beta`) and subclasses that delegate
    (e.g. :class:`Normal`) both work.
    """

    def from_distribution(self, distribution: Any) -> PredictiveDistribution:
        """Return a predictive distribution carrying the wrapped mean and variance."""
        mean = jnp.asarray(_artifex_mean(distribution))
        variance = _artifex_variance(distribution)
        return PredictiveDistribution(
            mean=mean,
            variance=jnp.asarray(variance) if variance is not None else None,
            metadata=(("source_package", "artifex"),),
        )


class DistrAxAdapter:
    """Wrap a Distrax-like distribution into a :class:`PredictiveDistribution`.

    Accepts any object exposing ``mean()``, ``variance()``, ``sample(...)``,
    and ``log_prob(...)`` — the de-facto Distrax / TFP-on-JAX shape.
    """

    def from_distribution(self, distribution: Any) -> PredictiveDistribution:
        """Return a predictive distribution carrying the wrapped mean and variance."""
        mean_fn = distribution.mean
        var_fn = distribution.variance
        mean = jnp.asarray(mean_fn() if callable(mean_fn) else mean_fn)
        variance = jnp.asarray(var_fn() if callable(var_fn) else var_fn)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            metadata=(("source_package", "distrax"),),
        )


def from_distribution(distribution: Any) -> PredictiveDistribution:
    """Adapter dispatch: Artifex first, Distrax fallback.

    Raises:
        TypeError: When the supplied object is not recognized as either an
            Artifex ``Distribution`` or a Distrax-like distribution.

    """
    if isinstance(distribution, Distribution):
        return ArtifexDistributionAdapter().from_distribution(distribution)
    required = ("sample", "log_prob", "mean", "variance")
    if all(hasattr(distribution, attr) for attr in required):
        return DistrAxAdapter().from_distribution(distribution)
    raise TypeError(
        f"Unsupported distribution object {type(distribution).__name__!r}. "
        "Expected an Artifex Distribution subclass or a Distrax-like object "
        f"exposing {required!r}."
    )


def _artifex_mean(distribution: Distribution) -> jax.Array:
    """Return the mean of an Artifex distribution.

    Subclasses may either override ``mean()`` directly (e.g. ``Beta``) or
    delegate to an internal wrapped distribution (e.g. ``Normal`` wraps
    ``distrax.Normal``). Both routes are tried in turn.
    """
    mean_fn = getattr(distribution, "mean", None)
    if callable(mean_fn):
        return jnp.asarray(mean_fn())
    wrapped = distribution._distribution()
    return jnp.asarray(wrapped.mean())


def _artifex_variance(distribution: Distribution) -> jax.Array | None:
    """Return the variance of an Artifex distribution, or ``None`` if unavailable."""
    variance_fn = getattr(distribution, "variance", None)
    try:
        if callable(variance_fn):
            return jnp.asarray(variance_fn())
        wrapped = distribution._distribution()
        return jnp.asarray(wrapped.variance())
    except (AttributeError, NotImplementedError):
        return None


def _validate_pairwise_covariance_eagerly(
    var_a: jax.Array,
    var_b: jax.Array,
    cov_ab: jax.Array,
) -> None:
    """Validate non-negative variances and Cauchy-Schwarz inequality outside ``jit``."""
    dtype = jnp.result_type(var_a, var_b, cov_ab, jnp.float32)
    eps = float(jnp.finfo(dtype).eps)
    try:
        neg_a = bool(jnp.any(var_a < 0))
        neg_b = bool(jnp.any(var_b < 0))
    except jax.errors.TracerBoolConversionError:
        return
    if neg_a:
        raise ValueError(f"var_a must be non-negative; got {var_a!r}.")
    if neg_b:
        raise ValueError(f"var_b must be non-negative; got {var_b!r}.")

    bound = jnp.sqrt(var_a * var_b) + 16.0 * eps * jnp.maximum(var_a, var_b)
    if bool(jnp.any(jnp.abs(cov_ab) > bound)):
        raise ValueError(
            "cov_ab violates the Cauchy-Schwarz inequality |cov_ab| <= sqrt(var_a * var_b)."
        )


def correlated_difference_predictive(
    mean_a: jax.Array,
    mean_b: jax.Array,
    *,
    var_a: jax.Array,
    var_b: jax.Array,
    cov_ab: jax.Array | float = 0.0,
) -> PredictiveDistribution:
    """Predictive distribution of ``Y_a - Y_b`` accounting for cross-covariance ``cov_ab``.

    Computes ``mean = mean_a - mean_b`` and
    ``variance = max(var_a + var_b - 2 * cov_ab, 0)``, validating non-negativity and the
    Cauchy-Schwarz PSD bound eagerly when called outside ``jax.jit``.

    Args:
        mean_a: Predictive mean of target ``a``.
        mean_b: Predictive mean of target ``b`` broadcastable with ``mean_a``.
        var_a: Marginal predictive variance of target ``a``.
        var_b: Marginal predictive variance of target ``b``.
        cov_ab: Predictive cross-covariance ``Cov(Y_a, Y_b)`` (defaults to ``0.0``).

    Returns:
        :class:`PredictiveDistribution` carrying ``mean``, ``variance``, and
        ``total_uncertainty``. Non-negative variances and the Cauchy-Schwarz inequality
        ``|cov_ab| <= sqrt(var_a * var_b)`` are validated eagerly outside ``jax.jit``.
    """
    dtype = jnp.result_type(mean_a, mean_b, var_a, var_b, cov_ab, jnp.float32)
    ma, mb, va, vb, cab = jnp.broadcast_arrays(
        jnp.asarray(mean_a, dtype=dtype),
        jnp.asarray(mean_b, dtype=dtype),
        jnp.asarray(var_a, dtype=dtype),
        jnp.asarray(var_b, dtype=dtype),
        jnp.asarray(cov_ab, dtype=dtype),
    )
    _validate_pairwise_covariance_eagerly(va, vb, cab)
    diff_mean = ma - mb
    diff_var = jnp.maximum(va + vb - 2.0 * cab, 0.0)
    return PredictiveDistribution(
        mean=diff_mean,
        variance=diff_var,
        total_uncertainty=diff_var,
        metadata=(("operation", "correlated_difference"),),
    )


def _validate_joint_covariance_eagerly(covariance: jax.Array) -> None:
    """Validate symmetry and positive semi-definiteness outside ``jit``."""
    eps = float(jnp.finfo(covariance.dtype).eps)
    try:
        is_sym = bool(
            jnp.allclose(covariance, jnp.swapaxes(covariance, -1, -2), rtol=1e-5, atol=1e-6)
        )
    except jax.errors.TracerBoolConversionError:
        return
    if not is_sym:
        raise ValueError("covariance must be symmetric along its trailing two axes.")
    sym_cov = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    eigvals = jnp.linalg.eigvalsh(sym_cov)
    diag_scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.diagonal(sym_cov, axis1=-2, axis2=-1))),
        jnp.asarray(1.0, dtype=covariance.dtype),
    )
    if bool(jnp.min(eigvals) < -16.0 * eps * diag_scale):
        raise ValueError("covariance must be positive semi-definite.")


def linear_contrast_predictive(
    mean: jax.Array,
    covariance: jax.Array,
    contrast: jax.Array,
) -> PredictiveDistribution:
    """Predictive distribution of a linear contrast ``c^T Y`` over joint Gaussian moments.

    Args:
        mean: Joint predictive mean array of shape ``(..., M)``.
        covariance: Joint predictive covariance array of shape ``(..., M, M)``.
        contrast: Linear contrast weights of shape ``(M,)`` or broadcastable to ``mean.shape``.

    Returns:
        :class:`PredictiveDistribution` with ``mean = c^T mean`` and
        ``variance = max(c^T covariance c, 0)``.

    Raises:
        ValueError: On incompatible array shapes, or eagerly when ``covariance`` is
            asymmetric or indefinite.
    """
    dtype = jnp.result_type(mean, covariance, contrast, jnp.float32)
    mean_arr = jnp.asarray(mean, dtype=dtype)
    cov_arr = jnp.asarray(covariance, dtype=dtype)
    contrast_arr = jnp.asarray(contrast, dtype=dtype)

    if mean_arr.ndim < 1:
        raise ValueError(f"mean must have at least 1 axis (..., M); got shape {mean_arr.shape}.")
    num_targets = mean_arr.shape[-1]
    expected_cov_shape = (*mean_arr.shape, num_targets)
    if cov_arr.shape != expected_cov_shape:
        raise ValueError(
            f"covariance shape {cov_arr.shape} must match (*mean.shape, M) = {expected_cov_shape}."
        )
    if contrast_arr.ndim < 1 or contrast_arr.shape[-1] != num_targets:
        raise ValueError(
            f"contrast trailing dimension {contrast_arr.shape} must match M={num_targets}."
        )
    contrast_b = jnp.broadcast_to(contrast_arr, mean_arr.shape)
    _validate_joint_covariance_eagerly(cov_arr)

    contrast_mean = jnp.einsum("...m,...m->...", contrast_b, mean_arr)
    raw_var = jnp.einsum("...m,...mn,...n->...", contrast_b, cov_arr, contrast_b)
    contrast_var = jnp.maximum(raw_var, 0.0)
    return PredictiveDistribution(
        mean=contrast_mean,
        variance=contrast_var,
        total_uncertainty=contrast_var,
        metadata=(("operation", "linear_contrast"),),
    )


__all__ = [
    "ArtifexDistributionAdapter",
    "DistrAxAdapter",
    "correlated_difference_predictive",
    "from_distribution",
    "linear_contrast_predictive",
]
