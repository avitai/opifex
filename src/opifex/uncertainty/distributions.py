"""Distribution adapter layer for predictive-distribution interoperability.

Wraps a backend distribution into an Opifex
:class:`opifex.uncertainty.types.PredictiveDistribution`.

Adapter resolution order (matches GUIDE_ALIGNMENT
``DistributionAdapterSpec.resolution_order``):

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


__all__ = [
    "ArtifexDistributionAdapter",
    "DistrAxAdapter",
    "from_distribution",
]
