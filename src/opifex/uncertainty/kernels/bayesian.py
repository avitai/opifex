"""Pure JAX Bayesian-kernel helpers.

Two binding rules:

* :func:`diagonal_gaussian_kl` MUST delegate to Artifex
  ``gaussian_kl_divergence`` whenever the prior is N(0, 1). The closed-form
  Gaussian-prior KL is implemented exactly once in the Avitai ecosystem
  (in Artifex); Opifex's helper is a thin wrapper that extends to parametric
  ``(prior_mean, prior_std)`` priors only.
* No ``flax.nnx`` imports in this module — pure JAX only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.divergence import (
    gaussian_kl_divergence as artifex_gaussian_kl_divergence,
)


def _is_standard_normal_prior(prior_mean: float, prior_std: float) -> bool:
    return prior_mean == 0.0 and prior_std == 1.0


def diagonal_gaussian_kl(
    mean: jax.Array,
    logvar: jax.Array,
    *,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
) -> jax.Array:
    """KL(N(mean, exp(logvar)) || N(prior_mean, prior_std^2)) summed over features.

    For the canonical ``N(0, 1)`` prior, this delegates to
    :func:`artifex.generative_models.core.losses.divergence.gaussian_kl_divergence`
    with ``reduction='sum'`` so the returned scalar is the total KL across all
    parameter dimensions (the form Bayesian-NN ELBO objectives expect).

    For a parametric prior ``(prior_mean, prior_std)`` the helper applies the
    closed-form location/scale correction directly.

    Args:
        mean: Posterior mean, any shape.
        logvar: Posterior log-variance, same shape as ``mean``.
        prior_mean: Scalar mean of the diagonal Gaussian prior.
        prior_std: Scalar standard deviation of the diagonal Gaussian prior.

    Returns:
        Scalar KL divergence (summed over every feature dimension).

    Raises:
        ValueError: If ``prior_std`` is not strictly positive.
    """
    if prior_std <= 0.0:
        raise ValueError(f"prior_std must be > 0, got {prior_std}.")

    if _is_standard_normal_prior(prior_mean, prior_std):
        return artifex_gaussian_kl_divergence(mean, logvar, reduction="sum")

    var = jnp.exp(logvar)
    prior_var = prior_std * prior_std
    log_ratio = jnp.log(prior_std) - 0.5 * logvar
    kl_per_element = log_ratio + (var + (mean - prior_mean) ** 2) / (2.0 * prior_var) - 0.5
    return jnp.sum(kl_per_element)


def sample_diagonal_gaussian(
    mean: jax.Array,
    logvar: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Reparameterization-trick sample from ``N(mean, exp(logvar))``.

    Args:
        mean: Posterior mean, any shape.
        logvar: Posterior log-variance, same shape as ``mean``.
        key: PRNG key (caller-owned; no hidden fixed seeds).

    Returns:
        A sample with the same shape as ``mean``.
    """
    std = jnp.exp(0.5 * logvar)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    return mean + std * noise
