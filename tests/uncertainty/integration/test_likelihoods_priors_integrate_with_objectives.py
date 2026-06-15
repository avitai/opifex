"""Pin likelihood and prior helpers' integration with the objective stack.

The Phase 1 likelihood / prior helpers
(:mod:`opifex.uncertainty.likelihoods`, :mod:`opifex.uncertainty.priors`)
exist to feed the Phase 1 objective surface
(:class:`opifex.uncertainty.UQLossComponents`). These integration tests pin
the cross-module wiring:

1. ``negative_log_likelihood`` from any of the five likelihood helpers flows
   into ``UQLossComponents.from_components`` as the ``negative_log_likelihood``
   term.
2. ``diagonal_gaussian_log_prior`` shares the
   ``(prior_mean, prior_std)`` parameterization with
   ``diagonal_gaussian_kl`` so the same prior config drives both an ELBO's
   KL term and a posterior log-density evaluation.
3. Both helpers are differentiable end-to-end: ``jax.grad`` over a small
   loss that uses likelihood + prior produces finite gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty import (
    diagonal_gaussian_kl,
    diagonal_gaussian_log_prior,
    gaussian_log_likelihood,
    ObjectiveConfig,
    UQLossComponents,
)


def _make_objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=128,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=1.0,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )


def test_gaussian_log_likelihood_feeds_uq_loss_components() -> None:
    """``-mean(log p(y|x))`` integrates as the ``negative_log_likelihood`` field."""
    y = jnp.array([0.1, -0.2, 0.3, 0.4])
    mean_pred = jnp.zeros_like(y)
    scale = jnp.ones_like(y)

    nll = -jnp.mean(gaussian_log_likelihood(y, mean=mean_pred, scale=scale))
    components = UQLossComponents.from_components(
        config=_make_objective_config(), negative_log_likelihood=nll
    )
    assert jnp.isfinite(components.total)
    assert components.negative_log_likelihood is not None
    assert jnp.isfinite(components.negative_log_likelihood)


def test_diagonal_prior_and_kl_share_prior_parameterization() -> None:
    """A posterior centred at the prior mean yields the closed-form KL = 0 link."""
    n_params = 16
    posterior_mean = jnp.zeros(n_params)
    posterior_logvar = jnp.zeros(n_params)
    prior_mean = 0.0
    prior_std = 1.0

    # When the posterior equals the prior, the diagonal Gaussian KL is exactly
    # zero — using the SAME (prior_mean, prior_std) drives both helpers.
    kl = diagonal_gaussian_kl(
        posterior_mean, posterior_logvar, prior_mean=prior_mean, prior_std=prior_std
    )
    log_prior_at_origin = diagonal_gaussian_log_prior(
        posterior_mean, prior_mean=prior_mean, prior_std=prior_std
    )
    assert float(kl) == 0.0
    assert jnp.isfinite(log_prior_at_origin)


def test_combined_loss_is_differentiable() -> None:
    """A loss that uses both helpers must produce finite gradients under jax.grad."""
    y = jnp.array([0.1, -0.2, 0.3])

    def loss_fn(params: jax.Array) -> jax.Array:
        mean_pred = params
        scale = jnp.ones_like(params)
        nll = -jnp.mean(gaussian_log_likelihood(y, mean=mean_pred, scale=scale))
        log_prior = diagonal_gaussian_log_prior(params, prior_mean=0.0, prior_std=1.0)
        return nll - log_prior

    params = jnp.array([0.1, 0.2, 0.3])
    grads = jax.grad(loss_fn)(params)
    assert grads.shape == params.shape
    assert jnp.all(jnp.isfinite(grads))


def test_combined_loss_is_jit_compatible() -> None:
    """Same loss must trace cleanly under jax.jit."""
    y = jnp.array([0.1, -0.2, 0.3])

    @jax.jit
    def loss_fn(params: jax.Array) -> jax.Array:
        mean_pred = params
        scale = jnp.ones_like(params)
        nll = -jnp.mean(gaussian_log_likelihood(y, mean=mean_pred, scale=scale))
        log_prior = diagonal_gaussian_log_prior(params, prior_mean=0.0, prior_std=1.0)
        return nll - log_prior

    out = loss_fn(jnp.array([0.1, 0.2, 0.3]))
    assert jnp.isfinite(out)
