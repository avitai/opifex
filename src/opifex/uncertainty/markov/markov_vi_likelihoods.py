r"""Markov-VI per-likelihood wrappers — Task 11.2 slice 27.

Mirrors :mod:`opifex.uncertainty.markov.markov_laplace_likelihoods`
but routes through the conjugate-computation VI path
(:func:`opifex.uncertainty.markov.markov_vi.fit_markov_vi_gp`).
Reuses the D5 ``LikelihoodComponentsFn`` factories so there is no
duplication of per-likelihood maths.

For Gaussian likelihood the VI iteration coincides with the Laplace
iteration (linear-in-``f`` log-lik gradient + constant curvature),
so VI and Laplace produce identical posteriors — the slice-27
cross-check ``test_markov_vi_gaussian_likelihood_matches_markov_laplace_gaussian``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.laplace_classification import (
    _bernoulli_log_likelihood_components,
)
from opifex.uncertainty.gp.laplace_likelihoods import (
    _beta_components_factory,
    _poisson_log_likelihood_components,
    _studentst_components_factory,
)
from opifex.uncertainty.markov.markov_laplace_likelihoods import (
    _gaussian_components_factory,
)
from opifex.uncertainty.markov.markov_vi import (
    fit_markov_vi_gp,
    MarkovVIGPState,
    predict_markov_vi_gp,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001
from opifex.uncertainty.types import PredictiveDistribution


_MARKOV_VI_SOURCE_PACKAGE = "opifex.uncertainty.markov"


def _replace_metadata_vi(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
) -> PredictiveDistribution:
    """Refresh predictive metadata to advertise the VI inference path."""
    return PredictiveDistribution(
        mean=predictive.mean,
        variance=predictive.variance,
        epistemic=predictive.epistemic,
        aleatoric=predictive.aleatoric,
        total_uncertainty=predictive.total_uncertainty,
        samples=predictive.samples,
        covariance=predictive.covariance,
        quantiles=predictive.quantiles,
        interval=predictive.interval,
        prediction_set=predictive.prediction_set,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_MARKOV_VI_SOURCE_PACKAGE,
            extra=(
                ("estimator", estimator),
                (
                    "paper",
                    "Khan & Lin 2017 / Chang+ 2020 (Conjugate-Computation VI on Markov GPs)",
                ),
                ("likelihood", likelihood),
                ("link", link),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Bernoulli (binary classification, logit link, MacKay-probit response)
# -----------------------------------------------------------------------------


def fit_bernoulli_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 25,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    """Bernoulli classification on a Markov-GP prior via conjugate VI."""
    return fit_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_bernoulli_markov_vi_gp(
    *, state: MarkovVIGPState, times_test: jax.Array
) -> PredictiveDistribution:
    """Predict ``p(y_* = +1)`` via the MacKay probit collapse."""
    latent = predict_markov_vi_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent.variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent.mean)
    return _replace_metadata_vi(
        PredictiveDistribution(
            mean=class_probability,
            variance=latent.variance,
            epistemic=latent.variance,
            total_uncertainty=latent.variance,
        ),
        estimator="bernoulli_markov_vi_gp",
        likelihood="bernoulli",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Poisson (counts, exp link, log-normal predictive)
# -----------------------------------------------------------------------------


def fit_poisson_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 25,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    """Poisson count regression on a Markov-GP prior via conjugate VI."""
    return fit_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_poisson_log_likelihood_components,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_poisson_markov_vi_gp(
    *, state: MarkovVIGPState, times_test: jax.Array
) -> PredictiveDistribution:
    r"""Predict Poisson intensity ``E[λ] = exp(μ + ½ V)`` under log-normal moments."""
    latent = predict_markov_vi_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    intensity_mean = jnp.exp(latent.mean + 0.5 * latent.variance)
    intensity_variance = (intensity_mean**2) * jnp.expm1(latent.variance)
    return _replace_metadata_vi(
        PredictiveDistribution(
            mean=intensity_mean,
            variance=intensity_variance,
            epistemic=latent.variance,
            total_uncertainty=intensity_variance,
        ),
        estimator="poisson_markov_vi_gp",
        likelihood="poisson",
        link="exp",
    )


# -----------------------------------------------------------------------------
# Student-t (robust regression, identity link, Fisher-info curvature)
# -----------------------------------------------------------------------------


def fit_studentst_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    df: float = 4.0,
    scale: float = 1.0,
    num_iterations: int = 25,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    """Student-t robust regression on a Markov-GP prior via conjugate VI."""
    components = _studentst_components_factory(df=df, scale=scale)
    return fit_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_studentst_markov_vi_gp(
    *,
    state: MarkovVIGPState,
    times_test: jax.Array,
    df: float = 4.0,
    scale: float = 1.0,
) -> PredictiveDistribution:
    r"""Predict Student-t response: mean ``μ``, variance ``V + σ²ν/(ν−2)``."""
    latent = predict_markov_vi_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = latent.variance + scale_sq * df_arr / (df_arr - 2.0)
    return _replace_metadata_vi(
        PredictiveDistribution(
            mean=latent.mean,
            variance=response_variance,
            epistemic=latent.variance,
            total_uncertainty=response_variance,
        ),
        estimator="studentst_markov_vi_gp",
        likelihood="students_t",
        link="identity",
    )


# -----------------------------------------------------------------------------
# Beta (proportion regression, logit link, Fisher-info curvature)
# -----------------------------------------------------------------------------


def fit_beta_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    scale: float = 10.0,
    num_iterations: int = 25,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    """Beta proportion regression on a Markov-GP prior via conjugate VI."""
    components = _beta_components_factory(scale=scale)
    return fit_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_beta_markov_vi_gp(
    *,
    state: MarkovVIGPState,
    times_test: jax.Array,
    scale: float = 10.0,
) -> PredictiveDistribution:
    r"""Predict Beta response: ``E[y] = σ(κ μ)``, variance ``m̂(1-m̂)/(s+1)``."""
    latent = predict_markov_vi_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent.variance / 8.0)
    response_mean = jax.nn.sigmoid(kappa * latent.mean)
    response_variance = response_mean * (1.0 - response_mean) / (scale + 1.0)
    return _replace_metadata_vi(
        PredictiveDistribution(
            mean=response_mean,
            variance=response_variance,
            epistemic=latent.variance,
            total_uncertainty=response_variance,
        ),
        estimator="beta_markov_vi_gp",
        likelihood="beta",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Gaussian (identity link, conjugate — VI ≡ Laplace ≡ exact Kalman in 1 iter)
# -----------------------------------------------------------------------------


def fit_gaussian_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    noise_std: float = 0.1,
    num_iterations: int = 1,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    r"""Gaussian regression on a Markov-GP prior.

    For Gaussian likelihood the log-likelihood gradient is linear in
    ``f`` and the curvature is constant, so the expected components
    (over any ``q(f)``) coincide with the mode-evaluated components.
    VI converges in **one** iteration to the exact conjugate Kalman
    solution — identical to
    :func:`opifex.uncertainty.markov.fit_gaussian_markov_laplace_gp`.
    """
    components = _gaussian_components_factory(noise_std=noise_std)
    return fit_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_gaussian_markov_vi_gp(
    *,
    state: MarkovVIGPState,
    times_test: jax.Array,
    noise_std: float = 0.1,
) -> PredictiveDistribution:
    r"""Predict ``y*``: mean ``μ``, variance ``V + σ²`` (latent + obs noise)."""
    latent = predict_markov_vi_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    response_variance = latent.variance + noise_std**2
    return _replace_metadata_vi(
        PredictiveDistribution(
            mean=latent.mean,
            variance=response_variance,
            epistemic=latent.variance,
            total_uncertainty=response_variance,
        ),
        estimator="gaussian_markov_vi_gp",
        likelihood="gaussian",
        link="identity",
    )


__all__ = [
    "fit_bernoulli_markov_vi_gp",
    "fit_beta_markov_vi_gp",
    "fit_gaussian_markov_vi_gp",
    "fit_poisson_markov_vi_gp",
    "fit_studentst_markov_vi_gp",
    "predict_bernoulli_markov_vi_gp",
    "predict_beta_markov_vi_gp",
    "predict_gaussian_markov_vi_gp",
    "predict_poisson_markov_vi_gp",
    "predict_studentst_markov_vi_gp",
]
