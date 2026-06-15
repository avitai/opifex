r"""Markov-PL per-likelihood wrappers — Task 11.2 slice 30.

Routes the Posterior-Linearisation path
(:func:`opifex.uncertainty.markov.markov_pl.fit_markov_pl_gp`)
through per-likelihood ``conditional_moments_fn`` callables — closed-form
``(E[y|f], Var[y|f])`` for each link function.

Ships Bernoulli (logit link, +/-1 labels), Poisson (exp link), and
Gaussian (identity link). Student-t / Beta wrappers on the PL path
are filed as deferred item D9 alongside the matching PEP entries
(D8); the SLR + IEKS machinery is generic and adding them is a
one-commit follow-up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.markov.markov_pl import (
    fit_markov_pl_gp,
    MarkovPLGPState,
    predict_markov_pl_gp,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001
from opifex.uncertainty.types import PredictiveDistribution


_MARKOV_PL_SOURCE_PACKAGE = "opifex.uncertainty.markov"


def _latent_variance(predictive: PredictiveDistribution) -> jax.Array:
    """Unwrap the latent variance, which ``predict_markov_pl_gp`` always sets."""
    if predictive.variance is None:
        raise RuntimeError(
            "predict_markov_pl_gp returned a PredictiveDistribution with no variance"
        )
    return predictive.variance


def _replace_metadata_pl(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
) -> PredictiveDistribution:
    """Refresh predictive metadata to advertise the PL inference path."""
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
            source_package=_MARKOV_PL_SOURCE_PACKAGE,
            extra=(
                ("estimator", estimator),
                (
                    "paper",
                    "Garcia-Fernandez, Tronarp, Sarkka 2018 + Wilkinson, "
                    "Solin, Adam 2020+ (Posterior Linearisation on Markov GPs)",
                ),
                ("likelihood", likelihood),
                ("link", link),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Bernoulli (+/-1 labels, logit link)
# -----------------------------------------------------------------------------


def _bernoulli_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""``E[y|f] = 2 sigmoid(f) - 1, Var[y|f] = 1 - mean**2`` for +/-1 labels."""
    mean = 2.0 * jax.nn.sigmoid(f) - 1.0
    variance = 1.0 - mean * mean
    return mean, variance


def fit_bernoulli_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 20,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    """Bernoulli classification on a Markov-GP prior via Posterior Linearisation."""
    return fit_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=_bernoulli_conditional_moments,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_bernoulli_markov_pl_gp(
    *, state: MarkovPLGPState, times_test: jax.Array
) -> PredictiveDistribution:
    """Predict ``p(y_* = +1)`` via the MacKay probit collapse."""
    latent = predict_markov_pl_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent_variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent.mean)
    return _replace_metadata_pl(
        PredictiveDistribution(
            mean=class_probability,
            variance=latent_variance,
            epistemic=latent_variance,
            total_uncertainty=latent_variance,
        ),
        estimator="bernoulli_markov_pl_gp",
        likelihood="bernoulli",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Poisson (exp link)
# -----------------------------------------------------------------------------


def _poisson_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""``E[y|f] = exp(f), Var[y|f] = exp(f)`` (Poisson with exp link)."""
    rate = jnp.exp(f)
    return rate, rate


def fit_poisson_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 20,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    """Poisson count regression on a Markov-GP prior via Posterior Linearisation."""
    return fit_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=_poisson_conditional_moments,
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_poisson_markov_pl_gp(
    *, state: MarkovPLGPState, times_test: jax.Array
) -> PredictiveDistribution:
    r"""Predict Poisson intensity ``E[lambda] = exp(mu + 0.5 V)``."""
    latent = predict_markov_pl_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    intensity_mean = jnp.exp(latent.mean + 0.5 * latent_variance)
    intensity_variance = (intensity_mean**2) * jnp.expm1(latent_variance)
    return _replace_metadata_pl(
        PredictiveDistribution(
            mean=intensity_mean,
            variance=intensity_variance,
            epistemic=latent_variance,
            total_uncertainty=intensity_variance,
        ),
        estimator="poisson_markov_pl_gp",
        likelihood="poisson",
        link="exp",
    )


# -----------------------------------------------------------------------------
# Gaussian (identity link)
# -----------------------------------------------------------------------------


def _gaussian_conditional_moments_factory(*, noise_std: float):
    """Build ``f -> (f, noise_std**2)`` for the Gaussian identity link."""
    noise_var = noise_std * noise_std

    def _moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        return f, jnp.full_like(f, noise_var)

    return _moments


def fit_gaussian_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    noise_std: float = 0.1,
    num_iterations: int = 3,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    r"""Gaussian regression via Posterior Linearisation.

    For Gaussian likelihood the SLR linearisation is exact (the
    conditional mean is the identity in ``f``, so ``A = 1, b = 0,
    omega = noise_std**2``), and one iteration suffices to recover
    the conjugate Kalman solution.
    """
    return fit_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=_gaussian_conditional_moments_factory(noise_std=noise_std),
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_gaussian_markov_pl_gp(
    *,
    state: MarkovPLGPState,
    times_test: jax.Array,
    noise_std: float = 0.1,
) -> PredictiveDistribution:
    r"""Predict ``y*``: mean ``mu``, variance ``V + sigma**2`` (latent + obs noise)."""
    latent = predict_markov_pl_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    response_variance = latent_variance + noise_std**2
    return _replace_metadata_pl(
        PredictiveDistribution(
            mean=latent.mean,
            variance=response_variance,
            epistemic=latent_variance,
            total_uncertainty=response_variance,
        ),
        estimator="gaussian_markov_pl_gp",
        likelihood="gaussian",
        link="identity",
    )


# -----------------------------------------------------------------------------
# Student-t (location-scale, robust regression — closes deferral D9)
# -----------------------------------------------------------------------------


def _studentst_conditional_moments_factory(*, df: float, scale: float):
    r"""``f -> (f, scale**2 * df / (df - 2))`` for location-scale Student-t (df > 2)."""
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = scale_sq * df_arr / (df_arr - 2.0)

    def _moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        return f, jnp.full_like(f, response_variance)

    return _moments


def fit_studentst_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    df: float = 4.0,
    scale: float = 1.0,
    num_iterations: int = 20,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    r"""Robust regression on a Markov-GP prior via Posterior Linearisation.

    For a location-scale Student-t with ``df > 2``, ``E[y|f] = f`` and
    the response variance is constant, so the SLR linearisation collapses
    to a Gaussian observation model with that constant variance.
    """
    return fit_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=_studentst_conditional_moments_factory(df=df, scale=scale),
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_studentst_markov_pl_gp(
    *,
    state: MarkovPLGPState,
    times_test: jax.Array,
    df: float = 4.0,
    scale: float = 1.0,
) -> PredictiveDistribution:
    r"""Predict ``y*`` under the Student-t response (df > 2 finite variance)."""
    latent = predict_markov_pl_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = latent_variance + scale_sq * df_arr / (df_arr - 2.0)
    return _replace_metadata_pl(
        PredictiveDistribution(
            mean=latent.mean,
            variance=response_variance,
            epistemic=latent_variance,
            total_uncertainty=response_variance,
        ),
        estimator="studentst_markov_pl_gp",
        likelihood="studentst",
        link="identity",
    )


# -----------------------------------------------------------------------------
# Beta (unit-interval regression, logit link — closes deferral D9)
# -----------------------------------------------------------------------------


def _beta_conditional_moments_factory(*, scale: float):
    r"""``f -> (sigmoid(f), sigmoid(f) (1 - sigmoid(f)) / (scale + 1))``."""
    scale_arr = jnp.asarray(scale)

    def _moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jax.nn.sigmoid(f)
        variance = mean * (1.0 - mean) / (scale_arr + 1.0)
        return mean, variance

    return _moments


def fit_beta_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    scale: float = 10.0,
    num_iterations: int = 20,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    """Beta regression on a Markov-GP prior via Posterior Linearisation."""
    return fit_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=_beta_conditional_moments_factory(scale=scale),
        num_iterations=num_iterations,
        num_quadrature_points=num_quadrature_points,
    )


def predict_beta_markov_pl_gp(
    *,
    state: MarkovPLGPState,
    times_test: jax.Array,
    scale: float = 10.0,
) -> PredictiveDistribution:
    r"""Predict Beta response mean ``sigmoid(latent_mean)`` + Beta marginal variance."""
    latent = predict_markov_pl_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    response_mean = jax.nn.sigmoid(latent.mean)
    response_variance = response_mean * (1.0 - response_mean) / (scale + 1.0)
    return _replace_metadata_pl(
        PredictiveDistribution(
            mean=response_mean,
            variance=response_variance,
            epistemic=latent_variance,
            total_uncertainty=response_variance,
        ),
        estimator="beta_markov_pl_gp",
        likelihood="beta",
        link="logit",
    )


__all__ = [
    "fit_bernoulli_markov_pl_gp",
    "fit_beta_markov_pl_gp",
    "fit_gaussian_markov_pl_gp",
    "fit_poisson_markov_pl_gp",
    "fit_studentst_markov_pl_gp",
    "predict_bernoulli_markov_pl_gp",
    "predict_beta_markov_pl_gp",
    "predict_gaussian_markov_pl_gp",
    "predict_poisson_markov_pl_gp",
    "predict_studentst_markov_pl_gp",
]
