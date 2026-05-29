r"""Markov-PEP per-likelihood wrappers — Task 11.2 slice 28.

Routes the Power-EP path
(:func:`opifex.uncertainty.markov.markov_pep.fit_markov_pep_gp`)
through the three per-observation log-likelihood callables already
shipped in :mod:`opifex.uncertainty.gp.svgp_stochastic`:

* :func:`bernoulli_log_likelihood` — binary classification, logit
  link.
* :func:`poisson_log_likelihood` — count regression, ``exp`` link.
* Gaussian — closed-form log-density; conjugate / one-iteration
  fixed point at ``α = 1``.

The three are sufficient for the Phase 11 Task 11.2 exit criterion
(*"PEP/VI/Laplace inference paths with calibration tests"*); the
remaining likelihoods (Student-t / Beta) on the PEP path are filed
as the deferred item ``D8`` for future work — both have factory
parameters (df / scale) and are straightforward to add once a
per-observation Student-t and per-observation Beta log-likelihood
are exposed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.svgp_stochastic import (
    bernoulli_log_likelihood,
    poisson_log_likelihood,
)
from opifex.uncertainty.markov.markov_pep import (
    fit_markov_pep_gp,
    LogZAndDerivativesFn,
    MarkovPEPGPState,
    predict_markov_pep_gp,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001
from opifex.uncertainty.types import PredictiveDistribution


_MARKOV_PEP_SOURCE_PACKAGE = "opifex.uncertainty.markov"


def _latent_variance(predictive: PredictiveDistribution) -> jax.Array:
    """Unwrap the latent variance, which ``predict_markov_pep_gp`` always sets."""
    if predictive.variance is None:
        raise RuntimeError(
            "predict_markov_pep_gp returned a PredictiveDistribution with no variance"
        )
    return predictive.variance


def _replace_metadata_pep(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
) -> PredictiveDistribution:
    """Refresh predictive metadata to advertise the PEP inference path."""
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
            source_package=_MARKOV_PEP_SOURCE_PACKAGE,
            extra=(
                ("estimator", estimator),
                (
                    "paper",
                    "Minka 2001/2004 + Wilkinson, Solin, Adam 2020+ (PEP on Markov GPs)",
                ),
                ("likelihood", likelihood),
                ("link", link),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Bernoulli (logit link, MacKay-probit response)
# -----------------------------------------------------------------------------


def fit_bernoulli_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    power: float = 0.5,
    num_iterations: int = 25,
    learning_rate: float = 0.5,
    num_quadrature_points: int = 20,
) -> MarkovPEPGPState:
    """Bernoulli classification on a Markov-GP prior via Power EP."""
    return fit_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=bernoulli_log_likelihood,
        power=power,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        num_quadrature_points=num_quadrature_points,
    )


def predict_bernoulli_markov_pep_gp(
    *, state: MarkovPEPGPState, times_test: jax.Array
) -> PredictiveDistribution:
    """Predict ``p(y_* = +1)`` via the MacKay probit collapse."""
    latent = predict_markov_pep_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent_variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent.mean)
    return _replace_metadata_pep(
        PredictiveDistribution(
            mean=class_probability,
            variance=latent_variance,
            epistemic=latent_variance,
            total_uncertainty=latent_variance,
        ),
        estimator="bernoulli_markov_pep_gp",
        likelihood="bernoulli",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Poisson (exp link, log-normal predictive)
# -----------------------------------------------------------------------------


def fit_poisson_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    power: float = 0.5,
    num_iterations: int = 25,
    learning_rate: float = 0.5,
    num_quadrature_points: int = 20,
) -> MarkovPEPGPState:
    """Poisson count regression on a Markov-GP prior via Power EP."""
    return fit_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=poisson_log_likelihood,
        power=power,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        num_quadrature_points=num_quadrature_points,
    )


def predict_poisson_markov_pep_gp(
    *, state: MarkovPEPGPState, times_test: jax.Array
) -> PredictiveDistribution:
    r"""Predict Poisson intensity ``E[λ] = exp(μ + ½ V)`` under log-normal moments."""
    latent = predict_markov_pep_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    intensity_mean = jnp.exp(latent.mean + 0.5 * latent_variance)
    intensity_variance = (intensity_mean**2) * jnp.expm1(latent_variance)
    return _replace_metadata_pep(
        PredictiveDistribution(
            mean=intensity_mean,
            variance=intensity_variance,
            epistemic=latent_variance,
            total_uncertainty=intensity_variance,
        ),
        estimator="poisson_markov_pep_gp",
        likelihood="poisson",
        link="exp",
    )


# -----------------------------------------------------------------------------
# Gaussian (identity link, conjugate — PEP at power=1 reduces to exact Kalman)
# -----------------------------------------------------------------------------


def _gaussian_per_obs_log_likelihood_factory(*, noise_std: float):
    """Build a per-observation Gaussian log-likelihood callable."""
    noise_var = noise_std * noise_std

    def _log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        return -0.5 * jnp.log(2.0 * jnp.pi * noise_var) - 0.5 * (y - f) ** 2 / noise_var

    return _log_lik


def _gaussian_log_partition_factory(*, noise_std: float) -> LogZAndDerivativesFn:
    r"""Closed-form ``log Z(m, v)`` for Gaussian likelihood.

    For ``p(y|f) = N(y; f, σ²)`` and proposal ``N(f; m, v)``::

        Z(m, v) = ∫ N(y; f, σ²)^α N(f; m, v) df
                = (2π)^(-α/2) σ^(-α) (σ²/α)^(1/2) (v + σ²/α)^(-1/2)
                  · exp(-α (y - m)² / (2 (v α + σ²))).

    Differentiating w.r.t. ``m`` (the only argument relevant for EP)::

        dlogZ/dm = α (y - m) / (v α + σ²),
        d²logZ/dm² = -α / (v α + σ²).

    Bypasses Gauss-Hermite cubature entirely — numerically stable for
    any ``σ²``, including the ``σ² = 0.0025`` regression regime that
    breaks naive cavity-centered cubature. See the SOTA discussion in
    [[feedback_no_impulsive_technical_fixes]].
    """
    noise_var = noise_std * noise_std
    log_2pi = float(jnp.log(2.0 * jnp.pi))

    def _log_partition_fn(
        cavity_means: jax.Array,
        cavity_variances: jax.Array,
        observations: jax.Array,
        power: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        denom = cavity_variances * power + noise_var
        log_Z = (
            -0.5 * power * (log_2pi + jnp.log(noise_var))
            + 0.5 * jnp.log(noise_var / power)
            - 0.5 * jnp.log(denom)
            - 0.5 * power * (observations - cavity_means) ** 2 / denom
        )
        dlogZ_dm = power * (observations - cavity_means) / denom
        d2logZ_dm2 = -power / denom * jnp.ones_like(cavity_means)
        return log_Z, dlogZ_dm, d2logZ_dm2

    return _log_partition_fn


def fit_gaussian_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    noise_std: float = 0.1,
    power: float = 1.0,
    num_iterations: int = 5,
    learning_rate: float = 1.0,
    num_quadrature_points: int = 20,
) -> MarkovPEPGPState:
    r"""Gaussian regression on a Markov-GP prior via Power EP.

    Defaults to classical EP (``power = 1, learning_rate = 1``). For
    Gaussian likelihood the tilted distribution is exactly Gaussian,
    so we use the **closed-form** partition function and its
    derivatives (``_gaussian_log_partition_factory``) — bypassing
    Gauss-Hermite cubature, which would break down for narrow
    ``noise_std`` relative to the cavity scale (a known limitation of
    cavity-centered GH; see e.g. Hernández-Lobato et al. on BB-α).
    One EP sweep then exactly recovers the conjugate Kalman solution.
    """
    log_lik_fn = _gaussian_per_obs_log_likelihood_factory(noise_std=noise_std)
    log_partition_fn = _gaussian_log_partition_factory(noise_std=noise_std)
    return fit_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=log_lik_fn,
        power=power,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        num_quadrature_points=num_quadrature_points,
        log_partition_fn=log_partition_fn,
    )


def predict_gaussian_markov_pep_gp(
    *,
    state: MarkovPEPGPState,
    times_test: jax.Array,
    noise_std: float = 0.1,
) -> PredictiveDistribution:
    r"""Predict ``y*``: mean ``μ``, variance ``V + σ²`` (latent + obs noise)."""
    latent = predict_markov_pep_gp(state=state, times_test=times_test)
    latent_variance = _latent_variance(latent)
    response_variance = latent_variance + noise_std**2
    return _replace_metadata_pep(
        PredictiveDistribution(
            mean=latent.mean,
            variance=response_variance,
            epistemic=latent_variance,
            total_uncertainty=response_variance,
        ),
        estimator="gaussian_markov_pep_gp",
        likelihood="gaussian",
        link="identity",
    )


__all__ = [
    "fit_bernoulli_markov_pep_gp",
    "fit_gaussian_markov_pep_gp",
    "fit_poisson_markov_pep_gp",
    "predict_bernoulli_markov_pep_gp",
    "predict_gaussian_markov_pep_gp",
    "predict_poisson_markov_pep_gp",
]
