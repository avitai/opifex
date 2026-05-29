r"""Markov-Laplace per-likelihood wrappers — Task 11.2 slice 26.

Reuses the slice-25 :func:`fit_markov_laplace_gp` machinery and the
D5 ``LikelihoodComponentsFn`` factories shipped in
:mod:`opifex.uncertainty.gp.laplace_likelihoods` /
:mod:`opifex.uncertainty.gp.laplace_classification`. Each wrapper

* selects the appropriate components callable,
* calls :func:`fit_markov_laplace_gp` to run the Newton-Kalman loop,
* exposes a ``predict_*_markov_laplace_gp`` companion that maps the
  latent posterior through the per-likelihood response link
  (MacKay-probit for Bernoulli/Beta, log-normal for Poisson,
  identity-plus-noise for Student-t / Gaussian).

The exit criterion for Task 11.2 — *"at least 5 non-Gaussian
likelihoods + PEP/VI/Laplace inference paths with calibration
tests"* — needs five likelihoods on at least one inference path. This
slice ships **Bernoulli + Poisson + Student-t + Beta + Gaussian on
the Laplace path** (Bernoulli was shipped in slice 25; the other four
land here).

References
----------
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton`` likelihood +
  inference catalogue.
* Rasmussen & Williams 2006 §3.4-3.5 (Laplace + per-likelihood
  response links).
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
from opifex.uncertainty.markov.markov_laplace import (
    fit_markov_laplace_gp,
    MarkovLaplaceGPState,
    predict_markov_laplace_gp,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001 — runtime use
from opifex.uncertainty.types import PredictiveDistribution


_MARKOV_SOURCE_PACKAGE = "opifex.uncertainty.markov"


def _replace_metadata(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
) -> PredictiveDistribution:
    """Return a copy of ``predictive`` with refreshed provenance metadata."""
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
            source_package=_MARKOV_SOURCE_PACKAGE,
            extra=(
                ("estimator", estimator),
                (
                    "paper",
                    "bayesnewton / Sarkka 2013 §9 (Iterated-EKS Laplace on Markov GPs)",
                ),
                ("likelihood", likelihood),
                ("link", link),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Bernoulli (slice-25 carry-over with response-link mapping)
# -----------------------------------------------------------------------------


def fit_bernoulli_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 25,
) -> MarkovLaplaceGPState:
    """Bernoulli classification on a Markov-GP prior (slice-25 wrapper)."""
    return fit_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=num_iterations,
    )


def predict_bernoulli_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    """Predict ``p(y_* = +1 | t_*)`` via the MacKay probit collapse."""
    latent = predict_markov_laplace_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent.variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent.mean)
    response = PredictiveDistribution(
        mean=class_probability,
        variance=latent.variance,
        epistemic=latent.variance,
        total_uncertainty=latent.variance,
    )
    return _replace_metadata(
        response,
        estimator="bernoulli_markov_laplace_gp",
        likelihood="bernoulli",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Poisson (exp link)
# -----------------------------------------------------------------------------


def fit_poisson_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    num_iterations: int = 25,
) -> MarkovLaplaceGPState:
    """Poisson count regression on a Markov-GP prior (``exp`` link)."""
    return fit_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_poisson_log_likelihood_components,
        num_iterations=num_iterations,
    )


def predict_poisson_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict Poisson intensity ``E[λ*] = exp(μ + ½ V)`` under log-normal moments."""
    latent = predict_markov_laplace_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    intensity_mean = jnp.exp(latent.mean + 0.5 * latent.variance)
    intensity_variance = (intensity_mean**2) * jnp.expm1(latent.variance)
    response = PredictiveDistribution(
        mean=intensity_mean,
        variance=intensity_variance,
        epistemic=latent.variance,
        total_uncertainty=intensity_variance,
    )
    return _replace_metadata(
        response,
        estimator="poisson_markov_laplace_gp",
        likelihood="poisson",
        link="exp",
    )


# -----------------------------------------------------------------------------
# Student-t (identity link, df nu, scale sigma)
# -----------------------------------------------------------------------------


def fit_studentst_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    df: float = 4.0,
    scale: float = 1.0,
    num_iterations: int = 25,
) -> MarkovLaplaceGPState:
    """Student-t robust regression on a Markov-GP prior (Fisher-info ``W``)."""
    components = _studentst_components_factory(df=df, scale=scale)
    return fit_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
    )


def predict_studentst_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
    df: float = 4.0,
    scale: float = 1.0,
) -> PredictiveDistribution:
    r"""Student-t predictive: mean ``μ``, variance ``V + σ² ν / (ν − 2)``."""
    latent = predict_markov_laplace_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = latent.variance + scale_sq * df_arr / (df_arr - 2.0)
    response = PredictiveDistribution(
        mean=latent.mean,
        variance=response_variance,
        epistemic=latent.variance,
        total_uncertainty=response_variance,
    )
    return _replace_metadata(
        response,
        estimator="studentst_markov_laplace_gp",
        likelihood="students_t",
        link="identity",
    )


# -----------------------------------------------------------------------------
# Beta (logit link, scale s)
# -----------------------------------------------------------------------------


def fit_beta_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    scale: float = 10.0,
    num_iterations: int = 25,
) -> MarkovLaplaceGPState:
    """Beta proportion regression on a Markov-GP prior (logit link, Fisher ``W``)."""
    components = _beta_components_factory(scale=scale)
    return fit_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
    )


def predict_beta_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
    scale: float = 10.0,
) -> PredictiveDistribution:
    r"""Predict Beta response: mean ``σ(κ μ)``, variance ``m̂(1-m̂)/(s+1)``."""
    latent = predict_markov_laplace_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent.variance / 8.0)
    response_mean = jax.nn.sigmoid(kappa * latent.mean)
    response_variance = response_mean * (1.0 - response_mean) / (scale + 1.0)
    response = PredictiveDistribution(
        mean=response_mean,
        variance=response_variance,
        epistemic=latent.variance,
        total_uncertainty=response_variance,
    )
    return _replace_metadata(
        response,
        estimator="beta_markov_laplace_gp",
        likelihood="beta",
        link="logit",
    )


# -----------------------------------------------------------------------------
# Gaussian (identity link, fixed sigma) — conjugate, one-step Newton convergence
# -----------------------------------------------------------------------------


def _gaussian_components_factory(*, noise_std: float):
    """Build the Gaussian likelihood components (constant-curvature)."""
    noise_var = noise_std * noise_std

    def _components(
        f: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # log N(y_i; f_i, σ²) = -½ log(2π σ²) - (y - f)² / (2 σ²)
        log_lik = jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi * noise_var) - 0.5 * (y - f) ** 2 / noise_var)
        grad = (y - f) / noise_var
        w_diag = jnp.full_like(f, 1.0 / noise_var)
        return log_lik, grad, w_diag, jnp.sqrt(w_diag)

    return _components


def fit_gaussian_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    noise_std: float = 0.1,
    num_iterations: int = 1,
) -> MarkovLaplaceGPState:
    r"""Gaussian regression on a Markov-GP prior.

    For Gaussian likelihood the log-likelihood is quadratic in ``f``,
    so Newton converges in **one** step (the algorithm collapses to
    the exact conjugate Kalman smoother). Additional iterations are
    fixed-point no-ops.
    """
    components = _gaussian_components_factory(noise_std=noise_std)
    return fit_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=components,
        num_iterations=num_iterations,
    )


def predict_gaussian_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
    noise_std: float = 0.1,
) -> PredictiveDistribution:
    r"""Predict ``y* | t*``: mean ``μ``, variance ``V + σ²`` (latent + noise)."""
    latent = predict_markov_laplace_gp(state=state, times_test=times_test)
    assert latent.variance is not None
    response_variance = latent.variance + noise_std**2
    response = PredictiveDistribution(
        mean=latent.mean,
        variance=response_variance,
        epistemic=latent.variance,
        total_uncertainty=response_variance,
    )
    return _replace_metadata(
        response,
        estimator="gaussian_markov_laplace_gp",
        likelihood="gaussian",
        link="identity",
    )


__all__ = [
    "fit_bernoulli_markov_laplace_gp",
    "fit_beta_markov_laplace_gp",
    "fit_gaussian_markov_laplace_gp",
    "fit_poisson_markov_laplace_gp",
    "fit_studentst_markov_laplace_gp",
    "predict_bernoulli_markov_laplace_gp",
    "predict_beta_markov_laplace_gp",
    "predict_gaussian_markov_laplace_gp",
    "predict_poisson_markov_laplace_gp",
    "predict_studentst_markov_laplace_gp",
]
