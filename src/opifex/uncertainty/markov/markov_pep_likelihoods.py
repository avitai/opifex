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

from opifex.uncertainty._predictive import (
    gaussian_process_predictive,
    replace_predictive_metadata,
)
from opifex.uncertainty.gp.svgp_stochastic import (
    bernoulli_log_likelihood,
    poisson_log_likelihood,
)
from opifex.uncertainty.markov._likelihood_support import latent_variance
from opifex.uncertainty.markov.markov_pep import (
    fit_markov_pep_gp,
    LogZAndDerivativesFn,
    MarkovPEPGPState,
    predict_markov_pep_gp,
)
from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_MARKOV_PEP_PAPER = "Minka 2001/2004 + Wilkinson, Solin, Adam 2020+ (PEP on Markov GPs)"


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
    variance = latent_variance(latent)
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent.mean)
    return replace_predictive_metadata(
        gaussian_process_predictive(
            class_probability,
            variance,
            epistemic=variance,
            total_uncertainty=variance,
        ),
        estimator="bernoulli_markov_pep_gp",
        likelihood="bernoulli",
        link="logit",
        paper=_MARKOV_PEP_PAPER,
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
    variance = latent_variance(latent)
    intensity_mean = jnp.exp(latent.mean + 0.5 * variance)
    intensity_variance = (intensity_mean**2) * jnp.expm1(variance)
    return replace_predictive_metadata(
        gaussian_process_predictive(
            intensity_mean,
            intensity_variance,
            epistemic=variance,
            total_uncertainty=intensity_variance,
        ),
        estimator="poisson_markov_pep_gp",
        likelihood="poisson",
        link="exp",
        paper=_MARKOV_PEP_PAPER,
    )


# -----------------------------------------------------------------------------
# Gaussian (identity link, conjugate — PEP at power=1 reduces to exact Kalman)
# -----------------------------------------------------------------------------


def _gaussian_per_obs_log_likelihood_factory(*, noise_std: float):
    """Build a per-observation Gaussian log-likelihood callable."""
    noise_var = noise_std * noise_std

    def _log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        """Return the Gaussian log-likelihood of ``y`` given latent ``f``."""
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
        """Return the power-EP Gaussian log-partition and its first two derivatives."""
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
    variance = latent_variance(latent)
    response_variance = variance + noise_std**2
    return replace_predictive_metadata(
        gaussian_process_predictive(
            latent.mean,
            response_variance,
            epistemic=variance,
            total_uncertainty=response_variance,
        ),
        estimator="gaussian_markov_pep_gp",
        likelihood="gaussian",
        link="identity",
        paper=_MARKOV_PEP_PAPER,
    )


# -----------------------------------------------------------------------------
# Student-t (location-scale, robust regression — closes deferral D8)
# -----------------------------------------------------------------------------


def _studentst_per_obs_log_likelihood_factory(*, df: float, scale: float):
    r"""Per-observation Student-t log-likelihood ``(f, y) -> log p(y|f)``.

    Matches the closed-form Student-t density in
    :func:`opifex.uncertainty.gp.laplace_likelihoods._studentst_components_factory`.
    """
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    df_times_scale_sq = df_arr * scale_sq
    log_const = (
        jax.scipy.special.gammaln(0.5 * (df_arr + 1.0))
        - jax.scipy.special.gammaln(0.5 * df_arr)
        - 0.5 * (jnp.log(scale_sq) + jnp.log(df_arr) + jnp.log(jnp.pi))
    )

    def _log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        """Return the Student-t log-likelihood of ``y`` given latent ``f``."""
        residual_sq = (y - f) ** 2
        return log_const - 0.5 * (df_arr + 1.0) * jnp.log(1.0 + residual_sq / df_times_scale_sq)

    return _log_lik


def fit_studentst_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    df: float = 4.0,
    scale: float = 1.0,
    power: float = 0.5,
    num_iterations: int = 25,
    learning_rate: float = 0.5,
    num_quadrature_points: int = 20,
) -> MarkovPEPGPState:
    """Robust regression on a Markov-GP prior via Power EP with Student-t likelihood."""
    return fit_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=_studentst_per_obs_log_likelihood_factory(df=df, scale=scale),
        power=power,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        num_quadrature_points=num_quadrature_points,
    )


def predict_studentst_markov_pep_gp(
    *,
    state: MarkovPEPGPState,
    times_test: jax.Array,
    df: float = 4.0,
    scale: float = 1.0,
) -> PredictiveDistribution:
    r"""Predict ``y*`` under the Student-t response.

    Latent mean ``μ``; response variance is the latent variance plus the
    Student-t marginal variance ``scale² ν / (ν - 2)`` (finite for ``ν > 2``).
    """
    latent = predict_markov_pep_gp(state=state, times_test=times_test)
    variance = latent_variance(latent)
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = variance + scale_sq * df_arr / (df_arr - 2.0)
    return replace_predictive_metadata(
        gaussian_process_predictive(
            latent.mean,
            response_variance,
            epistemic=variance,
            total_uncertainty=response_variance,
        ),
        estimator="studentst_markov_pep_gp",
        likelihood="studentst",
        link="identity",
        paper=_MARKOV_PEP_PAPER,
    )


# -----------------------------------------------------------------------------
# Beta (unit-interval regression, logit link — closes deferral D8)
# -----------------------------------------------------------------------------


def _beta_per_obs_log_likelihood_factory(*, scale: float):
    r"""Per-observation Beta log-likelihood with logit link.

    Matches the closed-form Beta density in
    :func:`opifex.uncertainty.gp.laplace_likelihoods._beta_components_factory`.
    """
    scale_arr = jnp.asarray(scale)

    def _log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        """Return the Beta log-likelihood of ``y`` given latent ``f`` (logit link)."""
        mean = jax.nn.sigmoid(f)
        alpha = mean * scale_arr
        beta = scale_arr - alpha
        y_clipped = jnp.clip(y, a_min=1e-6, a_max=1.0 - 1e-6)
        return (
            (alpha - 1.0) * jnp.log(y_clipped)
            + (beta - 1.0) * jnp.log(1.0 - y_clipped)
            + jax.scipy.special.gammaln(scale_arr)
            - jax.scipy.special.gammaln(alpha)
            - jax.scipy.special.gammaln(beta)
        )

    return _log_lik


def fit_beta_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    scale: float = 10.0,
    power: float = 0.5,
    num_iterations: int = 25,
    learning_rate: float = 0.5,
    num_quadrature_points: int = 20,
) -> MarkovPEPGPState:
    """Beta regression on a Markov-GP prior via Power EP with logit link."""
    return fit_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=_beta_per_obs_log_likelihood_factory(scale=scale),
        power=power,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        num_quadrature_points=num_quadrature_points,
    )


def predict_beta_markov_pep_gp(
    *,
    state: MarkovPEPGPState,
    times_test: jax.Array,
    scale: float = 10.0,
) -> PredictiveDistribution:
    r"""Predict Beta response mean ``E[y*] = sigmoid(latent_mean)``.

    Predictive variance under the unit-interval link uses the standard
    Beta(α, β) marginal variance ``mean (1 - mean) / (scale + 1)`` at
    ``mean = sigmoid(latent_mean)``.
    """
    latent = predict_markov_pep_gp(state=state, times_test=times_test)
    variance = latent_variance(latent)
    response_mean = jax.nn.sigmoid(latent.mean)
    response_variance = response_mean * (1.0 - response_mean) / (scale + 1.0)
    return replace_predictive_metadata(
        gaussian_process_predictive(
            response_mean,
            response_variance,
            epistemic=variance,
            total_uncertainty=response_variance,
        ),
        estimator="beta_markov_pep_gp",
        likelihood="beta",
        link="logit",
        paper=_MARKOV_PEP_PAPER,
    )


__all__ = [
    "fit_bernoulli_markov_pep_gp",
    "fit_beta_markov_pep_gp",
    "fit_gaussian_markov_pep_gp",
    "fit_poisson_markov_pep_gp",
    "fit_studentst_markov_pep_gp",
    "predict_bernoulli_markov_pep_gp",
    "predict_beta_markov_pep_gp",
    "predict_gaussian_markov_pep_gp",
    "predict_poisson_markov_pep_gp",
    "predict_studentst_markov_pep_gp",
]
