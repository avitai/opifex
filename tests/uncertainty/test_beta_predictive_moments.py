"""Response moments under a Gaussian latent, and the Beta predictors built on them.

For a response ``y`` with conditional moments ``E[y | f]`` and ``Var[y | f]`` and a latent
``f ~ N(m, v)``, ``E[y] = E[E[y | f]]`` and ``Var[y] = E[Var[y | f]] + Var[E[y | f]]``.
:func:`predictive_moments` evaluates both expectations with a Gauss-Hermite rule and takes the second
about ``E[y]``. The Beta response (logit link, precision ``s``) has ``E[y | f] = sigmoid(f)`` and
``Var[y | f] = sigmoid(f) (1 - sigmoid(f)) / (s + 1)``. References are scipy adaptive quadrature in
float64 on the same latent moments.

Tolerances. A 20-point rule integrates these Beta moments within 2.3e-8 relative for latent
variances up to 1, so float32 rounding dominates. A mean is a sum of 20 terms bounded by their
weights, compared within ``24 eps32``; variances are compared within 1e-5 relative. Measured in
float32: the grid below reached a mean error of 7.3e-8 and a relative variance error of 2.8e-6 (at
precision 1000), and the five fitted predictors 1.1e-7 and 2.3e-7, with latent variances up to 0.33.
Before these predictors used quadrature, their mean errors on the same fits were 4.0e-4 to 5.1e-3.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from scipy.special import expit

from opifex.uncertainty._gauss_hermite import predictive_moments
from opifex.uncertainty.gp.laplace import predict_laplace_latent_moments
from opifex.uncertainty.gp.laplace_likelihoods import fit_beta_laplace_gp, predict_beta_laplace_gp
from opifex.uncertainty.markov import (
    fit_beta_markov_laplace_gp,
    fit_beta_markov_pep_gp,
    fit_beta_markov_pl_gp,
    fit_beta_markov_vi_gp,
    predict_beta_markov_laplace_gp,
    predict_beta_markov_pep_gp,
    predict_beta_markov_pl_gp,
    predict_beta_markov_vi_gp,
    predict_markov_laplace_gp,
    predict_markov_pep_gp,
    predict_markov_pl_gp,
    predict_markov_vi_gp,
)
from opifex.uncertainty.statespace import matern32_kernel


if TYPE_CHECKING:
    from collections.abc import Callable

    from opifex.uncertainty.types import PredictiveDistribution


_EPS32 = float(np.finfo(np.float32).eps)
_MEAN_TOLERANCE = 24 * _EPS32
_RELATIVE_VARIANCE_TOLERANCE = 1e-5
_SCALE = 20.0
_PREDICTORS = ("gp_laplace", "markov_laplace", "markov_vi", "markov_pep", "markov_pl")


def _beta_conditional_moments(scale: float) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Return ``f -> (sigmoid(f), sigmoid(f) (1 - sigmoid(f)) / (scale + 1))``."""

    def moments(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jax.nn.sigmoid(latent)
        return mean, mean * (1.0 - mean) / (scale + 1.0)

    return moments


def _beta_reference(mean: float, variance: float, scale: float) -> tuple[float, float]:
    """Return float64 ``(E[y], Var[y])`` of a Beta response by adaptive quadrature."""
    deviation = math.sqrt(variance)

    def density(latent: float) -> float:
        return math.exp(-0.5 * (latent - mean) ** 2 / variance) / math.sqrt(
            2.0 * math.pi * variance
        )

    limits = (mean - 40.0 * deviation, mean + 40.0 * deviation)
    options = {"limit": 400, "epsabs": 1e-14, "epsrel": 1e-12}
    expected_mean = integrate.quad(lambda f: expit(f) * density(f), *limits, **options)[0]
    expected_square = integrate.quad(lambda f: expit(f) ** 2 * density(f), *limits, **options)[0]
    expected_conditional_variance = integrate.quad(
        lambda f: expit(f) * (1.0 - expit(f)) / (scale + 1.0) * density(f), *limits, **options
    )[0]
    return expected_mean, expected_conditional_variance + expected_square - expected_mean**2


def _assert_matches_reference(
    means: jax.Array,
    variances: jax.Array,
    latent_means: jax.Array,
    latent_variances: jax.Array,
    scale: float,
) -> None:
    """Compare response moments with the adaptive-quadrature reference at every point."""
    for index in range(latent_means.shape[0]):
        reference_mean, reference_variance = _beta_reference(
            float(latent_means[index]), float(latent_variances[index]), scale
        )
        mean_error = abs(float(means[index]) - reference_mean)
        variance_error = abs(float(variances[index]) - reference_variance) / reference_variance
        assert mean_error <= _MEAN_TOLERANCE, (index, mean_error)
        assert variance_error <= _RELATIVE_VARIANCE_TOLERANCE, (index, variance_error)


@pytest.mark.parametrize("scale", [10.0, 1000.0])
def test_moments_match_adaptive_quadrature_for_a_beta_response(scale: float) -> None:
    """Beta response moments agree with adaptive quadrature over a grid of latent moments."""
    grid_means, grid_variances = np.meshgrid(np.linspace(-4.0, 4.0, 9), [1e-3, 0.1, 1.0])
    latent_means = jnp.asarray(grid_means.ravel())
    latent_variances = jnp.asarray(grid_variances.ravel())
    means, variances = predictive_moments(
        _beta_conditional_moments(scale), latent_means, latent_variances
    )
    _assert_matches_reference(means, variances, latent_means, latent_variances, scale)


def test_moments_of_a_linear_gaussian_response_are_exact() -> None:
    """``E[y | f] = 2 f + 1`` and ``Var[y | f] = 0.3`` give ``E[y] = 2 m + 1``, ``Var[y] = 0.3 + 4 v``."""
    latent_means = jnp.linspace(-2.0, 2.0, 5)
    latent_variances = jnp.linspace(0.0, 2.0, 5)
    means, variances = predictive_moments(
        lambda latent: (2.0 * latent + 1.0, jnp.full_like(latent, 0.3)),
        latent_means,
        latent_variances,
    )
    expected_means = 2.0 * latent_means + 1.0
    expected_variances = 0.3 + 4.0 * latent_variances
    assert jnp.all(
        jnp.abs(means - expected_means)
        <= _MEAN_TOLERANCE * jnp.maximum(1.0, jnp.abs(expected_means))
    )
    assert jnp.all(jnp.abs(variances - expected_variances) <= _MEAN_TOLERANCE * expected_variances)


def test_moments_support_jit_vmap_and_grad() -> None:
    """The moments compile once, vectorise, and differentiate in the latent mean and variance."""
    traces: list[None] = []

    def linear(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        return 2.0 * latent + 1.0, jnp.full_like(latent, 0.3)

    @jax.jit
    def moments(mean: jax.Array, variance: jax.Array) -> tuple[jax.Array, jax.Array]:
        traces.append(None)
        return predictive_moments(linear, mean, variance)

    moments(jnp.asarray(0.1), jnp.asarray(0.5))
    moments(jnp.asarray(-0.4), jnp.asarray(0.2))
    assert len(traces) == 1

    means = jnp.linspace(-1.0, 1.0, 7)
    batched_means, batched_variances = jax.vmap(moments, in_axes=(0, None))(means, jnp.asarray(0.5))
    assert batched_means.shape == (7,)
    assert jnp.all(jnp.abs(batched_variances - 2.3) <= _MEAN_TOLERANCE * 2.3)

    mean_slope = jax.grad(lambda m: moments(m, jnp.asarray(0.5))[0])(jnp.asarray(0.3))
    variance_slope = jax.grad(lambda v: moments(jnp.asarray(0.3), v)[1])(jnp.asarray(0.5))
    assert abs(float(mean_slope) - 2.0) <= _MEAN_TOLERANCE * 2.0
    assert abs(float(variance_slope) - 4.0) <= _MEAN_TOLERANCE * 4.0


def test_gradients_stay_finite_at_zero_latent_variance() -> None:
    """Differentiating the Beta moments at a latent variance of zero gives finite values."""

    def response(mean: jax.Array, variance: jax.Array) -> jax.Array:
        response_mean, response_variance = predictive_moments(
            _beta_conditional_moments(10.0), mean, variance
        )
        return response_mean + response_variance

    gradients = jax.grad(response, argnums=(0, 1))(jnp.asarray(0.4), jnp.asarray(0.0))
    assert all(bool(jnp.isfinite(gradient)) for gradient in gradients)


def _beta_observations(key: jax.Array, latent: jax.Array) -> jax.Array:
    """Draw Beta observations with mean ``sigmoid(latent)`` and precision ``_SCALE``."""
    mean = jax.nn.sigmoid(latent)
    return jax.random.beta(key, mean * _SCALE, _SCALE * (1.0 - mean))


@pytest.fixture(scope="module")
def beta_predictions() -> dict[str, tuple[PredictiveDistribution, jax.Array, jax.Array]]:
    """Fit each Beta model on toy data; return its predictive and its latent moments."""
    x_train = jnp.linspace(-1.5, 1.5, 30)[:, None]
    y_train = _beta_observations(jax.random.PRNGKey(3), jnp.sin(2.0 * x_train[:, 0]))
    gp_state = fit_beta_laplace_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        scale=_SCALE,
        num_newton_iterations=30,
    )
    x_test = jnp.linspace(-1.5, 1.5, 8)[:, None]
    gp_mean, gp_variance = predict_laplace_latent_moments(state=gp_state, x_test=x_test)
    results = {
        "gp_laplace": (
            predict_beta_laplace_gp(state=gp_state, x_test=x_test, scale=_SCALE),
            gp_mean,
            gp_variance,
        )
    }

    times = jnp.linspace(0.0, 4.0, 18)
    observations = _beta_observations(jax.random.PRNGKey(11), jnp.sin(2.0 * times))
    kernel = matern32_kernel(variance=1.0, lengthscale=0.5)
    common = {"times": times, "observations": observations, "state_space_kernel": kernel}
    times_test = jnp.linspace(0.5, 3.5, 6)
    markov_models = {
        "markov_laplace": (
            fit_beta_markov_laplace_gp(**common, scale=_SCALE, num_iterations=25),
            predict_beta_markov_laplace_gp,
            predict_markov_laplace_gp,
        ),
        "markov_vi": (
            fit_beta_markov_vi_gp(**common, scale=_SCALE, num_iterations=25),
            predict_beta_markov_vi_gp,
            predict_markov_vi_gp,
        ),
        "markov_pep": (
            fit_beta_markov_pep_gp(**common, scale=_SCALE, num_iterations=25, learning_rate=0.3),
            predict_beta_markov_pep_gp,
            predict_markov_pep_gp,
        ),
        "markov_pl": (
            fit_beta_markov_pl_gp(**common, scale=_SCALE, num_iterations=20),
            predict_beta_markov_pl_gp,
            predict_markov_pl_gp,
        ),
    }
    for name, (state, beta_predictor, latent_predictor) in markov_models.items():
        latent = latent_predictor(state=state, times_test=times_test)
        assert latent.variance is not None
        results[name] = (
            beta_predictor(state=state, times_test=times_test, scale=_SCALE),
            latent.mean,
            latent.variance,
        )
    return results


@pytest.mark.parametrize("name", _PREDICTORS)
def test_beta_predictor_moments_match_adaptive_quadrature(
    beta_predictions: dict[str, tuple[PredictiveDistribution, jax.Array, jax.Array]], name: str
) -> None:
    """Each Beta predictor returns the response mean and variance of its latent moments."""
    predictive, latent_means, latent_variances = beta_predictions[name]
    assert float(jnp.max(latent_variances)) <= 1.0, "tolerances assume latent variances up to 1"
    assert predictive.variance is not None
    _assert_matches_reference(
        predictive.mean, predictive.variance, latent_means, latent_variances, _SCALE
    )


@pytest.mark.parametrize("name", _PREDICTORS)
def test_beta_predictor_keeps_its_uncertainty_fields(
    beta_predictions: dict[str, tuple[PredictiveDistribution, jax.Array, jax.Array]], name: str
) -> None:
    """``epistemic`` is the latent variance and ``total_uncertainty`` the response variance."""
    predictive, _, latent_variances = beta_predictions[name]
    predictive.validate()
    assert predictive.epistemic is not None
    assert predictive.total_uncertainty is not None
    assert predictive.variance is not None
    assert predictive.aleatoric is None
    assert bool(jnp.all(predictive.epistemic == latent_variances))
    assert bool(jnp.all(predictive.total_uncertainty == predictive.variance))
    metadata = predictive.metadata_dict()
    assert metadata["likelihood"] == "beta"
    assert metadata["link"] == "logit"
