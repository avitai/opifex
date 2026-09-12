"""Evidence values of the Markov-GP inference methods.

For a Gaussian likelihood, variational inference, the Laplace approximation and power expectation
propagation all recover the exact posterior, so each method's evidence must equal the exact log
marginal likelihood. The conjugate tests port bayesnewton's ``tests/test_vs_exact_marg_lik.py``
(AaltoML/BayesNewton f72ae9a, Apache-2.0): the same wiggly series on an uneven grid, the same
Matern-5/2 hyperparameter grid, and the same four-decimal comparison. The series uses a fixed seed.

The formulas are those of Chang, Wilkinson, Khan and Solin (2020, eq. 11) for the ELBO and of
Wilkinson, Sarkka and Solin (JMLR 2023, eqs. 16, 17 and 27) for VI, Laplace and power EP. For a
non-conjugate likelihood, eq. (17) at the Newton fixed point equals the dense Laplace
approximation of Rasmussen and Williams (2006, eq. 3.32).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from opifex.uncertainty.gp import (
    matern32_kernel as dense_matern32_kernel,
    matern52_kernel as dense_matern52_kernel,
)
from opifex.uncertainty.markov import (
    fit_bernoulli_markov_laplace_gp,
    fit_gaussian_markov_laplace_gp,
    fit_gaussian_markov_pep_gp,
    fit_gaussian_markov_vi_gp,
)
from opifex.uncertainty.statespace import matern32_kernel, matern52_kernel, StateSpaceKernel
from tests.uncertainty.markov._helpers import binary_labels


pytestmark = pytest.mark.usefixtures("float64")

_LOG_2PI = float(np.log(2.0 * np.pi))
# Hyperparameter grid of bayesnewton tests/test_vs_exact_marg_lik.py.
_VARIANCES = [0.5, 1.5]
_LENGTHSCALES = [0.75, 2.5]
_NOISE_VARIANCES = [0.1, 0.5]
_NUM_POINTS = [30, 60]


def _wiggly_time_series(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bayesnewton's uneven-grid test series ``(times, observations)``."""
    rng = np.random.default_rng(12345)
    times = np.sort(
        np.linspace(-25.0, 150.0, num=num_points) + 0.5 * rng.standard_normal(num_points)
    )
    signal = np.cos(0.04 * times + 0.33 * np.pi) * np.sin(0.2 * times)
    return times, signal + np.sqrt(0.15) * rng.standard_normal(num_points)


def _exact_log_marginal_likelihood(
    times: np.ndarray,
    observations: np.ndarray,
    variance: float,
    lengthscale: float,
    noise_variance: float,
) -> float:
    """Return ``log N(y | 0, K + noise I)`` for the dense Matern-5/2 covariance."""
    points = jnp.asarray(times)[:, None]
    gram = np.asarray(
        dense_matern52_kernel(
            points, points, lengthscale=lengthscale, output_scale=float(np.sqrt(variance))
        )
    )
    factor = cho_factor(gram + noise_variance * np.eye(times.size), lower=True)
    return float(
        -0.5 * observations @ cho_solve(factor, observations)
        - np.sum(np.log(np.diag(factor[0])))
        - 0.5 * times.size * _LOG_2PI
    )


def _conjugate_case(
    variance: float, lengthscale: float, noise_variance: float, num_points: int
) -> tuple[jax.Array, jax.Array, StateSpaceKernel, float, float]:
    """Return ``(times, observations, kernel, noise_std, exact log marginal likelihood)``."""
    times, observations = _wiggly_time_series(num_points)
    kernel = matern52_kernel(variance=variance, lengthscale=lengthscale)
    exact = _exact_log_marginal_likelihood(
        times, observations, variance, lengthscale, noise_variance
    )
    return (
        jnp.asarray(times),
        jnp.asarray(observations),
        kernel,
        float(np.sqrt(noise_variance)),
        exact,
    )


@pytest.mark.parametrize("num_points", _NUM_POINTS)
@pytest.mark.parametrize("noise_variance", _NOISE_VARIANCES)
@pytest.mark.parametrize("lengthscale", _LENGTHSCALES)
@pytest.mark.parametrize("variance", _VARIANCES)
def test_markov_vi_elbo_equals_the_exact_log_marginal_likelihood_for_a_gaussian_likelihood(
    variance: float, lengthscale: float, noise_variance: float, num_points: int
) -> None:
    """One natural-gradient step with unit step size makes the ELBO exact (Chang et al. eq. 11)."""
    times, observations, kernel, noise_std, exact = _conjugate_case(
        variance, lengthscale, noise_variance, num_points
    )
    state = fit_gaussian_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=noise_std,
        num_iterations=1,
    )
    np.testing.assert_almost_equal(float(state.evidence_lower_bound), exact, decimal=4)


@pytest.mark.parametrize("num_points", _NUM_POINTS)
@pytest.mark.parametrize("noise_variance", _NOISE_VARIANCES)
@pytest.mark.parametrize("lengthscale", _LENGTHSCALES)
@pytest.mark.parametrize("variance", _VARIANCES)
def test_markov_laplace_evidence_equals_the_exact_log_marginal_likelihood_for_a_gaussian_likelihood(
    variance: float, lengthscale: float, noise_variance: float, num_points: int
) -> None:
    """The Laplace evidence of JMLR eq. (17) is exact for a Gaussian likelihood."""
    times, observations, kernel, noise_std, exact = _conjugate_case(
        variance, lengthscale, noise_variance, num_points
    )
    state = fit_gaussian_markov_laplace_gp(
        times=times, observations=observations, state_space_kernel=kernel, noise_std=noise_std
    )
    np.testing.assert_almost_equal(float(state.log_marginal_likelihood), exact, decimal=4)


@pytest.mark.parametrize("power", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("noise_variance", _NOISE_VARIANCES)
@pytest.mark.parametrize("lengthscale", _LENGTHSCALES)
@pytest.mark.parametrize("variance", _VARIANCES)
def test_markov_pep_evidence_equals_the_exact_log_marginal_likelihood_for_a_gaussian_likelihood(
    variance: float, lengthscale: float, noise_variance: float, power: float
) -> None:
    """The power-EP energy of JMLR eq. (27) is exact for a Gaussian likelihood at every power."""
    times, observations, kernel, noise_std, exact = _conjugate_case(
        variance, lengthscale, noise_variance, 30
    )
    state = fit_gaussian_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=noise_std,
        power=power,
    )
    np.testing.assert_almost_equal(float(state.log_marginal_likelihood), exact, decimal=4)


def test_markov_laplace_evidence_equals_dense_laplace_evidence_for_a_bernoulli_likelihood() -> None:
    """At the Newton fixed point, JMLR eq. (17) equals Rasmussen and Williams eq. (3.32)."""
    times = jnp.linspace(0.0, 6.0, 40)
    labels = binary_labels(times)
    lengthscale = 0.7
    state = fit_bernoulli_markov_laplace_gp(
        times=times,
        observations=labels,
        state_space_kernel=matern32_kernel(variance=1.0, lengthscale=lengthscale),
        num_iterations=50,
    )
    mode = np.asarray(state.smoothed_means)
    labels_np = np.asarray(labels)
    sigmoid = 1.0 / (1.0 + np.exp(-mode))
    curvature = sigmoid * (1.0 - sigmoid)
    points = times[:, None]
    gram = np.asarray(
        dense_matern32_kernel(points, points, lengthscale=lengthscale, output_scale=1.0)
    )
    sqrt_curvature = np.sqrt(curvature)
    b_matrix = np.eye(mode.size) + sqrt_curvature[:, None] * gram * sqrt_curvature[None, :]
    log_likelihood = float(np.sum(-np.logaddexp(0.0, -labels_np * mode)))
    dense_laplace = (
        -0.5 * mode @ np.linalg.solve(gram, mode)
        + log_likelihood
        - 0.5 * np.linalg.slogdet(b_matrix)[1]
    )
    np.testing.assert_almost_equal(float(state.log_marginal_likelihood), dense_laplace, decimal=4)
