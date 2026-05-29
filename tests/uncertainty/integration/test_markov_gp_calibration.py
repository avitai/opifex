r"""Calibration integration tests for the Markov-GP inference paths.

Closes the **Task 11.2 exit criterion** (*"PEP/VI/Laplace inference
paths with calibration tests"*). Each test simulates from a known
generative model and checks that the held-out predictive moments are
calibrated under the canonical metric for that likelihood family:

* **Gaussian regression** — *Prediction Interval Coverage Probability*
  (``picp``) at the nominal 90% level should land in [0.80, 1.00] for
  every Markov-GP inference path (a conservative band, allowing for
  finite-sample noise on n=40).
* **Bernoulli classification** — *Brier score* < 0.25 (a constant
  random predictor scores 0.25; non-trivial signal must beat it) and
  *Expected Calibration Error* (ECE) < 0.20 across the four paths.
* **Poisson count regression** — predictive log-likelihood of held-out
  counts under the log-normal intensity collapse must beat a constant
  intensity baseline (mean of training counts).

Each likelihood is exercised against all four inference paths that
ship in Task 11.2:

* Markov-Laplace (slice 25 / 26),
* Markov-VI (slice 27),
* Markov-PEP (slice 28),
* Markov-PL  (slice 30).

References
----------
* Kuleshov, Fenner, Ermon 2018 — *Accurate uncertainties for deep
  learning using calibrated regression*, ICML.
* Naeini, Cooper, Hauskrecht 2015 — *Obtaining well-calibrated
  probabilities using Bayesian binning*, AAAI.
* Gneiting, Raftery 2007 — *Strictly proper scoring rules*, JASA.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from opifex.uncertainty.calibration.base import brier_score, expected_calibration_error
from opifex.uncertainty.calibration.regression import picp
from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel


_PICP_LOWER = 0.80
_PICP_UPPER = 1.00
_BRIER_MAX = 0.25
_ECE_MAX = 0.20


# -----------------------------------------------------------------------------
# Gaussian regression — PICP at 90%
# -----------------------------------------------------------------------------


def _build_gaussian_truth(seed: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Smooth sinusoidal truth + Gaussian noise; returns train/test split."""
    key = jax.random.PRNGKey(seed)
    times = jnp.linspace(0.0, 4.0 * jnp.pi, 60)
    truth = jnp.sin(times) + 0.3 * jnp.cos(2.5 * times)
    noise = 0.1 * jax.random.normal(key, times.shape)
    observations = truth + noise
    train_indices = jnp.arange(0, 60, 2)
    test_indices = jnp.arange(1, 60, 2)
    return (
        times[train_indices],
        observations[train_indices],
        times[test_indices],
        observations[test_indices],
    )


def _gaussian_predictive_interval(
    *,
    predictive_mean: jax.Array,
    predictive_variance: jax.Array,
    level: float = 0.90,
) -> tuple[jax.Array, jax.Array]:
    r"""Symmetric ``level`` central-quantile band assuming Gaussian predictive."""
    z = jax.scipy.stats.norm.ppf(0.5 + 0.5 * level)
    std = jnp.sqrt(predictive_variance)
    return predictive_mean - z * std, predictive_mean + z * std


def test_markov_laplace_gaussian_picp_at_90_percent_is_well_calibrated() -> None:
    """Markov-Laplace Gaussian predict gives ~90% PICP at the 90% level."""
    from opifex.uncertainty.markov import fit_gaussian_markov_laplace_gp

    times_train, y_train, times_test, y_test = _build_gaussian_truth(seed=11)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_gaussian_markov_laplace_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        noise_std=0.1,
        num_iterations=5,
    )
    from opifex.uncertainty.markov import predict_gaussian_markov_laplace_gp

    predictive = predict_gaussian_markov_laplace_gp(
        state=state, times_test=times_test, noise_std=0.1
    )
    assert predictive.variance is not None
    lower, upper = _gaussian_predictive_interval(
        predictive_mean=predictive.mean,
        predictive_variance=predictive.variance,
    )
    coverage = picp(target=y_test, lower=lower, upper=upper)
    assert _PICP_LOWER <= float(coverage) <= _PICP_UPPER


def test_markov_vi_gaussian_picp_at_90_percent_is_well_calibrated() -> None:
    """Markov-VI Gaussian predict gives ~90% PICP at the 90% level."""
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_vi_gp,
        predict_gaussian_markov_vi_gp,
    )

    times_train, y_train, times_test, y_test = _build_gaussian_truth(seed=12)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_gaussian_markov_vi_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        noise_std=0.1,
        num_iterations=10,
    )
    predictive = predict_gaussian_markov_vi_gp(state=state, times_test=times_test, noise_std=0.1)
    assert predictive.variance is not None
    lower, upper = _gaussian_predictive_interval(
        predictive_mean=predictive.mean,
        predictive_variance=predictive.variance,
    )
    coverage = picp(target=y_test, lower=lower, upper=upper)
    assert _PICP_LOWER <= float(coverage) <= _PICP_UPPER


def test_markov_pep_gaussian_picp_at_90_percent_is_well_calibrated() -> None:
    """Markov-PEP Gaussian predict gives ~90% PICP at the 90% level."""
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_pep_gp,
        predict_gaussian_markov_pep_gp,
    )

    times_train, y_train, times_test, y_test = _build_gaussian_truth(seed=13)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_gaussian_markov_pep_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        noise_std=0.1,
        power=1.0,
        num_iterations=5,
        learning_rate=1.0,
    )
    predictive = predict_gaussian_markov_pep_gp(state=state, times_test=times_test, noise_std=0.1)
    assert predictive.variance is not None
    lower, upper = _gaussian_predictive_interval(
        predictive_mean=predictive.mean,
        predictive_variance=predictive.variance,
    )
    coverage = picp(target=y_test, lower=lower, upper=upper)
    assert _PICP_LOWER <= float(coverage) <= _PICP_UPPER


def test_markov_pl_gaussian_picp_at_90_percent_is_well_calibrated() -> None:
    """Markov-PL Gaussian predict gives ~90% PICP at the 90% level."""
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_pl_gp,
        predict_gaussian_markov_pl_gp,
    )

    times_train, y_train, times_test, y_test = _build_gaussian_truth(seed=14)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_gaussian_markov_pl_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        noise_std=0.1,
        num_iterations=3,
    )
    predictive = predict_gaussian_markov_pl_gp(state=state, times_test=times_test, noise_std=0.1)
    assert predictive.variance is not None
    lower, upper = _gaussian_predictive_interval(
        predictive_mean=predictive.mean,
        predictive_variance=predictive.variance,
    )
    coverage = picp(target=y_test, lower=lower, upper=upper)
    assert _PICP_LOWER <= float(coverage) <= _PICP_UPPER


# -----------------------------------------------------------------------------
# Bernoulli classification — Brier + ECE
# -----------------------------------------------------------------------------


def _build_bernoulli_truth(
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Logit-link binary truth: latent ``2 sin(t)``, ±1 labels.

    Returns ``(times_train, y_train_pm1, times_test, y_test_pm1, p_true_test)``
    where the last array is the held-out generative probability ``p(y=+1|t)``.
    """
    key = jax.random.PRNGKey(seed)
    times = jnp.linspace(0.0, 4.0 * jnp.pi, 200)
    latent = 2.0 * jnp.sin(times)
    p_true = jax.nn.sigmoid(latent)
    samples = jax.random.bernoulli(key, p_true).astype(jnp.float32)
    pm1_labels = 2.0 * samples - 1.0
    train_indices = jnp.arange(0, 200, 2)
    test_indices = jnp.arange(1, 200, 2)
    return (
        times[train_indices],
        pm1_labels[train_indices],
        times[test_indices],
        pm1_labels[test_indices],
        p_true[test_indices],
    )


def _bernoulli_calibration_metrics(
    *, p_test_predicted: jax.Array, y_test_pm1: jax.Array
) -> tuple[float, float]:
    """``(Brier, ECE)`` against {0, 1} targets converted from ±1 labels."""
    targets_binary = (y_test_pm1 > 0.0).astype(jnp.float32)
    brier = float(brier_score(probabilities=p_test_predicted, targets=targets_binary))
    ece = float(
        expected_calibration_error(
            probabilities=p_test_predicted,
            targets=targets_binary,
        )
    )
    return brier, ece


BernoulliFitPredict = Callable[
    [jax.Array, jax.Array, jax.Array],
    jax.Array,
]
"""Fit-then-predict signature returning held-out class probabilities."""


def _bernoulli_fit_predict_markov_laplace(
    times_train: jax.Array, y_train: jax.Array, times_test: jax.Array
) -> jax.Array:
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_laplace_gp,
        predict_bernoulli_markov_laplace_gp,
    )

    kernel = state_space_matern32_kernel(variance=1.5, lengthscale=0.7)
    state = fit_bernoulli_markov_laplace_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    return predict_bernoulli_markov_laplace_gp(state=state, times_test=times_test).mean


def _bernoulli_fit_predict_markov_vi(
    times_train: jax.Array, y_train: jax.Array, times_test: jax.Array
) -> jax.Array:
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_vi_gp,
        predict_bernoulli_markov_vi_gp,
    )

    kernel = state_space_matern32_kernel(variance=1.5, lengthscale=0.7)
    state = fit_bernoulli_markov_vi_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=25,
    )
    return predict_bernoulli_markov_vi_gp(state=state, times_test=times_test).mean


def _bernoulli_fit_predict_markov_pep(
    times_train: jax.Array, y_train: jax.Array, times_test: jax.Array
) -> jax.Array:
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_pep_gp,
        predict_bernoulli_markov_pep_gp,
    )

    kernel = state_space_matern32_kernel(variance=1.5, lengthscale=0.7)
    state = fit_bernoulli_markov_pep_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        power=0.5,
        num_iterations=25,
        learning_rate=0.5,
    )
    return predict_bernoulli_markov_pep_gp(state=state, times_test=times_test).mean


def _bernoulli_fit_predict_markov_pl(
    times_train: jax.Array, y_train: jax.Array, times_test: jax.Array
) -> jax.Array:
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_pl_gp,
        predict_bernoulli_markov_pl_gp,
    )

    kernel = state_space_matern32_kernel(variance=1.5, lengthscale=0.7)
    state = fit_bernoulli_markov_pl_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    return predict_bernoulli_markov_pl_gp(state=state, times_test=times_test).mean


def test_markov_laplace_bernoulli_beats_constant_brier_and_ece_under_threshold() -> None:
    """Markov-Laplace Bernoulli predictions beat random + ECE < 0.20."""
    times_train, y_train, times_test, y_test, _ = _build_bernoulli_truth(seed=21)
    p_pred = _bernoulli_fit_predict_markov_laplace(times_train, y_train, times_test)
    brier, ece = _bernoulli_calibration_metrics(p_test_predicted=p_pred, y_test_pm1=y_test)
    assert brier < _BRIER_MAX
    assert ece < _ECE_MAX


def test_markov_vi_bernoulli_beats_constant_brier_and_ece_under_threshold() -> None:
    """Markov-VI Bernoulli predictions beat random + ECE < 0.20."""
    times_train, y_train, times_test, y_test, _ = _build_bernoulli_truth(seed=22)
    p_pred = _bernoulli_fit_predict_markov_vi(times_train, y_train, times_test)
    brier, ece = _bernoulli_calibration_metrics(p_test_predicted=p_pred, y_test_pm1=y_test)
    assert brier < _BRIER_MAX
    assert ece < _ECE_MAX


def test_markov_pep_bernoulli_beats_constant_brier_and_ece_under_threshold() -> None:
    """Markov-PEP Bernoulli predictions beat random + ECE < 0.20."""
    times_train, y_train, times_test, y_test, _ = _build_bernoulli_truth(seed=23)
    p_pred = _bernoulli_fit_predict_markov_pep(times_train, y_train, times_test)
    brier, ece = _bernoulli_calibration_metrics(p_test_predicted=p_pred, y_test_pm1=y_test)
    assert brier < _BRIER_MAX
    assert ece < _ECE_MAX


def test_markov_pl_bernoulli_beats_constant_brier_and_ece_under_threshold() -> None:
    """Markov-PL Bernoulli predictions beat random + ECE < 0.20."""
    times_train, y_train, times_test, y_test, _ = _build_bernoulli_truth(seed=24)
    p_pred = _bernoulli_fit_predict_markov_pl(times_train, y_train, times_test)
    brier, ece = _bernoulli_calibration_metrics(p_test_predicted=p_pred, y_test_pm1=y_test)
    assert brier < _BRIER_MAX
    assert ece < _ECE_MAX


# -----------------------------------------------------------------------------
# Poisson count regression — predictive log-likelihood beats constant baseline
# -----------------------------------------------------------------------------


def _build_poisson_truth(
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Smooth exp-link Poisson counts; train/test split with held-out ``y_test``."""
    key = jax.random.PRNGKey(seed)
    times = jnp.linspace(0.0, 4.0 * jnp.pi, 60)
    log_rate = jnp.sin(times) + 1.5
    rate = jnp.exp(log_rate)
    counts = jax.random.poisson(key, rate).astype(jnp.float32)
    train_indices = jnp.arange(0, 60, 2)
    test_indices = jnp.arange(1, 60, 2)
    return (
        times[train_indices],
        counts[train_indices],
        times[test_indices],
        counts[test_indices],
    )


def _poisson_log_likelihood_against_intensity(
    *, y_test: jax.Array, predicted_intensity: jax.Array
) -> jax.Array:
    r"""Sum of per-obs Poisson log-likelihoods at predicted intensity."""
    safe_intensity = jnp.maximum(predicted_intensity, 1e-6)
    return jnp.sum(y_test * jnp.log(safe_intensity) - safe_intensity - jax.lax.lgamma(y_test + 1.0))


def _poisson_baseline_log_likelihood(*, y_train: jax.Array, y_test: jax.Array) -> jax.Array:
    """Constant-intensity baseline: mean of training counts."""
    return _poisson_log_likelihood_against_intensity(
        y_test=y_test,
        predicted_intensity=jnp.full_like(y_test, jnp.mean(y_train)),
    )


def test_markov_laplace_poisson_beats_constant_intensity_baseline() -> None:
    """Markov-Laplace Poisson predictive log-lik beats constant baseline."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_laplace_gp,
        predict_poisson_markov_laplace_gp,
    )

    times_train, y_train, times_test, y_test = _build_poisson_truth(seed=31)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=1.0)
    state = fit_poisson_markov_laplace_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    predictive = predict_poisson_markov_laplace_gp(state=state, times_test=times_test)
    fitted_ll = _poisson_log_likelihood_against_intensity(
        y_test=y_test, predicted_intensity=predictive.mean
    )
    baseline_ll = _poisson_baseline_log_likelihood(y_train=y_train, y_test=y_test)
    assert float(fitted_ll) > float(baseline_ll)


def test_markov_vi_poisson_beats_constant_intensity_baseline() -> None:
    """Markov-VI Poisson predictive log-lik beats constant baseline."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_vi_gp,
        predict_poisson_markov_vi_gp,
    )

    times_train, y_train, times_test, y_test = _build_poisson_truth(seed=32)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=1.0)
    state = fit_poisson_markov_vi_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=25,
    )
    predictive = predict_poisson_markov_vi_gp(state=state, times_test=times_test)
    fitted_ll = _poisson_log_likelihood_against_intensity(
        y_test=y_test, predicted_intensity=predictive.mean
    )
    baseline_ll = _poisson_baseline_log_likelihood(y_train=y_train, y_test=y_test)
    assert float(fitted_ll) > float(baseline_ll)


def test_markov_pep_poisson_beats_constant_intensity_baseline() -> None:
    """Markov-PEP Poisson predictive log-lik beats constant baseline."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_pep_gp,
        predict_poisson_markov_pep_gp,
    )

    times_train, y_train, times_test, y_test = _build_poisson_truth(seed=33)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=1.0)
    state = fit_poisson_markov_pep_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        power=0.5,
        num_iterations=25,
        learning_rate=0.3,
    )
    predictive = predict_poisson_markov_pep_gp(state=state, times_test=times_test)
    fitted_ll = _poisson_log_likelihood_against_intensity(
        y_test=y_test, predicted_intensity=predictive.mean
    )
    baseline_ll = _poisson_baseline_log_likelihood(y_train=y_train, y_test=y_test)
    assert float(fitted_ll) > float(baseline_ll)


def test_markov_pl_poisson_beats_constant_intensity_baseline() -> None:
    """Markov-PL Poisson predictive log-lik beats constant baseline."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_pl_gp,
        predict_poisson_markov_pl_gp,
    )

    times_train, y_train, times_test, y_test = _build_poisson_truth(seed=34)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=1.0)
    state = fit_poisson_markov_pl_gp(
        times=times_train,
        observations=y_train,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    predictive = predict_poisson_markov_pl_gp(state=state, times_test=times_test)
    fitted_ll = _poisson_log_likelihood_against_intensity(
        y_test=y_test, predicted_intensity=predictive.mean
    )
    baseline_ll = _poisson_baseline_log_likelihood(y_train=y_train, y_test=y_test)
    assert float(fitted_ll) > float(baseline_ll)
