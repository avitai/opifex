r"""Tests for generic non-conjugate Laplace GP and Poisson/Student-t/Beta wrappers.

The Laplace approximation (RW06 §3.4, Algorithm 3.1) generalises to any
factorising likelihood ``p(y | f) = Π_i p(y_i | f_i)`` whose
``(log_lik, ∇log_lik, W = -∇²log_lik)`` triple is closed-form. Task
11.1 D5 refactors the binary-Bernoulli classifier into a generic
``fit_laplace_gp`` and adds Poisson (log-link) for count regression,
Student-t for robust regression, and Beta (logit-link) for proportion
regression.

Each per-likelihood ``(log_lik, grad, W)`` block is verified against the
canonical formulas in:

* Rasmussen & Williams 2006 §3.4 (Bernoulli + generic algorithm).
* bayesnewton/bayesnewton/likelihoods.py:Poisson (line 891).
* bayesnewton/bayesnewton/likelihoods.py:StudentsT (line 1011).
* bayesnewton/bayesnewton/likelihoods.py:Beta (line 1047).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    fit_bernoulli_laplace_gp,
    fit_beta_laplace_gp,
    fit_laplace_gp,
    fit_poisson_laplace_gp,
    fit_studentst_laplace_gp,
    LaplaceGPState,
    predict_beta_laplace_gp,
    predict_poisson_laplace_gp,
    predict_studentst_laplace_gp,
)
from opifex.uncertainty.gp.laplace_classification import (
    _bernoulli_log_likelihood_components,
)
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Generic core: regression against existing Bernoulli surface
# -----------------------------------------------------------------------------


def _toy_binary_data(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """1-D binary classification toy: ``y = sign(sin(2 x))``."""
    x = jax.random.uniform(jax.random.PRNGKey(seed), (20, 1), minval=-1.5, maxval=1.5)
    latent = jnp.sin(2.0 * x.squeeze(-1))
    y = jnp.sign(latent)
    return x, y


def test_generic_fit_laplace_gp_matches_bernoulli_specialisation() -> None:
    """``fit_laplace_gp`` with the Bernoulli components matches the wrapper exactly."""
    x, y = _toy_binary_data(0)
    generic_state = fit_laplace_gp(
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=20,
    )
    bernoulli_state = fit_bernoulli_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=20,
    )
    assert isinstance(generic_state, LaplaceGPState)
    assert isinstance(bernoulli_state, LaplaceGPState)
    assert jnp.allclose(generic_state.f_mode, bernoulli_state.f_mode, atol=1e-5)
    assert jnp.allclose(
        generic_state.log_marginal_likelihood,
        bernoulli_state.log_marginal_likelihood,
        atol=1e-5,
    )


# -----------------------------------------------------------------------------
# Poisson likelihood (exp link)
# -----------------------------------------------------------------------------


def _toy_poisson_data(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """1-D Poisson regression toy: ``λ(x) = exp(sin(2x) + 1)``."""
    key = jax.random.PRNGKey(seed)
    key_x, key_y = jax.random.split(key)
    x = jax.random.uniform(key_x, (25, 1), minval=-1.5, maxval=1.5)
    rate = jnp.exp(jnp.sin(2.0 * x.squeeze(-1)) + 1.0)
    y = jax.random.poisson(key_y, rate).astype(jnp.float32)
    return x, y


def test_fit_poisson_laplace_gp_finds_a_finite_mode_on_count_data() -> None:
    """Newton converges to a finite mode and finite log marginal likelihood."""
    x, y = _toy_poisson_data(0)
    state = fit_poisson_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=30,
    )
    assert state.f_mode.shape == y.shape
    assert jnp.all(jnp.isfinite(state.f_mode))
    assert jnp.isfinite(state.log_marginal_likelihood)


def test_poisson_laplace_gp_mode_tracks_log_observed_counts() -> None:
    """``exp(f̂_i) ≈ y_i`` so ``f̂_i`` correlates positively with ``log(y_i + 1)``."""
    x, y = _toy_poisson_data(1)
    state = fit_poisson_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.4,
        output_scale=1.5,
        num_newton_iterations=30,
    )
    log_y = jnp.log(y + 1.0)
    correlation = jnp.corrcoef(state.f_mode, log_y)[0, 1]
    assert float(correlation) > 0.5


def test_predict_poisson_laplace_gp_returns_positive_intensity() -> None:
    """Predicted Poisson intensity ``E[λ(x*)]`` is strictly positive."""
    x, y = _toy_poisson_data(2)
    state = fit_poisson_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=30,
    )
    x_test = jnp.linspace(-1.5, 1.5, 25).reshape(-1, 1)
    predictive = predict_poisson_laplace_gp(state=state, x_test=x_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert jnp.all(predictive.mean > 0.0)
    assert jnp.all(predictive.variance > 0.0)
    assert jnp.all(jnp.isfinite(predictive.mean))


def test_poisson_laplace_gp_full_pipeline_is_jit_compatible() -> None:
    """Full fit + predict compiles under ``jax.jit`` with traced training data."""
    x, y = _toy_poisson_data(3)
    x_test = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_poisson_laplace_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            num_newton_iterations=20,
        )
        predictive = predict_poisson_laplace_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x, y, x_test)
    assert out.shape == (8,)
    assert jnp.all(jnp.isfinite(out))


def test_poisson_laplace_predictive_metadata_records_likelihood() -> None:
    """Metadata advertises ``estimator=poisson_laplace_gp`` and ``link=exp``."""
    x, y = _toy_poisson_data(4)
    state = fit_poisson_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.4,
        output_scale=1.0,
        num_newton_iterations=15,
    )
    predictive = predict_poisson_laplace_gp(state=state, x_test=jnp.zeros((3, 1)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "poisson_laplace_gp"
    assert metadata.get("likelihood") == "poisson"
    assert metadata.get("link") == "exp"


# -----------------------------------------------------------------------------
# Student-t likelihood (robust regression)
# -----------------------------------------------------------------------------


def _toy_robust_data(seed: int = 0, *, outlier_count: int = 2) -> tuple[jax.Array, jax.Array]:
    """1-D regression toy with sinusoidal mean and a handful of outliers."""
    key = jax.random.PRNGKey(seed)
    key_noise, key_outlier_idx = jax.random.split(key, 2)
    x = jnp.linspace(-1.5, 1.5, 25).reshape(-1, 1)
    clean = jnp.sin(2.0 * x.squeeze(-1))
    noise = 0.05 * jax.random.normal(key_noise, (25,))
    y = clean + noise
    outlier_idx = jax.random.choice(
        key_outlier_idx, jnp.arange(25), shape=(outlier_count,), replace=False
    )
    y = y.at[outlier_idx].set(jnp.array([3.0, -3.0])[:outlier_count])
    return x, y


def test_fit_studentst_laplace_gp_finds_a_finite_mode() -> None:
    """Student-t Newton with W-clipping converges to a finite mode."""
    x, y = _toy_robust_data(0)
    state = fit_studentst_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        df=4.0,
        scale=0.1,
        num_newton_iterations=40,
    )
    assert state.f_mode.shape == y.shape
    assert jnp.all(jnp.isfinite(state.f_mode))
    assert jnp.isfinite(state.log_marginal_likelihood)


def test_studentst_laplace_gp_mode_is_robust_to_outliers() -> None:
    """At outlier indices the mode pulls toward the trend rather than the spike."""
    x, y = _toy_robust_data(1, outlier_count=2)
    state = fit_studentst_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.4,
        output_scale=1.0,
        df=3.0,
        scale=0.5,
        num_newton_iterations=50,
    )
    # The trend ``sin(2 x)`` is in [-1, 1]; an outlier of ±3 should be
    # damped by the Student-t loss — the mode magnitude at every index
    # should stay well below ``|y_outlier|``.
    assert jnp.max(jnp.abs(state.f_mode)) < 2.5


def test_predict_studentst_laplace_gp_returns_finite_moments() -> None:
    """Predict returns finite mean / variance at new points."""
    x, y = _toy_robust_data(2)
    state = fit_studentst_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.4,
        output_scale=1.0,
        df=4.0,
        scale=0.1,
        num_newton_iterations=30,
    )
    x_test = jnp.linspace(-1.5, 1.5, 15).reshape(-1, 1)
    predictive = predict_studentst_laplace_gp(state=state, x_test=x_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (15,)
    assert predictive.variance.shape == (15,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_studentst_laplace_gp_pipeline_is_jit_compatible() -> None:
    """Full fit + predict compiles under ``jax.jit``."""
    x, y = _toy_robust_data(3)
    x_test = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_studentst_laplace_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            df=4.0,
            scale=0.1,
            num_newton_iterations=20,
        )
        predictive = predict_studentst_laplace_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x, y, x_test)
    assert out.shape == (6,)
    assert jnp.all(jnp.isfinite(out))


def test_studentst_laplace_predictive_metadata_records_likelihood() -> None:
    """Metadata advertises ``estimator=studentst_laplace_gp``."""
    x, y = _toy_robust_data(4)
    state = fit_studentst_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        df=4.0,
        scale=0.1,
        num_newton_iterations=15,
    )
    predictive = predict_studentst_laplace_gp(state=state, x_test=jnp.zeros((2, 1)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "studentst_laplace_gp"
    assert metadata.get("likelihood") == "students_t"


# -----------------------------------------------------------------------------
# Beta likelihood (proportion regression, logit link)
# -----------------------------------------------------------------------------


def _toy_beta_data(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """1-D proportion regression toy: ``y(x) = sigmoid(sin(2x))``."""
    key = jax.random.PRNGKey(seed)
    key_x, key_noise = jax.random.split(key)
    x = jax.random.uniform(key_x, (25, 1), minval=-1.5, maxval=1.5)
    mean = jax.nn.sigmoid(jnp.sin(2.0 * x.squeeze(-1)))
    # Sample y from a Beta with mean ``mean`` and scale ``s = 20`` for low variance.
    scale = 20.0
    alpha = mean * scale
    beta = scale * (1.0 - mean)
    y = jax.random.beta(key_noise, alpha, beta)
    return x, y


def test_fit_beta_laplace_gp_finds_a_finite_mode() -> None:
    """Newton on Beta likelihood with Fisher-info ``W`` converges."""
    x, y = _toy_beta_data(0)
    state = fit_beta_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        scale=20.0,
        num_newton_iterations=30,
    )
    assert state.f_mode.shape == y.shape
    assert jnp.all(jnp.isfinite(state.f_mode))
    assert jnp.isfinite(state.log_marginal_likelihood)


def test_predict_beta_laplace_gp_returns_proportions_in_unit_interval() -> None:
    """Predicted ``E[y* | x*] ∈ (0, 1)`` and variance > 0."""
    x, y = _toy_beta_data(1)
    state = fit_beta_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.4,
        output_scale=1.0,
        scale=20.0,
        num_newton_iterations=30,
    )
    x_test = jnp.linspace(-1.5, 1.5, 20).reshape(-1, 1)
    predictive = predict_beta_laplace_gp(state=state, x_test=x_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)
    assert jnp.all(predictive.variance > 0.0)


def test_beta_laplace_gp_pipeline_is_jit_compatible() -> None:
    """Full fit + predict compiles under ``jax.jit``."""
    x, y = _toy_beta_data(2)
    x_test = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_beta_laplace_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            scale=20.0,
            num_newton_iterations=20,
        )
        predictive = predict_beta_laplace_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x, y, x_test)
    assert out.shape == (6,)
    assert jnp.all(jnp.isfinite(out))


def test_beta_laplace_predictive_metadata_records_likelihood() -> None:
    """Metadata advertises ``estimator=beta_laplace_gp`` and ``link=logit``."""
    x, y = _toy_beta_data(3)
    state = fit_beta_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        scale=20.0,
        num_newton_iterations=15,
    )
    predictive = predict_beta_laplace_gp(state=state, x_test=jnp.zeros((2, 1)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "beta_laplace_gp"
    assert metadata.get("likelihood") == "beta"
    assert metadata.get("link") == "logit"
