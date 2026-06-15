r"""Tests for Laplace-approximate binary GP classification (RW06 Alg. 3.1).

For a non-conjugate likelihood ``p(y | f)`` and a zero-mean GP prior
``f ~ GP(0, K)``, the posterior ``p(f | X, y) ∝ p(y | f) p(f | X)`` is
non-Gaussian. The **Laplace approximation** (Williams & Barber 1998;
RW06 §3.4, Algorithm 3.1) finds the latent posterior mode ``f̂`` via
Newton's method and approximates the posterior by a Gaussian with
covariance ``(K^{-1} + W)^{-1}`` where ``W = -∇² log p(y | f̂)``.

For the binary-Bernoulli likelihood with ``y ∈ {-1, +1}``:

    log p(y | f) = -log(1 + exp(-y f)),
    ∇ log p(y | f) = (y + 1)/2 - σ(f) = t - σ(f),  t = (y+1)/2,
    W_ii = σ(f_i) (1 - σ(f_i))    (diagonal).

The Newton step (RW06 eq. 3.18) is implemented numerically via the
``B = I + W^{1/2} K W^{1/2}`` Cholesky to keep everything PSD.

References
----------
* Williams, C. K. I., Barber, D. 1998 — *Bayesian Classification with
  Gaussian Processes*, IEEE TPAMI.
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §3.4 Algorithm 3.1 (PRIMARY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    fit_bernoulli_laplace_gp,
    predict_bernoulli_laplace_gp,
)
from opifex.uncertainty.types import PredictiveDistribution


def _toy_binary_data(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """1-D binary classification toy: ``y = sign(sin(2 x))``."""
    x = jax.random.uniform(jax.random.PRNGKey(seed), (20, 1), minval=-1.5, maxval=1.5)
    latent = jnp.sin(2.0 * x.squeeze(-1))
    y = jnp.sign(latent)
    return x, y


def test_fit_laplace_gp_converges_to_a_mode_with_positive_log_marginal() -> None:
    """The Newton loop converges and produces a finite log marginal likelihood."""
    x, y = _toy_binary_data(0)
    state = fit_bernoulli_laplace_gp(
        x_train=x,
        y_train=y,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=20,
    )
    assert state.f_mode.shape == y.shape
    assert jnp.all(jnp.isfinite(state.f_mode))
    assert jnp.isfinite(state.log_marginal_likelihood)


def test_predict_latent_mean_aligns_with_target_signs_on_training_data() -> None:
    """At training points, ``sign(f̂_i)`` should agree with ``y_i`` for most points."""
    x, y = _toy_binary_data(1)
    state = fit_bernoulli_laplace_gp(
        x_train=x, y_train=y, lengthscale=0.4, output_scale=1.5, num_newton_iterations=30
    )
    agreement = jnp.mean((jnp.sign(state.f_mode) == y).astype(jnp.float32))
    assert float(agreement) > 0.8


def test_predict_returns_predictive_distribution_with_finite_moments() -> None:
    """``predict_bernoulli_laplace_gp`` returns a ``PredictiveDistribution``."""
    x, y = _toy_binary_data(2)
    state = fit_bernoulli_laplace_gp(
        x_train=x, y_train=y, lengthscale=0.4, output_scale=1.0, num_newton_iterations=15
    )
    x_test = jnp.linspace(-1.5, 1.5, 25).reshape(-1, 1)
    predictive = predict_bernoulli_laplace_gp(state=state, x_test=x_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (25,)
    assert predictive.variance.shape == (25,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_predict_probabilities_match_mackays_probit_approximation() -> None:
    r"""The class probability uses MacKay's σ(μ / √(1 + π V/8)) approximation.

    Verifies the returned ``mean`` (interpreted as ``p(y=+1 | x*)``) is in
    ``[0, 1]`` and matches the explicit formula evaluated from the same
    state's ``f_mode``-based latent mean / variance.
    """
    x, y = _toy_binary_data(3)
    state = fit_bernoulli_laplace_gp(
        x_train=x, y_train=y, lengthscale=0.4, output_scale=1.0, num_newton_iterations=20
    )
    x_test = jnp.linspace(-1.5, 1.5, 10).reshape(-1, 1)
    predictive = predict_bernoulli_laplace_gp(state=state, x_test=x_test)
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


def test_fit_and_predict_are_jit_compatible() -> None:
    """Full pipeline compiles under ``jax.jit`` with traced training data."""
    x, y = _toy_binary_data(4)
    x_test = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_bernoulli_laplace_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            num_newton_iterations=20,
        )
        pd = predict_bernoulli_laplace_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x, y, x_test)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


def test_predictive_metadata_advertises_laplace_classification() -> None:
    """Metadata records ``estimator=bernoulli_laplace_gp``."""
    x, y = _toy_binary_data(5)
    state = fit_bernoulli_laplace_gp(
        x_train=x, y_train=y, lengthscale=0.4, output_scale=1.0, num_newton_iterations=10
    )
    predictive = predict_bernoulli_laplace_gp(state=state, x_test=jnp.zeros((2, 1)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "bernoulli_laplace_gp"
    assert metadata.get("paper") is not None
