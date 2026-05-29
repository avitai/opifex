r"""Tests for the Titsias / collapsed-bound SVGP.

For a Gaussian likelihood with fixed kernel hyperparameters, Titsias
2009 (*Variational Learning of Inducing Variables in Sparse Gaussian
Processes*, AISTATS) derives the closed-form optimal variational
posterior ``q*(u) = N(μ*, S*)`` over the inducing values
``u = f(Z)``. The opifex implementation fits this in
``O(n m² + m³)`` time (where ``m << n`` is the inducing count) via
the GPJax ``CollapsedVariationalGaussian`` /
``collapsed_elbo`` recipe.

Key identities (RW06 §8.4 / Titsias 2009):

    A = L_z^{-1} K_zx / σ,    B = I + A A^T,    L_B = chol(B),
    μ(x*) = (L_z^{-1} K_zt)^T B^{-1} (A y / σ),
    Var(x*) = K(x*, x*) - ||L_z^{-1} K_zt||² + ||L_B^{-1} L_z^{-1} K_zt||²,
    log N(y; 0, σ² I + Q) = -n/2 log(2πσ²) - ½ log|B|
                            - 1/(2σ²) (||y||² - ||L_B^{-1} A y||²),
    collapsed ELBO = log N(y; 0, σ²I + Q)
                     - 1/(2σ²) [tr(K_xx)_diag - tr(A A^T)].

References
----------
* Titsias, M. K. 2009 — *Variational Learning of Inducing Variables in
  Sparse Gaussian Processes*, AISTATS (PRIMARY).
* Hensman, J., Fusi, N., Lawrence, N. D. 2013 — *Gaussian Processes
  for Big Data*, UAI (stochastic-optimisation extension).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    fit_exact_gp,
    fit_svgp,
    predict_exact_gp,
    predict_svgp,
    svgp_collapsed_elbo,
)
from opifex.uncertainty.types import PredictiveDistribution


def _toy_data() -> tuple[jax.Array, jax.Array]:
    x = jnp.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    y = jnp.sin(2.0 * x.squeeze(-1)) + 0.05 * jax.random.normal(
        jax.random.PRNGKey(0), x.squeeze(-1).shape
    )
    return x, y


def test_svgp_converges_to_exact_gp_as_inducing_set_covers_training_inputs() -> None:
    """When the inducing inputs equal the training inputs, SVGP ≈ exact GP."""
    x_train, y_train = _toy_data()
    x_inducing = x_train
    lengthscale, output_scale, noise_std = 0.4, 1.0, 0.05

    state_exact = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )
    state_svgp = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )
    x_test = jnp.linspace(-0.8, 0.8, 7).reshape(-1, 1)
    exact_predictive = predict_exact_gp(state=state_exact, x_test=x_test)
    svgp_predictive = predict_svgp(state=state_svgp, x_test=x_test)
    assert svgp_predictive.variance is not None
    assert exact_predictive.variance is not None
    assert jnp.allclose(svgp_predictive.mean, exact_predictive.mean, atol=1e-3)
    # SVGP variance ≥ exact variance (Titsias bound is conservative); equal
    # when Z = X, modulo numerical jitter.
    assert jnp.all(svgp_predictive.variance >= exact_predictive.variance - 1e-3)


def test_svgp_predictive_recovers_the_target_within_a_few_noise_scales() -> None:
    """At training points, SVGP predictive mean stays within a few ``σ``."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    predictive = predict_svgp(state=state, x_test=x_train)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.1


def test_svgp_predictive_variance_is_strictly_positive_at_test_points() -> None:
    """The collapsed predictive variance is positive everywhere."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    x_test = jnp.linspace(-1.5, 1.5, 20).reshape(-1, 1)
    predictive = predict_svgp(state=state, x_test=x_test)
    assert predictive.variance is not None
    assert jnp.all(predictive.variance > 0.0)


def test_svgp_collapsed_elbo_is_finite_scalar() -> None:
    """The collapsed ELBO is a finite scalar."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    elbo = svgp_collapsed_elbo(state=state)
    assert jnp.isfinite(elbo)


def test_svgp_collapsed_elbo_improves_with_more_inducing_points() -> None:
    """More inducing points → tighter ELBO (closer to exact log marginal)."""
    x_train, y_train = _toy_data()

    def elbo_with(m: int) -> float:
        x_inducing = jnp.linspace(-1.0, 1.0, m).reshape(-1, 1)
        state = fit_svgp(
            x_train=x_train,
            y_train=y_train,
            x_inducing=x_inducing,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.05,
        )
        return float(svgp_collapsed_elbo(state=state))

    assert elbo_with(15) > elbo_with(4)


def test_svgp_fit_and_predict_are_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    x_test = jnp.linspace(-0.5, 0.5, 4).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, z: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_svgp(
            x_train=x_t,
            y_train=y_t,
            x_inducing=z,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.05,
        )
        pd = predict_svgp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x_train, y_train, x_inducing, x_test)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


def test_svgp_metadata_advertises_titsias_estimator() -> None:
    """Metadata records ``estimator=titsias_collapsed_svgp``."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    predictive = predict_svgp(state=state, x_test=x_train)
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "titsias_collapsed_svgp"


def test_fit_svgp_rejects_nonpositive_noise_std() -> None:
    """``noise_std`` must be strictly positive."""
    x_train, y_train = _toy_data()
    x_inducing = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    with pytest.raises(ValueError, match="noise_std"):
        fit_svgp(
            x_train=x_train,
            y_train=y_train,
            x_inducing=x_inducing,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.0,
        )
