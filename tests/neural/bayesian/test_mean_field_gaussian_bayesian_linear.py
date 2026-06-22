"""Bayesian-linear predictive / ELBO surface of :class:`MeanFieldGaussian`.

A mean-field Gaussian posterior over a weight vector ``w in R^num_params`` is,
on its own, a Bayesian linear model (Bishop, *PRML* 3.3): with input ``x`` the
prediction is ``f(x) = w . x`` and, because ``q(w) = N(mu, diag(sigma^2))`` is
Gaussian and the map is linear, the predictive ``f(x) ~ N(mu . x, sum_i x_i^2
sigma_i^2)`` is available in closed form. These tests pin that closed form
against a Monte-Carlo estimate, check the optimizer-facing ELBO decomposition,
and confirm the surfaces are ``jit`` / ``grad`` / ``vmap`` clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.variational_framework import MeanFieldGaussian
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution


def _objective(*, dataset_size: int | None = None) -> ObjectiveConfig:
    """Minimal data + KL objective (all other weights zero)."""
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=dataset_size,
        physics_weight=0.0,
        data_weight=1.0,
        boundary_weight=0.0,
        initial_condition_weight=0.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=0.0,
    )


def _layer(num_params: int = 4, *, seed: int = 0) -> MeanFieldGaussian:
    """Build a posterior with non-trivial mean/log_std so predictions vary."""
    layer = MeanFieldGaussian(num_params=num_params, rngs=nnx.Rngs(seed))
    layer.mean[...] = jnp.linspace(-1.0, 1.0, num_params)
    layer.log_std[...] = jnp.linspace(-1.5, -0.5, num_params)
    return layer


class TestPredictDistribution:
    """Closed-form Bayesian-linear predictive distribution."""

    def test_shapes_and_validate(self) -> None:
        """Predictive fields are all ``(batch,)`` and pass invariant checks."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(1), (8, 4))

        predictive = layer.predict_distribution(x)

        assert isinstance(predictive, PredictiveDistribution)
        assert predictive.mean.shape == (8,)
        assert predictive.epistemic is not None and predictive.epistemic.shape == (8,)
        assert predictive.aleatoric is not None and predictive.aleatoric.shape == (8,)
        predictive.validate()  # total_uncertainty == epistemic + aleatoric, shapes match

    def test_closed_form_matches_monte_carlo(self) -> None:
        """``mean``/``epistemic`` match a sampled weight-posterior estimate."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(2), (5, 4))

        predictive = layer.predict_distribution(x)

        # Reference: draw weights from q(w), push through the linear map.
        weights = layer.sample(20_000, rngs=nnx.Rngs(sample=3))  # (S, num_params)
        sampled = weights @ x.T  # (S, batch)
        mc_mean = jnp.mean(sampled, axis=0)
        mc_var = jnp.var(sampled, axis=0)

        assert jnp.allclose(predictive.mean, mc_mean, atol=2e-2)
        assert predictive.epistemic is not None
        assert jnp.allclose(predictive.epistemic, mc_var, rtol=5e-2, atol=2e-3)

    def test_rejects_wrong_feature_dim(self) -> None:
        """A feature dimension other than ``num_params`` is a hard error."""
        layer = _layer(num_params=4)
        with pytest.raises(ValueError, match="num_params"):
            layer.predict_distribution(jnp.ones((3, 5)))


class TestElboSurface:
    """``loss_components`` / ``negative_elbo`` optimizer-facing surface."""

    def test_loss_components_populates_nll_and_kl(self) -> None:
        """The decomposition carries a finite NLL, KL, and total."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(4), (16, 4))
        y = x @ jnp.array([0.5, -0.5, 1.0, 0.0])

        components = layer.loss_components({"x": x, "y": y}, config=_objective())

        assert isinstance(components, UQLossComponents)
        assert components.negative_log_likelihood is not None
        assert components.kl is not None
        assert bool(jnp.isfinite(components.total))
        components.validate()

    def test_total_is_weighted_nll_plus_scaled_kl(self) -> None:
        """``total`` equals ``data_weight * nll + kl / N`` (per-example ELBO)."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(5), (16, 4))
        y = x @ jnp.array([0.5, -0.5, 1.0, 0.0])
        config = _objective(dataset_size=16)

        components = layer.loss_components({"x": x, "y": y}, config=config)

        assert components.negative_log_likelihood is not None and components.kl is not None
        expected = components.negative_log_likelihood + components.kl / 16.0
        assert jnp.allclose(components.total, expected)

    def test_negative_elbo_is_scalar_total(self) -> None:
        """``negative_elbo`` returns the scalar objective (== ``total``)."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(6), (16, 4))
        y = x @ jnp.array([0.5, -0.5, 1.0, 0.0])
        config = _objective(dataset_size=16)

        scalar = layer.negative_elbo({"x": x, "y": y}, config=config)
        components = layer.loss_components({"x": x, "y": y}, config=config)

        assert scalar.shape == ()
        assert jnp.allclose(scalar, components.total)

    def test_missing_batch_field_raises(self) -> None:
        """A batch without ``y`` is rejected before any compute."""
        layer = _layer()
        with pytest.raises(ValueError, match="y"):
            layer.loss_components({"x": jnp.ones((2, 4))}, config=_objective())


class TestTransformCompatibility:
    """The ELBO must survive ``jit(grad(...))`` and ``vmap``."""

    def test_jit_grad_of_negative_elbo_is_finite(self) -> None:
        """Gradient of the negative ELBO wrt variational params is finite."""
        layer = _layer()
        x = jax.random.normal(jax.random.key(7), (16, 4))
        y = x @ jnp.array([0.5, -0.5, 1.0, 0.0])
        config = _objective(dataset_size=16)

        @nnx.jit
        def grad_norm(model: MeanFieldGaussian) -> jax.Array:
            def objective(m: MeanFieldGaussian) -> jax.Array:
                return m.negative_elbo({"x": x, "y": y}, config=config)

            grads = nnx.grad(objective)(model)
            leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
            return sum((jnp.sum(leaf**2) for leaf in leaves), start=jnp.array(0.0))

        assert bool(jnp.isfinite(grad_norm(layer)))

    def test_vmap_predict_over_batch_of_inputs(self) -> None:
        """``predict_distribution`` vmaps over a stack of input batches."""
        layer = _layer()
        graphdef, state = nnx.split(layer)
        x = jax.random.normal(jax.random.key(8), (3, 5, 4))  # (outer, batch, feat)

        def predict_mean(xi: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state).predict_distribution(xi).mean

        means = jax.vmap(predict_mean)(x)
        assert means.shape == (3, 5)
        assert bool(jnp.all(jnp.isfinite(means)))


def test_kl_divergence_accepts_explicit_and_default_prior() -> None:
    """Regression: ``kl_divergence`` still works with and without prior args."""
    layer = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
    assert jnp.isfinite(layer.kl_divergence())
    assert jnp.isfinite(layer.kl_divergence(prior_mean=0.0, prior_std=1.0))
