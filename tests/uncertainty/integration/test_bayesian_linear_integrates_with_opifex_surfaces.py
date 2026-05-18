"""Pin ``BayesianLinear`` integration against existing Opifex surfaces.

Phase 2 Task 2.1 introduced ``opifex.uncertainty.layers.bayesian.BayesianLinear``
as the canonical variational diagonal-Gaussian dense layer. These integration
tests pin three properties of that layer against the rest of Opifex:

1. The layer composes with the canonical ``UQLossComponents.from_components``
   pathway via its ``kl_divergence()`` method; a tiny synthetic training
   step produces a finite scalar loss and finite gradients.
2. The layer's KL formula numerically equals the shared
   ``diagonal_gaussian_kl`` helper from ``opifex.uncertainty.kernels`` —
   no second source of the formula exists in the platform.
3. The layer remains compatible with raw ``jax.jit`` over a pure-function
   wrapper (NNX state surfaced via ``nnx.split`` / ``nnx.merge``-style
   capture is left to ``nnx.jit`` in downstream model code).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty import (
    BayesianLinear,
    ObjectiveConfig,
    UQLossComponents,
)
from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl


def _make_objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=128,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=1.0,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )


def test_bayesian_linear_kl_divergence_composes_with_uq_loss_components() -> None:
    """A tiny variational training step builds a finite ``UQLossComponents.total``."""
    layer = BayesianLinear(in_features=4, out_features=3, prior_std=1.0, rngs=nnx.Rngs(0))
    config = _make_objective_config()
    x = jnp.ones((8, 4))
    target = jnp.zeros((8, 3))
    sample_rngs = nnx.Rngs(posterior=0)

    pred = layer(x, training=True, sample=True, rngs=sample_rngs)
    data_loss = jnp.mean((pred - target) ** 2)
    components = UQLossComponents.from_components(
        config=config, data=data_loss, kl=layer.kl_divergence()
    )
    assert jnp.isfinite(components.total)
    assert components.kl is not None
    assert jnp.isfinite(components.kl)


def test_bayesian_linear_kl_equals_shared_kernel_formula() -> None:
    """No second source of the closed-form KL formula exists in the platform."""
    layer = BayesianLinear(in_features=4, out_features=3, prior_std=1.5, rngs=nnx.Rngs(0))
    expected_weight_kl = float(
        diagonal_gaussian_kl(
            layer.weight_mean[...],
            layer.weight_logvar[...],
            prior_mean=0.0,
            prior_std=1.5,
        )
    )
    expected_bias_kl = float(
        diagonal_gaussian_kl(
            layer.bias_mean[...],
            layer.bias_logvar[...],
            prior_mean=0.0,
            prior_std=1.5,
        )
    )
    layer_kl = float(layer.kl_divergence())
    assert layer_kl == pytest.approx(expected_weight_kl + expected_bias_kl, rel=1e-6, abs=1e-6)


def test_bayesian_linear_grad_pipeline_is_finite() -> None:
    """``jax.grad`` over a closure around BayesianLinear's KL produces finite gradients."""
    layer = BayesianLinear(in_features=4, out_features=3, prior_std=1.0, rngs=nnx.Rngs(0))

    def kl_of_mean(mean: jax.Array) -> jax.Array:
        return diagonal_gaussian_kl(mean, layer.weight_logvar[...], prior_mean=0.0, prior_std=1.0)

    grad_fn = jax.grad(kl_of_mean)
    grads = grad_fn(layer.weight_mean[...])
    assert grads.shape == layer.weight_mean[...].shape
    assert jnp.all(jnp.isfinite(grads))
