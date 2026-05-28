r"""Tests for deep kernels (NN feature map ∘ base kernel).

A *deep kernel* composes a neural-network feature map ``φ_θ: R^d →
R^d'`` with any base kernel ``k_base`` so the effective kernel on
inputs ``x`` becomes

.. math::

    k_{\text{deep}}(x, x') = k_{\text{base}}(\phi_{\theta}(x),\,
                                            \phi_{\theta}(x')).

Wilson, Hu, Salakhutdinov, Xing 2016 (arXiv:1511.02222) — *Deep Kernel
Learning* — introduced the construction; the kernel inherits the base
kernel's hyperparameters (length-scale, output-scale, …) while the
feature map ``φ_θ`` provides the learnable representation.

The opifex implementation is a *single-line composition over the
existing ``kernel_fn`` API*: any callable that maps ``(n, d) -> (n,
d')`` (e.g. a ``flax.nnx`` module, a plain Python lambda, or an
``nnx.Sequential`` feature extractor) can be wrapped. **No equinox
dependency**: opifex uses ``flax.nnx`` for the NN component (the
canonical opifex neural backbone).

References
----------
* Wilson, A. G., Hu, Z., Salakhutdinov, R., Xing, E. P. 2016 — *Deep
  Kernel Learning*, AISTATS, arXiv:1511.02222 (PRIMARY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty.gp import (
    deep_kernel,
    fit_exact_gp,
    matern32_kernel,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.types import PredictiveDistribution


def test_deep_kernel_with_identity_feature_map_equals_base_kernel() -> None:
    """The composition collapses to the base kernel when ``φ(x) = x``."""
    kernel = deep_kernel(feature_map=lambda x: x, base_kernel_fn=rbf_kernel)
    x = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    deep = kernel(x, x, lengthscale=0.7, output_scale=1.0)
    base = rbf_kernel(x, x, lengthscale=0.7, output_scale=1.0)
    assert jnp.allclose(deep, base, atol=1e-6)


def test_deep_kernel_routes_inputs_through_feature_map_before_base_kernel() -> None:
    """Verifies ``k_deep(x, x') = k_base(φ(x), φ(x'))`` against the explicit form."""

    def feature_map(x: jax.Array) -> jax.Array:
        return jnp.stack([x.squeeze(-1), x.squeeze(-1) ** 2], axis=-1)

    kernel = deep_kernel(feature_map=feature_map, base_kernel_fn=matern32_kernel)
    x = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    composed = kernel(x, x, lengthscale=0.5, output_scale=1.0)
    expected = matern32_kernel(
        feature_map(x), feature_map(x), lengthscale=0.5, output_scale=1.0
    )
    assert jnp.allclose(composed, expected, atol=1e-6)


def test_deep_kernel_plugs_into_fit_exact_gp_driver() -> None:
    """The composed kernel routes through ``fit_exact_gp(..., kernel_fn=…)``."""
    rngs = nnx.Rngs(0)
    linear_feature_map = nnx.Linear(in_features=1, out_features=4, rngs=rngs)
    kernel = deep_kernel(feature_map=linear_feature_map, base_kernel_fn=rbf_kernel)

    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=kernel,
    )
    predictive = predict_exact_gp(state=state, x_test=x_train)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    # The lifted-feature GP still interpolates the training data within a few noise scales.
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.05


def test_deep_kernel_is_jit_compatible_with_flax_feature_map() -> None:
    """End-to-end ``nnx.jit`` compatibility with a ``flax.nnx`` feature map."""
    rngs = nnx.Rngs(1)
    feature_extractor = nnx.Linear(in_features=1, out_features=3, rngs=rngs)
    kernel = deep_kernel(feature_map=feature_extractor, base_kernel_fn=rbf_kernel)

    x_train = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    x_test = jnp.linspace(-0.5, 0.5, 3).reshape(-1, 1)

    @nnx.jit
    def fit_predict(
        feat: nnx.Linear, x_t: jax.Array, y_t: jax.Array, x_q: jax.Array
    ) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
            kernel_fn=deep_kernel(feature_map=feat, base_kernel_fn=rbf_kernel),
        )
        pd = predict_exact_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(feature_extractor, x_train, y_train, x_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_deep_kernel_with_two_layer_mlp_extracts_nonlinear_features() -> None:
    """A 2-layer MLP feature map produces a non-trivially different kernel matrix."""
    rngs = nnx.Rngs(2)

    class _TwoLayer(nnx.Module):
        def __init__(self, *, in_features: int, hidden: int, out_features: int, rngs: nnx.Rngs):
            self.linear_1 = nnx.Linear(in_features=in_features, out_features=hidden, rngs=rngs)
            self.linear_2 = nnx.Linear(in_features=hidden, out_features=out_features, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.linear_2(jax.nn.tanh(self.linear_1(x)))

    mlp = _TwoLayer(in_features=1, hidden=8, out_features=4, rngs=rngs)
    kernel = deep_kernel(feature_map=mlp, base_kernel_fn=rbf_kernel)
    x = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    composed = kernel(x, x, lengthscale=0.5, output_scale=1.0)
    raw = rbf_kernel(x, x, lengthscale=0.5, output_scale=1.0)
    assert composed.shape == raw.shape
    # Random-init MLP features differ from the identity → the two Grams disagree.
    assert not jnp.allclose(composed, raw, atol=1e-3)
