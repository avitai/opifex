r"""Tests for the additive (OAK-base) kernel.

An *additive* kernel is the sum of per-dimension univariate kernels:

.. math::

    k_{\text{add}}(x, x') = \sum_{d=1}^{D} k_{d}(x_{d}, x'_{d}).

This is the ``max_order = 1`` case of the **Orthogonal Additive Kernel**
(OAK; Lu, Boukouvalas, Hensman 2022 ICML — *Additive Gaussian Processes
Revisited*). Higher-order OAK interactions through Newton-Girard
recursion + the orthogonality constraint under a Gaussian input
measure are deferred to a follow-up slice; the first-order (plain
additive) form already enables ANOVA-style interpretable GPs.

References
----------
* Lu, X., Boukouvalas, A., Hensman, J. 2022 — *Additive Gaussian
  Processes Revisited*, ICML (PRIMARY for the OAK construction).
* Duvenaud, D., Nickisch, H., Rasmussen, C. E. 2011 — *Additive
  Gaussian Processes*, NeurIPS (additive baseline).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    additive_kernel,
    fit_exact_gp,
    matern32_kernel,
    predict_exact_gp,
    rbf_kernel,
)


def test_additive_kernel_sums_per_dimension_components_exactly() -> None:
    """``k_add(x, x') = Σ_d k_d(x_d, x'_d)``."""
    additive = additive_kernel(component_kernel_fns=(rbf_kernel, matern32_kernel))
    x = jnp.asarray([[0.0, 1.0], [0.5, -0.5], [-1.0, 0.0]])
    result = additive(x, x, lengthscale=0.5, output_scale=1.0)
    expected = rbf_kernel(
        x[:, 0:1], x[:, 0:1], lengthscale=0.5, output_scale=1.0
    ) + matern32_kernel(x[:, 1:2], x[:, 1:2], lengthscale=0.5, output_scale=1.0)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_additive_kernel_single_component_collapses_to_that_kernel() -> None:
    """``additive([k_rbf])(x, x') = k_rbf(x, x')`` on 1-D inputs."""
    additive = additive_kernel(component_kernel_fns=(rbf_kernel,))
    x = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    assert jnp.allclose(
        additive(x, x, lengthscale=0.5, output_scale=1.0),
        rbf_kernel(x, x, lengthscale=0.5, output_scale=1.0),
        atol=1e-6,
    )


def test_additive_kernel_rejects_dimension_mismatch() -> None:
    """Number of components must match input dimensionality."""
    additive = additive_kernel(component_kernel_fns=(rbf_kernel, rbf_kernel, rbf_kernel))
    with pytest.raises(ValueError, match="components"):
        additive(jnp.zeros((3, 2)), jnp.zeros((3, 2)), lengthscale=1.0, output_scale=1.0)


def test_additive_kernel_rejects_empty_components() -> None:
    """At least one component kernel is required."""
    with pytest.raises(ValueError, match="component"):
        additive_kernel(component_kernel_fns=())


def test_additive_kernel_plugs_into_fit_exact_gp() -> None:
    """The additive kernel routes through ``fit_exact_gp(..., kernel_fn=…)``."""
    additive = additive_kernel(component_kernel_fns=(rbf_kernel, rbf_kernel))
    x_train = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
    y_train = jnp.sin(2.0 * x_train[:, 0]) + 0.5 * x_train[:, 1]
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=additive,
    )
    predictive = predict_exact_gp(state=state, x_test=x_train)
    assert predictive.variance is not None
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.05


def test_additive_kernel_is_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility."""
    additive = additive_kernel(component_kernel_fns=(rbf_kernel, matern32_kernel))
    x_train = jax.random.normal(jax.random.PRNGKey(1), (6, 2))
    y_train = jax.random.normal(jax.random.PRNGKey(2), (6,))
    x_test = jax.random.normal(jax.random.PRNGKey(3), (3, 2))

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.1,
            kernel_fn=additive,
        )
        pd = predict_exact_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))
