r"""Linear multi-fidelity GP (Kennedy & O'Hagan AR(1)) — Task 11.3 slice 33.

Tests the canonical linear autoregressive multi-fidelity model

    f_i(x) = rho_i * f_{i-1}(x) + delta_i(x),

where each ``delta_i`` is an independent GP, parametrised by per-level
length-scales and output-scales plus the inter-level scaling factor
``rho_i`` (Kennedy & O'Hagan 2000 §2.5; bayesnewton mirror —
``emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Joint kernel
# -----------------------------------------------------------------------------


def test_linear_multi_fidelity_kernel_gives_psd_gram_on_two_fidelities() -> None:
    """The K+O AR(1) block kernel produces a PSD joint Gram matrix."""
    from opifex.uncertainty.multi_fidelity import linear_multi_fidelity_kernel

    key = jax.random.PRNGKey(0)
    x_low = jax.random.uniform(key, (5, 1))
    x_high = jax.random.uniform(jax.random.PRNGKey(1), (3, 1))
    x = jnp.concatenate(
        [
            jnp.concatenate([x_low, jnp.zeros((5, 1))], axis=1),
            jnp.concatenate([x_high, jnp.ones((3, 1))], axis=1),
        ],
        axis=0,
    )
    gram = linear_multi_fidelity_kernel(
        x,
        x,
        lengthscales=(0.5, 0.5),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.5,),
    )
    eigvals = jnp.linalg.eigvalsh(gram + 1e-6 * jnp.eye(gram.shape[0]))
    assert jnp.all(eigvals > 0.0)


def test_linear_multi_fidelity_kernel_reduces_to_single_kernel_at_one_fidelity() -> None:
    """With a single fidelity level the joint kernel must equal the base kernel."""
    from opifex.uncertainty.gp import rbf_kernel
    from opifex.uncertainty.multi_fidelity import linear_multi_fidelity_kernel

    x = jnp.linspace(0.0, 1.0, 6).reshape(-1, 1)
    x_levelled = jnp.concatenate([x, jnp.zeros((6, 1))], axis=1)
    gram_mf = linear_multi_fidelity_kernel(
        x_levelled,
        x_levelled,
        lengthscales=(0.4,),
        output_scales=(1.0,),
        scaling_factors=(),
    )
    gram_single = rbf_kernel(x, x, lengthscale=0.4, output_scale=1.0)
    assert jnp.allclose(gram_mf, gram_single, atol=1e-6)


# -----------------------------------------------------------------------------
# Fit / predict
# -----------------------------------------------------------------------------


def test_fit_linear_multi_fidelity_gp_returns_finite_alpha_and_cholesky() -> None:
    """fit_linear_multi_fidelity_gp produces a usable Cholesky + alpha."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        LinearMultiFidelityGPState,
    )

    x_low = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten()) + 0.1 * jnp.cos(
        4.0 * jnp.pi * x_high.flatten()
    )
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    assert isinstance(state, LinearMultiFidelityGPState)
    assert jnp.all(jnp.isfinite(state.cholesky))
    assert jnp.all(jnp.isfinite(state.alpha))


def test_predict_linear_multi_fidelity_gp_recovers_high_fidelity_signal() -> None:
    """Predicting at high fidelity recovers the high-fidelity training signal."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        predict_linear_multi_fidelity_gp,
    )
    from opifex.uncertainty.types import PredictiveDistribution

    x_low = jnp.linspace(0.0, 1.0, 30).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 6).reshape(-1, 1)

    def low_fidelity(x: jax.Array) -> jax.Array:
        return jnp.sin(2.0 * jnp.pi * x.flatten())

    def high_fidelity(x: jax.Array) -> jax.Array:
        return jnp.sin(2.0 * jnp.pi * x.flatten()) + 0.1 * jnp.cos(4.0 * jnp.pi * x.flatten())

    y_low = low_fidelity(x_low)
    y_high = high_fidelity(x_high)
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.2, 0.15),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_test = jnp.linspace(0.15, 0.85, 5).reshape(-1, 1)
    predictive_high = predict_linear_multi_fidelity_gp(state=state, x_test=x_test, target_level=1)
    assert isinstance(predictive_high, PredictiveDistribution)
    assert predictive_high.variance is not None
    truth = high_fidelity(x_test)
    assert jnp.allclose(predictive_high.mean, truth, atol=0.3)


def test_predict_linear_multi_fidelity_gp_low_fidelity_has_smaller_variance_with_more_data() -> (
    None
):
    """Low-fidelity predict has smaller variance than high-fidelity at the same x."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        predict_linear_multi_fidelity_gp,
    )

    x_low = jnp.linspace(0.0, 1.0, 40).reshape(-1, 1)
    x_high = jnp.array([[0.3], [0.6]])
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.2, 0.2),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_test = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    low_pred = predict_linear_multi_fidelity_gp(state=state, x_test=x_test, target_level=0)
    high_pred = predict_linear_multi_fidelity_gp(state=state, x_test=x_test, target_level=1)
    assert low_pred.variance is not None
    assert high_pred.variance is not None
    assert jnp.all(low_pred.variance < high_pred.variance)


def test_predict_linear_multi_fidelity_gp_metadata_advertises_multi_fidelity() -> None:
    """``predict_linear_multi_fidelity_gp`` advertises the linear-MF estimator."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        predict_linear_multi_fidelity_gp,
    )

    x_low = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    x_high = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    predictive = predict_linear_multi_fidelity_gp(
        state=state,
        x_test=jnp.array([[0.5]]),
        target_level=1,
    )
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "linear_multi_fidelity_gp"
    assert "Kennedy" in str(metadata.get("paper", ""))
