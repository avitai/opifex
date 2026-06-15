r"""Non-linear multi-fidelity GP (NARGP, Perdikaris+ 2017) — Task 11.3 slice 34.

Tests the Perdikaris autoregressive non-linear multi-fidelity model

    f_0(x) ~ GP(0, k_0(x, x')),
    f_i(x) = g_i(x, f_{i-1}(x))    with g_i ~ GP composing the
                                   previous-fidelity output as an
                                   extra input dimension,

mirroring ``emukit.multi_fidelity.models.non_linear_multi_fidelity_model``
(PRIMARY reference). Prediction uncertainty propagates through the
fidelity chain via Monte-Carlo sampling on the previous-level
predictive distribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_fit_nonlinear_multi_fidelity_gp_returns_per_level_states() -> None:
    """fit_nonlinear_multi_fidelity_gp produces one fitted GP per fidelity level."""
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        NonLinearMultiFidelityGPState,
    )

    x_low = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 6).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten()) ** 2  # non-linear
    state = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.5),
        noise_std=0.05,
    )
    assert isinstance(state, NonLinearMultiFidelityGPState)
    assert len(state.level_states) == 2


def test_predict_nonlinear_multi_fidelity_gp_recovers_high_fidelity_signal() -> None:
    """Predicting at level 1 recovers the non-linear ``sin(2 pi x)^2`` truth."""
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )
    from opifex.uncertainty.types import PredictiveDistribution

    x_low = jnp.linspace(0.0, 1.0, 40).reshape(-1, 1)
    x_high = jnp.linspace(0.05, 0.95, 10).reshape(-1, 1)

    def low_fidelity(x: jax.Array) -> jax.Array:
        return jnp.sin(2.0 * jnp.pi * x.flatten())

    def high_fidelity(x: jax.Array) -> jax.Array:
        return jnp.sin(2.0 * jnp.pi * x.flatten()) ** 2

    state = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(low_fidelity(x_low), high_fidelity(x_high)),
        lengthscales=(0.15, 0.2),
        output_scales=(1.0, 0.5),
        noise_std=0.03,
    )
    x_test = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    predictive = predict_nonlinear_multi_fidelity_gp(
        state=state,
        x_test=x_test,
        target_level=1,
        num_samples=128,
        rng_key=jax.random.PRNGKey(0),
    )
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    truth = high_fidelity(x_test)
    assert jnp.allclose(predictive.mean, truth, atol=0.4)


def test_predict_nonlinear_multi_fidelity_gp_metadata_advertises_nargp() -> None:
    """Metadata advertises the NARGP non-linear MF estimator."""
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    x_low = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    x_high = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten()) ** 2
    state = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.5),
        noise_std=0.05,
    )
    predictive = predict_nonlinear_multi_fidelity_gp(
        state=state,
        x_test=jnp.array([[0.5]]),
        target_level=1,
        num_samples=64,
        rng_key=jax.random.PRNGKey(1),
    )
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "nonlinear_multi_fidelity_gp"
    assert "Perdikaris" in str(metadata.get("paper", ""))


def test_predict_nonlinear_multi_fidelity_gp_level_zero_matches_exact_gp() -> None:
    """Predict at level 0 reduces to the underlying low-fidelity ExactGP."""
    from opifex.uncertainty.gp import fit_exact_gp, predict_exact_gp
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    x_low = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    x_high = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten()) ** 2
    state = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.5),
        noise_std=0.05,
    )
    exact_state = fit_exact_gp(
        x_train=x_low,
        y_train=y_low,
        lengthscale=0.3,
        output_scale=1.0,
        noise_std=0.05,
    )
    x_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
    nargp_pred = predict_nonlinear_multi_fidelity_gp(
        state=state,
        x_test=x_test,
        target_level=0,
        num_samples=1,
        rng_key=jax.random.PRNGKey(2),
    )
    exact_pred = predict_exact_gp(state=exact_state, x_test=x_test)
    assert jnp.allclose(nargp_pred.mean, exact_pred.mean, atol=1e-5)
