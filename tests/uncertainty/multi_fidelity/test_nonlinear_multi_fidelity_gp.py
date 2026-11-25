r"""Non-linear multi-fidelity GP (NARGP, Perdikaris+ 2017) — Task 11.3 slice 34.

Tests the Perdikaris autoregressive non-linear multi-fidelity model

    f_0(x) ~ GP(0, k_0(x, x')),
    f_i(x) = g_i(x, f_{i-1}(x))    with g_i ~ GP composing the
                                   previous-fidelity output as an
                                   extra input dimension.

Prediction uncertainty propagates through the
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


def test_nargp_structured_base_kernel_isolates_domain_features_and_supports_jit_grad() -> None:
    """Custom ``base_kernel_fn`` only sees ``d``-dim ``x`` (never ``f_{i-1}``) and jit/grad work."""
    import pytest
    from substrax.testing import TraceCounter

    from opifex.uncertainty.gp import rbf_kernel
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    d = 4

    def tanimoto_domain_kernel(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float | jax.Array,
        output_scale: float | jax.Array,
    ) -> jax.Array:
        # Strict contract: domain fingerprint kernel only accepts d-dimensional x!
        if x1.shape[1] != d or x2.shape[1] != d:
            raise AssertionError(f"Expected feature dim {d}, got {x1.shape[1]} and {x2.shape[1]}")
        del lengthscale
        dot = x1 @ x2.T
        n1 = jnp.sum(x1 * x1, axis=-1, keepdims=True)
        n2 = jnp.sum(x2 * x2, axis=-1, keepdims=True).T
        denom = jnp.maximum(n1 + n2 - dot, jnp.asarray(1e-8, dtype=x1.dtype))
        return (output_scale**2) * (dot / denom)

    x_low = jnp.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    y_low = jnp.array([0.5, 0.8, -0.2, 0.1, 0.3, 1.1], dtype=jnp.float32)
    x_high = x_low[:4]
    y_high = y_low[:4] ** 2 + 0.2
    x_test = jnp.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)

    counter = TraceCounter()

    def run_nargp(params: jax.Array) -> jax.Array:
        st = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low, x_high),
            y_train_per_level=(y_low, y_high),
            lengthscales=(params[0], params[1]),
            output_scales=(params[2], params[3]),
            noise_std=params[4],
            base_kernel_fn=tanimoto_domain_kernel,
            f_kernel_fn=rbf_kernel,
        )
        pd = predict_nonlinear_multi_fidelity_gp(
            state=st,
            x_test=x_test,
            target_level=1,
            num_samples=32,
            rng_key=jax.random.key(7),
        )
        assert pd.variance is not None
        return jnp.sum(pd.mean**2 + pd.variance)

    jitted = jax.jit(counter.wrap(run_nargp))
    init_params = jnp.array([0.5, 0.6, 1.0, 0.8, 0.05], dtype=jnp.float32)
    with counter.expect(new_traces=1):
        val1 = jitted(init_params)
    with counter.expect(new_traces=0):
        _ = jitted(init_params + 0.05)
    assert bool(jnp.isfinite(val1))

    grads = jax.grad(run_nargp)(init_params)
    assert grads.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(grads)))

    with pytest.raises(ValueError, match="kernel_fns"):
        fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low, x_high),
            y_train_per_level=(y_low, y_high),
            lengthscales=(0.5, 0.5),
            output_scales=(1.0, 1.0),
            noise_std=0.05,
            kernel_fns=(tanimoto_domain_kernel,),
        )

    with jax.enable_x64(True):
        st64 = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low.astype(jnp.float64), x_high.astype(jnp.float64)),
            y_train_per_level=(y_low.astype(jnp.float64), y_high.astype(jnp.float64)),
            lengthscales=(0.5, 0.6),
            output_scales=(1.0, 0.8),
            noise_std=0.05,
            base_kernel_fn=tanimoto_domain_kernel,
            f_kernel_fn=rbf_kernel,
        )
        pd64 = predict_nonlinear_multi_fidelity_gp(
            state=st64,
            x_test=x_test.astype(jnp.float64),
            target_level=1,
            num_samples=16,
            rng_key=jax.random.key(9),
        )
        assert pd64.mean.dtype == jnp.float64
        assert pd64.variance is not None and pd64.variance.dtype == jnp.float64


def test_nargp_large_amplitude_linear_scaling_and_multilevel_contracts() -> None:
    """NARGP evaluates raw unnormalized f_{i-1} with explicit f_lengthscales, splits RNG keys
    per level, preserves small-scale variance without a 1e-6 floor, and validates target_level."""
    import pytest

    from opifex.uncertainty.gp import fit_exact_gp, predict_exact_gp, rbf_kernel
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    x_low = jnp.linspace(0.0, 1.0, 45).reshape(-1, 1)
    x_high_uniform = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    x_high_osc = jnp.linspace(0.0, 1.0, 4).reshape(-1, 1)
    y_low = 50.0 * jnp.sin(4.0 * x_low.flatten())
    y_high_uniform = 100.0 * jnp.sin(4.0 * x_high_uniform.flatten())
    y_low_osc = 50.0 * jnp.sin(10.0 * x_low.flatten())
    y_high_osc = 100.0 * jnp.sin(10.0 * x_high_osc.flatten())
    x_test = jnp.linspace(0.05, 0.95, 25).reshape(-1, 1)
    y_true = 100.0 * jnp.sin(4.0 * x_test.flatten())
    y_true_osc = 100.0 * jnp.sin(10.0 * x_test.flatten())

    # Positive control for raw unnormalized f_0 scaling: default unit f_lengthscales=(1.0,)
    # treats f_0 as O(1) when f_0 actually spans [-50, +50], collapsing k_f(f, f') toward
    # an identity matrix away from high-fidelity training points (~57.5 RMSE vs ~1.1 RMSE).
    unit_f_scale_uniform = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low[:30], x_high_uniform),
        y_train_per_level=(y_low[:30], y_high_uniform),
        lengthscales=(0.5, 5.0),
        output_scales=(50.0, 240.0),
        f_lengthscales=(1.0,),
        bias_lengthscales=(1.0,),
        bias_output_scales=(1.0,),
        noise_std=0.1,
    )
    unit_f_scale_uniform_rmse = float(
        jnp.sqrt(
            jnp.mean(
                (
                    predict_nonlinear_multi_fidelity_gp(
                        state=unit_f_scale_uniform,
                        x_test=x_test,
                        target_level=1,
                        num_samples=64,
                        rng_key=jax.random.key(11),
                    ).mean
                    - y_true
                )
                ** 2
            )
        )
    )
    sf_best_osc_rmse = min(
        float(
            jnp.sqrt(
                jnp.mean(
                    (
                        predict_exact_gp(
                            state=fit_exact_gp(
                                x_train=x_high_osc,
                                y_train=y_high_osc,
                                lengthscale=ls,
                                output_scale=100.0,
                                noise_std=0.5,
                            ),
                            x_test=x_test,
                        ).mean
                        - y_true_osc
                    )
                    ** 2
                )
            )
        )
        for ls in (0.15, 0.5, 1.0)
    )

    for kernel in (
        rbf_kernel,
        lambda x1, x2, *, lengthscale, output_scale: rbf_kernel(
            x1, x2, lengthscale=lengthscale, output_scale=output_scale
        ),
    ):
        mf_uniform = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low[:30], x_high_uniform),
            y_train_per_level=(y_low[:30], y_high_uniform),
            lengthscales=(0.5, 5.0),
            output_scales=(50.0, 240.0),
            f_lengthscales=(120.0,),
            bias_lengthscales=(1.0,),
            bias_output_scales=(1.0,),
            noise_std=0.1,
            base_kernel_fn=kernel,
        )
        pred_uniform = predict_nonlinear_multi_fidelity_gp(
            state=mf_uniform,
            x_test=x_test,
            target_level=1,
            num_samples=64,
            rng_key=jax.random.key(11),
        )
        rmse_uniform = float(jnp.sqrt(jnp.mean((pred_uniform.mean - y_true) ** 2)))
        assert rmse_uniform < unit_f_scale_uniform_rmse

        mf_osc = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low, x_high_osc),
            y_train_per_level=(y_low_osc, y_high_osc),
            lengthscales=(0.15, 5.0),
            output_scales=(50.0, 240.0),
            f_lengthscales=(120.0,),
            bias_lengthscales=(1.0,),
            bias_output_scales=(1.0,),
            noise_std=0.5,
            base_kernel_fn=kernel,
        )
        pred_osc = predict_nonlinear_multi_fidelity_gp(
            state=mf_osc,
            x_test=x_test,
            target_level=1,
            num_samples=64,
            rng_key=jax.random.key(11),
        )
        rmse_osc = float(jnp.sqrt(jnp.mean((pred_osc.mean - y_true_osc) ** 2)))
        assert rmse_osc < sf_best_osc_rmse

    x_high = x_high_uniform
    y_high = y_high_uniform

    # Small-scale (1e-4) output preserves ~1e-8/1e-12 variance without 1e-6 clamp
    small_state = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(1e-4 * jnp.sin(x_low.flatten()), 1e-4 * jnp.sin(x_high.flatten())),
        lengthscales=(0.4, 0.4),
        output_scales=(1e-4, 1e-4),
        noise_std=1e-6,
    )
    small_pred = predict_nonlinear_multi_fidelity_gp(
        state=small_state,
        x_test=x_high,
        target_level=1,
        num_samples=32,
        rng_key=jax.random.key(12),
    )
    assert small_pred.variance is not None
    assert float(jnp.max(small_pred.variance)) < 1e-9

    # 3-level fit + f_lengthscales + validation bounds (no RecursionError on target_level=-1)
    st3 = fit_nonlinear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_low[::2], x_high),
        y_train_per_level=(y_low, 1.5 * y_low[::2], y_high),
        lengthscales=(0.5, 0.5, 0.5),
        output_scales=(50.0, 75.0, 100.0),
        noise_std=0.05,
        f_lengthscales=(25.0, 40.0),
    )
    pd3 = predict_nonlinear_multi_fidelity_gp(
        state=st3,
        x_test=x_test[:4],
        target_level=2,
        num_samples=32,
        rng_key=jax.random.key(13),
    )
    assert bool(jnp.all(jnp.isfinite(pd3.mean)))

    for bad_level in (-1, 3):
        with pytest.raises(ValueError, match="target_level"):
            predict_nonlinear_multi_fidelity_gp(
                state=st3,
                x_test=x_test[:2],
                target_level=bad_level,
                rng_key=jax.random.key(14),
            )
    with pytest.raises(ValueError, match="at least 1 fidelity"):
        fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(),
            y_train_per_level=(),
            lengthscales=(),
            output_scales=(),
            noise_std=0.05,
        )


def test_perdikaris_two_term_gram_parity_and_level_state_contract() -> None:
    """Exact Gram-matrix parity with Perdikaris et al. (2017, Eq. 2.12) and per-level ExactGPState contract."""

    from opifex.uncertainty.gp import (
        exact_gp_loocv_log_predictive,
        predict_exact_gp,
        rbf_kernel,
    )
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        nargp_composite_kernel,
        predict_nonlinear_multi_fidelity_gp,
    )

    with jax.enable_x64(True):
        z1 = jnp.array([[0.1, -0.4, 1.2], [0.5, 0.2, -0.7], [0.9, -0.1, 0.3]], dtype=jnp.float64)
        z2 = jnp.array([[0.2, 0.3, 0.5], [0.8, -0.2, -0.4]], dtype=jnp.float64)
        ls_rho, os_rho = 0.35, 1.4
        ls_f = 0.85
        ls_bias, os_bias = 0.65, 0.45

        gram = nargp_composite_kernel(
            z1,
            z2,
            interaction_lengthscale=ls_rho,
            interaction_output_scale=os_rho,
            f_lengthscale=ls_f,
            bias_lengthscale=ls_bias,
            bias_output_scale=os_bias,
            base_kernel_fn=rbf_kernel,
            f_kernel_fn=rbf_kernel,
        )
        expected_gram = rbf_kernel(
            z1[:, :-1], z2[:, :-1], lengthscale=ls_rho, output_scale=os_rho
        ) * rbf_kernel(z1[:, -1:], z2[:, -1:], lengthscale=ls_f, output_scale=1.0) + rbf_kernel(
            z1[:, :-1], z2[:, :-1], lengthscale=ls_bias, output_scale=os_bias
        )
        assert jnp.allclose(gram, expected_gram, rtol=1e-14, atol=1e-14)

        x0 = jnp.linspace(0.0, 1.0, 15, dtype=jnp.float64).reshape(-1, 1)
        y0 = jnp.sin(2.0 * jnp.pi * x0[:, 0])
        x1 = jnp.array([[0.15], [0.45], [0.75], [0.90]], dtype=jnp.float64)
        y1 = (x1[:, 0] - 0.5) * (jnp.sin(2.0 * jnp.pi * x1[:, 0]) ** 2) + 0.1 * x1[:, 0]
        state_exact = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x0, x1),
            y_train_per_level=(y0, y1),
            lengthscales=(0.25, ls_rho),
            output_scales=(1.0, os_rho),
            f_lengthscales=(ls_f,),
            bias_lengthscales=(ls_bias,),
            bias_output_scales=(os_bias,),
            noise_std=(1e-4, 2e-3),
        )
        f0_on_x1 = predict_nonlinear_multi_fidelity_gp(
            state=state_exact, x_test=x1, target_level=0, num_samples=1, rng_key=jax.random.key(0)
        ).mean
        raw_aug = jnp.concatenate([x1, f0_on_x1[:, None]], axis=1)
        assert jnp.allclose(state_exact.level_states[1].x_train, raw_aug, rtol=1e-14, atol=1e-14)
        expected_chol = jnp.linalg.cholesky(
            nargp_composite_kernel(
                raw_aug,
                raw_aug,
                interaction_lengthscale=ls_rho,
                interaction_output_scale=os_rho,
                f_lengthscale=ls_f,
                bias_lengthscale=ls_bias,
                bias_output_scale=os_bias,
            )
            + (2e-3**2) * jnp.eye(x1.shape[0], dtype=jnp.float64)
        )
        assert jnp.allclose(
            state_exact.level_states[1].cholesky, expected_chol, rtol=1e-12, atol=1e-12
        )

        direct_lvl1 = predict_exact_gp(
            state=state_exact.level_states[1],
            x_test=state_exact.level_states[1].x_train,
        )
        assert jnp.allclose(direct_lvl1.mean, y1, rtol=1e-4, atol=1e-4)
        assert bool(jnp.isfinite(exact_gp_loocv_log_predictive(state=state_exact.level_states[1])))

        def strict_1d_x_kernel(
            xa: jax.Array,
            xb: jax.Array,
            *,
            lengthscale: float | jax.Array,
            output_scale: float | jax.Array,
        ) -> jax.Array:
            if xa.shape[1] != 1 or xb.shape[1] != 1:
                raise ValueError(f"Expected 1-D x columns, got {xa.shape} and {xb.shape}.")
            return rbf_kernel(xa, xb, lengthscale=lengthscale, output_scale=output_scale)

        state_strict = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x0, x1),
            y_train_per_level=(y0, y1),
            lengthscales=(0.25, ls_rho),
            output_scales=(1.0, os_rho),
            f_lengthscales=(ls_f,),
            bias_lengthscales=(ls_bias,),
            bias_output_scales=(os_bias,),
            noise_std=(1e-4, 2e-3),
            base_kernel_fn=strict_1d_x_kernel,
            f_kernel_fn=rbf_kernel,
        )
        direct_strict = predict_exact_gp(
            state=state_strict.level_states[1],
            x_test=state_strict.level_states[1].x_train,
        )
        assert jnp.allclose(direct_strict.mean, y1, rtol=1e-4, atol=1e-4)


def test_perdikaris_nonlinear_sin_and_forrester_benchmarks() -> None:
    """R^2 > 0.99 on Perdikaris et al. (2017) nonlinear_sin and multi-fidelity Forrester benchmarks."""

    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    with jax.enable_x64(True):
        x_grid = jnp.linspace(0.0, 1.0, 200, dtype=jnp.float64).reshape(-1, 1)

        # 1. Perdikaris et al. (2017) nonlinear_sin benchmark (n_low=50, n_high=14 -> R^2 > 0.99)
        x_low_sin = jnp.linspace(0.0, 1.0, 50, dtype=jnp.float64).reshape(-1, 1)
        x_high_sin = jnp.linspace(0.0, 1.0, 14, dtype=jnp.float64).reshape(-1, 1)
        y_low_sin = jnp.sin(8.0 * jnp.pi * x_low_sin[:, 0])
        y_high_sin = (x_high_sin[:, 0] - jnp.sqrt(2.0)) * (
            jnp.sin(8.0 * jnp.pi * x_high_sin[:, 0]) ** 2
        )
        y_true_sin = (x_grid[:, 0] - jnp.sqrt(2.0)) * (jnp.sin(8.0 * jnp.pi * x_grid[:, 0]) ** 2)

        state_sin = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low_sin, x_high_sin),
            y_train_per_level=(y_low_sin, y_high_sin),
            lengthscales=(0.06, 0.8),
            output_scales=(1.0, 1.2),
            f_lengthscales=(0.8,),
            bias_lengthscales=(1.0,),
            bias_output_scales=(0.05,),
            noise_std=(1e-5, 1e-4),
        )
        pred_sin = predict_nonlinear_multi_fidelity_gp(
            state=state_sin,
            x_test=x_grid,
            target_level=1,
            num_samples=128,
            rng_key=jax.random.key(21),
        )
        r2_sin = 1.0 - float(
            jnp.sum((pred_sin.mean - y_true_sin) ** 2)
            / jnp.sum((y_true_sin - jnp.mean(y_true_sin)) ** 2)
        )
        assert r2_sin > 0.99

        # 2. Multi-fidelity Forrester benchmark (R^2 > 0.99)
        x_low_for = jnp.linspace(0.0, 1.0, 25, dtype=jnp.float64).reshape(-1, 1)
        x_high_for = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64).reshape(-1, 1)
        y_high_for_train = ((6.0 * x_high_for[:, 0] - 2.0) ** 2) * jnp.sin(
            12.0 * x_high_for[:, 0] - 4.0
        )
        y_low_for_train = (
            0.5 * ((6.0 * x_low_for[:, 0] - 2.0) ** 2) * jnp.sin(12.0 * x_low_for[:, 0] - 4.0)
            + 10.0 * (x_low_for[:, 0] - 0.5)
            + 5.0
        )
        state_for = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x_low_for, x_high_for),
            y_train_per_level=(y_low_for_train, y_high_for_train),
            lengthscales=(0.15, 2.0),
            output_scales=(6.0, 20.0),
            f_lengthscales=(15.0,),
            bias_lengthscales=(1.0,),
            bias_output_scales=(15.0,),
            noise_std=(1e-4, 1e-3),
        )
        pred_for = predict_nonlinear_multi_fidelity_gp(
            state=state_for,
            x_test=x_grid,
            target_level=1,
            num_samples=128,
            rng_key=jax.random.key(22),
        )
        y_true_for = ((6.0 * x_grid[:, 0] - 2.0) ** 2) * jnp.sin(12.0 * x_grid[:, 0] - 4.0)
        r2_for = 1.0 - float(
            jnp.sum((pred_for.mean - y_true_for) ** 2)
            / jnp.sum((y_true_for - jnp.mean(y_true_for)) ** 2)
        )
        assert r2_for > 0.99


def test_perdikaris_nargp_jit_vmap_grad_dtype_and_validation() -> None:
    """Separate jit(fit) -> state PyTree -> single-trace jit(predict), vmap, FD grad
    parity across all hyperparameters, float32 preservation under x64, and validation."""
    import pytest
    from substrax.testing import TraceCounter

    from opifex.uncertainty.gp import predict_exact_gp, rbf_kernel
    from opifex.uncertainty.multi_fidelity import (
        fit_nonlinear_multi_fidelity_gp,
        predict_nonlinear_multi_fidelity_gp,
    )

    with jax.enable_x64(True):
        x0 = jnp.linspace(0.0, 1.0, 15, dtype=jnp.float64).reshape(-1, 1)
        y0 = jnp.sin(2.0 * jnp.pi * x0[:, 0])
        x1 = jnp.array([[0.15], [0.45], [0.75], [0.90]], dtype=jnp.float64)
        y1 = (x1[:, 0] - 0.5) * (jnp.sin(2.0 * jnp.pi * x1[:, 0]) ** 2) + 0.1 * x1[:, 0]
        hp_vec = jnp.array([0.22, 0.55, 0.9, 1.1, 0.75, 0.65, 0.35, 0.02, 0.03], dtype=jnp.float64)

        @jax.jit
        def jitted_fit(hp: jax.Array):
            return fit_nonlinear_multi_fidelity_gp(
                x_train_per_level=(x0, x1),
                y_train_per_level=(y0, y1),
                lengthscales=(hp[0], hp[1]),
                output_scales=(hp[2], hp[3]),
                f_lengthscales=(hp[4],),
                bias_lengthscales=(hp[5],),
                bias_output_scales=(hp[6],),
                noise_std=(hp[7], hp[8]),
            )

        st_jit = jitted_fit(hp_vec)
        pred_counter = TraceCounter()
        jitted_predict = jax.jit(
            pred_counter.wrap(
                lambda st, xt, k: predict_nonlinear_multi_fidelity_gp(
                    state=st, x_test=xt, target_level=1, num_samples=24, rng_key=k
                )
            )
        )
        xt_q = jnp.array([[0.25], [0.60], [0.85]], dtype=jnp.float64)
        with pred_counter.expect(new_traces=1):
            p1 = jitted_predict(st_jit, xt_q, jax.random.key(31))
        with pred_counter.expect(new_traces=0):
            _ = jitted_predict(st_jit, xt_q + 0.02, jax.random.key(32))
        assert p1.mean.shape == (3,)

        lvl_counter = TraceCounter()
        jitted_lvl_predict = jax.jit(
            lvl_counter.wrap(lambda lvl_st, zq: predict_exact_gp(state=lvl_st, x_test=zq))
        )
        st_jit_2 = jitted_fit(hp_vec * 1.05)
        with lvl_counter.expect(new_traces=1):
            _ = jitted_lvl_predict(st_jit.level_states[1], st_jit.level_states[1].x_train)
        with lvl_counter.expect(new_traces=0):
            _ = jitted_lvl_predict(st_jit_2.level_states[1], st_jit_2.level_states[1].x_train)

        batched_means = jax.vmap(
            lambda xt_b: (
                predict_nonlinear_multi_fidelity_gp(
                    state=st_jit,
                    x_test=xt_b,
                    target_level=1,
                    num_samples=16,
                    rng_key=jax.random.key(33),
                ).mean
            )
        )(jnp.stack([xt_q, xt_q + 0.05]))
        assert batched_means.shape == (2, 3)

        def scalar_obj(hp: jax.Array) -> jax.Array:
            st = jitted_fit(hp)
            pd = predict_nonlinear_multi_fidelity_gp(
                state=st, x_test=xt_q, target_level=1, num_samples=32, rng_key=jax.random.key(44)
            )
            assert pd.variance is not None
            return jnp.sum(pd.mean * jnp.array([0.4, -0.7, 0.9])) + 0.5 * jnp.sum(pd.variance)

        ad_grad = jax.grad(scalar_obj)(hp_vec)
        eps = 1e-6
        fd_grad = jnp.stack(
            [
                (scalar_obj(hp_vec.at[i].add(eps)) - scalar_obj(hp_vec.at[i].add(-eps)))
                / (2.0 * eps)
                for i in range(hp_vec.shape[0])
            ]
        )
        assert jnp.allclose(ad_grad, fd_grad, rtol=1e-5, atol=1e-6)

        st32 = fit_nonlinear_multi_fidelity_gp(
            x_train_per_level=(x0.astype(jnp.float32), x1.astype(jnp.float32)),
            y_train_per_level=(y0.astype(jnp.float32), y1.astype(jnp.float32)),
            lengthscales=(0.25, 0.4),
            output_scales=(1.0, 1.0),
            noise_std=0.05,
        )
        pd32 = predict_nonlinear_multi_fidelity_gp(
            state=st32,
            x_test=xt_q.astype(jnp.float32),
            target_level=1,
            num_samples=16,
            rng_key=jax.random.key(55),
        )
        assert st32.level_states[0].cholesky.dtype == jnp.float32
        assert st32.level_states[1].cholesky.dtype == jnp.float32
        assert pd32.mean.dtype == jnp.float32
        assert pd32.variance is not None and pd32.variance.dtype == jnp.float32

    for bad_kwargs, pattern in (
        ({"noise_std": 0.05, "bias_lengthscales": (0.2, 0.3)}, "bias_lengthscales"),
        ({"noise_std": 0.05, "bias_output_scales": (-0.1,)}, "strictly positive"),
        ({"noise_std": (0.05,)}, "noise_std"),
        ({"noise_std": (0.05, 0.0)}, "strictly positive"),
        (
            {"noise_std": 0.05, "kernel_fns": (rbf_kernel, rbf_kernel), "f_lengthscales": (0.5,)},
            "kernel_fns",
        ),
        (
            {
                "noise_std": 0.05,
                "kernel_fns": (rbf_kernel, rbf_kernel),
                "bias_lengthscales": (0.5,),
            },
            "kernel_fns",
        ),
    ):
        with pytest.raises(ValueError, match=pattern):
            fit_nonlinear_multi_fidelity_gp(
                x_train_per_level=(x0.astype(jnp.float32), x1.astype(jnp.float32)),
                y_train_per_level=(y0.astype(jnp.float32), y1.astype(jnp.float32)),
                lengthscales=(0.3, 0.3),
                output_scales=(1.0, 1.0),
                **bad_kwargs,  # type: ignore[arg-type]
            )

    for bad_level in (-1, 2, True, 1.5):
        with pytest.raises(ValueError, match="target_level"):
            predict_nonlinear_multi_fidelity_gp(
                state=st32,
                x_test=xt_q.astype(jnp.float32),
                target_level=bad_level,  # type: ignore[arg-type]
                num_samples=8,
                rng_key=jax.random.key(56),
            )
