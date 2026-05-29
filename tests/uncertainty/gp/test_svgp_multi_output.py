r"""Sparse multi-output SVGP via ICM/LCM kernels — Task 11.1 D6.

Slices 2 and 3 shipped the dense multi-output ICM and LCM kernels.
Slice 11 shipped the Titsias collapsed sparse-variational GP for
single-output stationary kernels. This slice closes the gap by
verifying the existing :func:`fit_svgp` / :func:`predict_svgp`
machinery accepts multi-output ICM/LCM kernels directly once the
``K_{xx}`` diagonal is computed per-point (rather than assumed
``output_scale^2``-constant — only true for single-output stationary
kernels).

The ICM/LCM input convention (slices 2-3) routes the **last column**
as an integer task index in ``{0, …, T-1}`` and the preceding columns
as the spatial / feature coordinates. The same convention applies to
``x_train``, ``x_inducing`` and ``x_test`` here.

References
----------
* Bonilla, E., Chai, K. M., Williams, C. K. I. 2007 — *Multi-task
  Gaussian Process Prediction*, NeurIPS (ICM intrinsic-coregionalisation
  basis).
* Alvarez, M. A., Lawrence, N. D. 2009 — *Sparse Convolved Gaussian
  Processes for Multi-output Regression*, NeurIPS (inducing-point
  sparse multi-output GPs).
* Titsias, M. K. 2009 — *Variational Learning of Inducing Variables in
  Sparse Gaussian Processes*, AISTATS (collapsed sparse-variational
  GP — single-output basis, generalised here to ICM/LCM kernels).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    fit_svgp,
    matern32_kernel,
    multi_output_icm_kernel,
    multi_output_lcm_kernel,
    predict_svgp,
    rbf_kernel,
    svgp_collapsed_elbo,
)


def _two_task_training_data(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    """Two synthetic tasks: task 0 = sin(2x), task 1 = -sin(2x) + 0.5."""
    key = jax.random.PRNGKey(seed)
    key_x_a, key_x_b, key_noise = jax.random.split(key, 3)
    x_task_0 = jax.random.uniform(key_x_a, (15, 1), minval=-1.5, maxval=1.5)
    x_task_1 = jax.random.uniform(key_x_b, (15, 1), minval=-1.5, maxval=1.5)
    y_task_0 = jnp.sin(2.0 * x_task_0.squeeze(-1))
    y_task_1 = -jnp.sin(2.0 * x_task_1.squeeze(-1)) + 0.5
    noise = 0.02 * jax.random.normal(key_noise, (30,))
    task_indices = jnp.concatenate([jnp.zeros((15,)), jnp.ones((15,))])
    x_features = jnp.concatenate([x_task_0, x_task_1], axis=0)
    x_train = jnp.concatenate([x_features, task_indices[:, None]], axis=1)
    y_train = jnp.concatenate([y_task_0, y_task_1], axis=0) + noise
    return x_train, y_train


def _two_task_inducing(num_per_task: int = 5) -> jax.Array:
    """Evenly-spaced inducing points across both tasks."""
    grid = jnp.linspace(-1.5, 1.5, num_per_task).reshape(-1, 1)
    z_task_0 = jnp.concatenate([grid, jnp.zeros((num_per_task, 1))], axis=1)
    z_task_1 = jnp.concatenate([grid, jnp.ones((num_per_task, 1))], axis=1)
    return jnp.concatenate([z_task_0, z_task_1], axis=0)


def test_fit_svgp_runs_with_multi_output_icm_kernel() -> None:
    """``fit_svgp`` accepts an ICM kernel and produces a finite fitted state."""
    x_train, y_train = _two_task_training_data(0)
    x_inducing = _two_task_inducing(5)
    coregionalisation = jnp.asarray([[1.0, -0.6], [-0.6, 1.2]])
    icm_kernel = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel, coregionalisation=coregionalisation
    )
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.1,
        kernel_fn=icm_kernel,
    )
    assert jnp.all(jnp.isfinite(state.cholesky_kmm))
    assert jnp.all(jnp.isfinite(state.scaled_alpha))
    elbo = svgp_collapsed_elbo(state=state)
    assert jnp.isfinite(elbo)


def test_predict_svgp_with_icm_recovers_per_task_training_signal() -> None:
    """Predict on task-0 inputs returns ``+sin(2x)``; task-1 returns ``-sin(2x)+0.5``."""
    x_train, y_train = _two_task_training_data(1)
    x_inducing = _two_task_inducing(6)
    coregionalisation = jnp.asarray([[1.0, -0.6], [-0.6, 1.2]])
    icm_kernel = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel, coregionalisation=coregionalisation
    )
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=icm_kernel,
    )
    x_test_features = jnp.linspace(-1.2, 1.2, 8).reshape(-1, 1)
    x_test_task_0 = jnp.concatenate([x_test_features, jnp.zeros((8, 1))], axis=1)
    x_test_task_1 = jnp.concatenate([x_test_features, jnp.ones((8, 1))], axis=1)
    pred_task_0 = predict_svgp(state=state, x_test=x_test_task_0)
    pred_task_1 = predict_svgp(state=state, x_test=x_test_task_1)
    target_task_0 = jnp.sin(2.0 * x_test_features.squeeze(-1))
    target_task_1 = -jnp.sin(2.0 * x_test_features.squeeze(-1)) + 0.5
    assert jnp.max(jnp.abs(pred_task_0.mean - target_task_0)) < 0.3
    assert jnp.max(jnp.abs(pred_task_1.mean - target_task_1)) < 0.3


def test_predict_svgp_with_icm_returns_per_task_finite_variance() -> None:
    """Per-task predictive variance is positive at every test point."""
    x_train, y_train = _two_task_training_data(2)
    x_inducing = _two_task_inducing(5)
    coregionalisation = jnp.asarray([[1.0, 0.4], [0.4, 1.5]])
    icm_kernel = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel, coregionalisation=coregionalisation
    )
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=icm_kernel,
    )
    x_test_features = jnp.linspace(-1.5, 1.5, 6).reshape(-1, 1)
    x_test = jnp.concatenate([x_test_features, jnp.ones((6, 1))], axis=1)
    pred = predict_svgp(state=state, x_test=x_test)
    assert pred.variance is not None
    assert jnp.all(pred.variance > 0.0)
    # Task-1 marginal variance at zero data overlap should approach
    # ``output_scale^2 * B[1, 1] = 1.5`` — the per-task prior variance.
    assert jnp.max(pred.variance) <= 1.5 + 1e-3


def test_fit_svgp_runs_with_multi_output_lcm_kernel() -> None:
    """``fit_svgp`` accepts a 2-component LCM kernel and produces a finite state."""
    x_train, y_train = _two_task_training_data(3)
    x_inducing = _two_task_inducing(5)
    b_q0 = jnp.asarray([[1.0, 0.3], [0.3, 1.0]])
    b_q1 = jnp.asarray([[0.5, -0.2], [-0.2, 0.5]])
    lcm_kernel = multi_output_lcm_kernel(components=((rbf_kernel, b_q0), (matern32_kernel, b_q1)))
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.1,
        kernel_fn=lcm_kernel,
    )
    elbo = svgp_collapsed_elbo(state=state)
    assert jnp.isfinite(elbo)
    x_test_features = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    x_test = jnp.concatenate([x_test_features, jnp.zeros((4, 1))], axis=1)
    pred = predict_svgp(state=state, x_test=x_test)
    assert pred.variance is not None
    assert jnp.all(jnp.isfinite(pred.mean))
    assert jnp.all(pred.variance > 0.0)


def test_multi_output_svgp_full_pipeline_is_jit_compatible() -> None:
    """``fit_svgp`` + ``predict_svgp`` with an ICM kernel compile under ``jax.jit``."""
    x_train, y_train = _two_task_training_data(4)
    x_inducing = _two_task_inducing(4)
    coregionalisation = jnp.asarray([[1.0, 0.3], [0.3, 1.0]])

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array) -> jax.Array:
        icm_kernel = multi_output_icm_kernel(
            base_kernel_fn=rbf_kernel, coregionalisation=coregionalisation
        )
        state = fit_svgp(
            x_train=x_t,
            y_train=y_t,
            x_inducing=x_inducing,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.1,
            kernel_fn=icm_kernel,
        )
        x_test = jnp.concatenate(
            [jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1), jnp.zeros((5, 1))], axis=1
        )
        predictive = predict_svgp(state=state, x_test=x_test)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


def test_single_output_svgp_baseline_still_passes_after_diag_refactor() -> None:
    """Existing single-output stationary SVGP path is unchanged by the K_diag refactor."""
    x_train = jnp.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    state = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=0.05,
    )
    x_test = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    pred = predict_svgp(state=state, x_test=x_test)
    assert pred.variance is not None
    # Single-output stationary diag K_xx is exactly output_scale^2 — the refactor
    # must preserve this baseline.
    expected_max_var = 1.0  # output_scale^2 = 1.0
    assert jnp.max(pred.variance) <= expected_max_var + 1e-4
