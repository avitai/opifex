r"""Tests for the LCM multi-output kernel + the LOOCV objective.

LCM (Linear Coregionalisation Model, Alvarez et al. 2012 §3.2) extends
ICM to a *sum* of base kernels each carrying its own coregionalisation
matrix:

    k_LCM((x, i), (x', j)) = Σ_q k_q(x, x') · B_q[i, j].

Single-component LCM with ``Q = 1`` collapses exactly to ICM, which
the first test verifies.

LOOCV (Rasmussen & Williams 2006 §5.4.2 eqs. 5.10-5.12) computes the
leave-one-out predictive distribution for a fitted exact GP **without
refitting** ``n`` times: from the inverse Gram matrix
``K_inv = (K + σ² I)^{-1}`` one obtains

    μ_loo[i] = y[i] - α[i] / K_inv[i, i],
    σ²_loo[i] = 1.0 / K_inv[i, i],

and the LOOCV log-predictive sums the per-point Gaussian log-density.
The test verifies that LOOCV log-pred decreases when training noise is
inflated past the data noise (overestimating σ² shrinks the predictive
and hurts the loo-likelihood).

References
----------
* Alvarez, M. A., Rosasco, L., Lawrence, N. D. 2012 — *Kernels for
  Vector-Valued Functions: A Review*, arXiv:1106.6251 §3.2 (LCM,
  PRIMARY).
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §5.4.2 (LOOCV closed-form, PRIMARY).
* Sundararajan, S., Keerthi, S. S. 2001 — *Predictive approaches for
  choosing hyperparameters in Gaussian processes*, Neural Computation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    exact_gp_loocv_log_predictive,
    fit_exact_gp,
    matern32_kernel,
    multi_output_icm_kernel,
    multi_output_lcm_kernel,
    rbf_kernel,
)


def test_lcm_with_single_component_collapses_to_icm() -> None:
    """``LCM([(k_base, B)])(x1, x2) == ICM(k_base, B)(x1, x2)``."""
    coregionalisation = jnp.asarray([[1.0, 0.4], [0.4, 0.6]])
    lcm = multi_output_lcm_kernel(components=((rbf_kernel, coregionalisation),))
    icm = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel,
        coregionalisation=coregionalisation,
    )
    x = jnp.asarray([[0.0, 0], [0.5, 1], [1.0, 0]])
    assert jnp.allclose(
        lcm(x, x, lengthscale=0.8, output_scale=1.0),
        icm(x, x, lengthscale=0.8, output_scale=1.0),
        atol=1e-6,
    )


def test_lcm_sums_multiple_components() -> None:
    """``LCM([(k1, B1), (k2, B2)]) = ICM(k1, B1) + ICM(k2, B2)`` exactly."""
    b_short = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    b_long = jnp.asarray([[0.5, 0.2], [0.2, 0.3]])
    lcm = multi_output_lcm_kernel(
        components=(
            (rbf_kernel, b_short),
            (matern32_kernel, b_long),
        )
    )
    icm_short = multi_output_icm_kernel(base_kernel_fn=rbf_kernel, coregionalisation=b_short)
    icm_long = multi_output_icm_kernel(base_kernel_fn=matern32_kernel, coregionalisation=b_long)
    x = jnp.asarray([[0.0, 0], [0.5, 1], [1.0, 0], [-0.5, 1]])
    expected = icm_short(x, x, lengthscale=0.4, output_scale=1.0) + icm_long(
        x, x, lengthscale=0.4, output_scale=1.0
    )
    assert jnp.allclose(
        lcm(x, x, lengthscale=0.4, output_scale=1.0),
        expected,
        atol=1e-6,
    )


def test_lcm_kernel_is_jit_compatible() -> None:
    """The LCM kernel compiles inside ``jax.jit``."""
    lcm = multi_output_lcm_kernel(
        components=(
            (rbf_kernel, jnp.eye(2)),
            (matern32_kernel, jnp.asarray([[1.0, 0.1], [0.1, 0.7]])),
        )
    )

    @jax.jit
    def evaluate(x: jax.Array) -> jax.Array:
        return lcm(x, x, lengthscale=0.5, output_scale=1.0)

    x = jnp.asarray([[0.0, 0], [0.3, 1]])
    k = evaluate(x)
    assert k.shape == (2, 2)
    assert jnp.all(jnp.isfinite(k))


def test_loocv_log_predictive_matches_closed_form_on_3_point_toy() -> None:
    r"""Verifies RW06 eqs. 5.10-5.12 on a 3-point GP.

    Direct computation: refit on ``[X \\ x_i]``, predict at ``x_i``, sum
    log-Gaussian densities; LOOCV closed-form should agree.
    """
    x_train = jnp.asarray([[-1.0], [0.0], [1.0]])
    y_train = jnp.asarray([0.5, -0.3, 0.8])
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.7,
        output_scale=1.0,
        noise_std=0.1,
    )

    loocv = exact_gp_loocv_log_predictive(state=state)

    # Direct LOO: refit on each leave-one-out subset and evaluate the
    # log-density of the held-out target under the predictive Gaussian.
    log_lik = 0.0
    for i in range(3):
        mask = jnp.arange(3) != i
        sub_x = x_train[mask]
        sub_y = y_train[mask]
        sub_state = fit_exact_gp(
            x_train=sub_x,
            y_train=sub_y,
            lengthscale=0.7,
            output_scale=1.0,
            noise_std=0.1,
        )
        from opifex.uncertainty.gp import predict_exact_gp

        pd = predict_exact_gp(state=sub_state, x_test=x_train[i : i + 1])
        assert pd.variance is not None
        mu, var = float(pd.mean[0]), float(pd.variance[0]) + 0.1**2
        target = float(y_train[i])
        log_lik += -0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(var) + (target - mu) ** 2 / var)

    assert jnp.allclose(loocv, log_lik, atol=1e-4)


def test_loocv_log_predictive_decreases_with_overinflated_noise() -> None:
    """LOOCV log-pred decreases when training noise is too large vs. the data noise."""
    x_train = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))

    base = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=0.05,
    )
    inflated = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=1.5,
    )
    base_loocv = float(exact_gp_loocv_log_predictive(state=base))
    inflated_loocv = float(exact_gp_loocv_log_predictive(state=inflated))
    assert inflated_loocv < base_loocv


def test_loocv_log_predictive_is_jit_compatible() -> None:
    """LOOCV objective compiles under ``jax.jit`` with traced training data."""
    x_train = jnp.asarray([[-1.0], [0.0], [1.0]])
    y_train = jnp.asarray([0.5, -0.3, 0.8])

    @jax.jit
    def loocv(x_t: jax.Array, y_t: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.1,
        )
        return exact_gp_loocv_log_predictive(state=state)

    value = loocv(x_train, y_train)
    assert jnp.isfinite(value)
