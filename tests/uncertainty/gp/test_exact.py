r"""Tests for exact conjugate-Gaussian GP regression.

The exact GP posterior under a zero-mean Gaussian prior with kernel ``K``
and observation noise ``σ²`` is closed-form (Rasmussen & Williams 2006
§2.2 Algorithm 2.1):

    α       = (K + σ² I)^{-1} y,
    mean    = K(X*, X) α,
    cov     = K(X*, X*) - K(X*, X) (K + σ² I)^{-1} K(X, X*).

Algorithm 2.1 implements the same identities via a single Cholesky
factor ``L = chol(K + σ² I)`` (the canonical numerically-stable form
that the opifex implementation uses).

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; Algorithm 2.1 §2.2 (PRIMARY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    ExactGPState,
    fit_exact_gp,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.types import PredictiveDistribution


def _train_y(x: jax.Array) -> jax.Array:
    """Reference function ``sin(2x) + 0.5 x``."""
    return jnp.sin(2.0 * x.squeeze(-1)) + 0.5 * x.squeeze(-1)


def test_rbf_kernel_returns_symmetric_positive_diagonal() -> None:
    """``rbf_kernel(X, X, …)`` is symmetric with ``output_scale²`` on the diagonal."""
    x = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    k = rbf_kernel(x, x, lengthscale=0.3, output_scale=1.5)
    assert k.shape == (5, 5)
    assert jnp.allclose(k, k.T, atol=1e-6)
    assert jnp.allclose(jnp.diag(k), jnp.full(5, 1.5**2), atol=1e-6)


def test_rbf_kernel_decays_with_distance() -> None:
    """RBF entries strictly decrease as inputs move apart."""
    x_close = jnp.asarray([[0.0]])
    x_far = jnp.asarray([[2.0]])
    near = float(rbf_kernel(x_close, x_close, lengthscale=1.0, output_scale=1.0)[0, 0])
    far = float(rbf_kernel(x_close, x_far, lengthscale=1.0, output_scale=1.0)[0, 0])
    assert far < near


def test_fit_exact_gp_returns_state_with_cholesky_and_alpha() -> None:
    """The fitted state carries the training data + Cholesky factor + ``α``."""
    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = _train_y(x_train)
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    assert isinstance(state, ExactGPState)
    assert state.cholesky.shape == (6, 6)
    assert state.alpha.shape == (6,)
    # The lower-triangular factor recovers ``K + σ² I``.
    k_plus_noise = rbf_kernel(
        x_train, x_train, lengthscale=0.4, output_scale=1.0
    ) + 0.05**2 * jnp.eye(6)
    reconstructed = state.cholesky @ state.cholesky.T
    assert jnp.allclose(reconstructed, k_plus_noise, atol=1e-5)


def test_predict_exact_gp_interpolates_training_points_within_noise() -> None:
    """At training points the predictive mean is within the noise scale of the targets."""
    x_train = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y_train = _train_y(x_train)
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.3,
        output_scale=1.0,
        noise_std=0.01,
    )
    predictive = predict_exact_gp(state=state, x_test=x_train)
    assert isinstance(predictive, PredictiveDistribution)
    # Predictive mean lies within ``5 σ`` of the training targets at training points.
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.01


def test_predict_exact_gp_matches_rw06_algorithm_21() -> None:
    r"""The predictive moments match the direct closed-form expressions.

    Reference: Rasmussen & Williams 2006 eqs. (2.25)-(2.26).

        mean = K(X*, X) (K + σ² I)^{-1} y,
        var  = K(X*, X*) - K(X*, X) (K + σ² I)^{-1} K(X, X*).
    """
    x_train = jax.random.normal(jax.random.PRNGKey(0), (5, 1))
    y_train = _train_y(x_train)
    x_test = jax.random.normal(jax.random.PRNGKey(1), (3, 1))
    lengthscale, output_scale, noise_std = 0.7, 1.2, 0.05

    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )
    predictive = predict_exact_gp(state=state, x_test=x_test)

    k_train = rbf_kernel(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    k_test = rbf_kernel(x_test, x_train, lengthscale=lengthscale, output_scale=output_scale)
    k_diag = jnp.full((3,), output_scale**2)
    k_train_inv = jnp.linalg.inv(k_train + noise_std**2 * jnp.eye(5))
    expected_mean = k_test @ k_train_inv @ y_train
    expected_var = k_diag - jnp.sum((k_test @ k_train_inv) * k_test, axis=-1)

    assert predictive.variance is not None
    assert jnp.allclose(predictive.mean, expected_mean, atol=1e-5)
    assert jnp.allclose(predictive.variance, expected_var, atol=1e-5)


def test_predict_variance_shrinks_with_more_training_data() -> None:
    """Doubling training data (in the same region) does not increase predictive variance."""
    x_test = jnp.asarray([[0.0]])

    def fit_with_n(n: int) -> jax.Array:
        x_train = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
        y_train = _train_y(x_train)
        state = fit_exact_gp(
            x_train=x_train,
            y_train=y_train,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
        )
        pd = predict_exact_gp(state=state, x_test=x_test)
        assert pd.variance is not None
        return pd.variance[0]

    small_var = float(fit_with_n(4))
    large_var = float(fit_with_n(16))
    assert large_var <= small_var + 1e-6


def test_fit_and_predict_are_jit_compatible() -> None:
    """The full pipeline compiles end-to-end under ``jax.jit``."""
    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = _train_y(x_train)
    x_test = jnp.linspace(-1.5, 1.5, 7).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.05,
        )
        pd = predict_exact_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (7,)
    assert jnp.all(jnp.isfinite(out))


def test_fit_exact_gp_rejects_nonpositive_noise_std() -> None:
    """``noise_std`` must be strictly positive (jitter prevents singular K)."""
    with pytest.raises(ValueError, match="noise_std"):
        fit_exact_gp(
            x_train=jnp.zeros((3, 1)),
            y_train=jnp.zeros((3,)),
            lengthscale=1.0,
            output_scale=1.0,
            noise_std=0.0,
        )


def test_predict_distribution_metadata_advertises_exact_gp_source() -> None:
    """Metadata records ``method`` and ``source_package``."""
    state = fit_exact_gp(
        x_train=jnp.zeros((3, 1)),
        y_train=jnp.zeros((3,)),
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=0.1,
    )
    predictive = predict_exact_gp(state=state, x_test=jnp.zeros((2, 1)))
    keys = {k for k, _ in predictive.metadata}
    assert "method" in keys
    assert "source_package" in keys


def test_predict_exact_gp_nonstationary_and_tanimoto_kernel_diagonal_in_linear_work() -> None:
    """``predict_exact_gp`` evaluates non-stationary ``k(x_i, x_i)`` in ``O(m)`` without ``(m, m)``."""
    call_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def linear_dot_kernel(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float | jax.Array,
        output_scale: float | jax.Array,
    ) -> jax.Array:
        call_shapes.append((x1.shape, x2.shape))
        return (output_scale**2) * (0.25 + (x1 @ x2.T) / (lengthscale**2))

    x_train = jnp.array([[-1.0, 0.5], [0.2, -0.3], [1.5, 2.0], [0.8, -1.2]], dtype=jnp.float32)
    y_train = jnp.array([0.4, -0.2, 2.1, -0.7], dtype=jnp.float32)
    x_test = jnp.array(
        [[0.0, 0.0], [2.0, 1.0], [-1.5, 3.0], [0.5, 0.5], [1.2, -0.4]], dtype=jnp.float32
    )
    ls, os, sigma = 0.8, 1.3, 0.05

    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=ls,
        output_scale=os,
        noise_std=sigma,
        kernel_fn=linear_dot_kernel,
    )
    call_shapes.clear()
    pred = predict_exact_gp(state=state, x_test=x_test)
    assert pred.variance is not None

    # Must NEVER materialize the (m, m) = (5, 5) test-test Gram matrix!
    assert (5, 2) not in [s1 for s1, s2 in call_shapes if s1 == s2]

    # Exact closed-form reference with non-constant prior diagonal k(x_i, x_i) != os^2
    k_tt = (os**2) * (0.25 + (x_train @ x_train.T) / (ls**2)) + (sigma**2) * jnp.eye(4)
    k_xt = (os**2) * (0.25 + (x_test @ x_train.T) / (ls**2))
    k_xx_diag = jnp.diag((os**2) * (0.25 + (x_test @ x_test.T) / (ls**2)))
    assert not bool(jnp.allclose(k_xx_diag, jnp.full_like(k_xx_diag, os**2)))
    expected_var = k_xx_diag - jnp.sum(k_xt * jnp.linalg.solve(k_tt, k_xt.T).T, axis=-1)
    assert jnp.allclose(pred.variance, expected_var, atol=1e-5)


def test_exact_gp_vmap_grad_trace_counter_and_float64() -> None:
    """``fit_exact_gp`` + ``predict_exact_gp`` compose under ``vmap``, ``grad``, and float64."""
    from substrax.testing import restored_jax_config, TraceCounter

    from opifex.uncertainty.gp import exact_gp_loocv_log_predictive

    x_train = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float32).reshape(-1, 1)
    y_train = _train_y(x_train)
    x_test = jnp.linspace(-0.5, 0.5, 4, dtype=jnp.float32).reshape(-1, 1)

    def pipeline(ls: jax.Array, os: jax.Array, ns: jax.Array) -> tuple[jax.Array, jax.Array]:
        st = fit_exact_gp(
            x_train=x_train, y_train=y_train, lengthscale=ls, output_scale=os, noise_std=ns
        )
        pd = predict_exact_gp(state=st, x_test=x_test)
        assert pd.variance is not None
        return pd.mean, pd.variance

    ls_vec = jnp.array([0.3, 0.6, 1.0], dtype=jnp.float32)
    os_vec = jnp.array([0.8, 1.0, 1.4], dtype=jnp.float32)
    ns_vec = jnp.array([0.05, 0.1, 0.2], dtype=jnp.float32)

    counter = TraceCounter()
    jitted_vmap = jax.jit(counter.wrap(jax.vmap(pipeline)))
    with counter.expect(new_traces=1):
        means1, vars1 = jitted_vmap(ls_vec, os_vec, ns_vec)
    with counter.expect(new_traces=0):
        _ = jitted_vmap(ls_vec + 0.1, os_vec, ns_vec)
    assert means1.shape == (3, 4)
    assert vars1.shape == (3, 4)
    assert bool(jnp.all(jnp.isfinite(vars1)))

    grads = jax.grad(
        lambda params: exact_gp_loocv_log_predictive(
            state=fit_exact_gp(
                x_train=x_train,
                y_train=y_train,
                lengthscale=params[0],
                output_scale=params[1],
                noise_std=params[2],
            )
        )
    )(jnp.array([0.4, 1.0, 0.08], dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(grads)))

    with restored_jax_config():
        jax.config.update("jax_enable_x64", True)
        x64 = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64).reshape(-1, 1)
        y64 = jnp.sin(x64.squeeze(-1))
        st64 = fit_exact_gp(
            x_train=x64, y_train=y64, lengthscale=0.5, output_scale=1.1, noise_std=0.02
        )
        pd64 = predict_exact_gp(state=st64, x_test=x64)
        loocv64 = exact_gp_loocv_log_predictive(state=st64)
        assert st64.cholesky.dtype == jnp.float64
        assert pd64.mean.dtype == jnp.float64
        assert pd64.variance is not None and pd64.variance.dtype == jnp.float64
        assert loocv64.dtype == jnp.float64
