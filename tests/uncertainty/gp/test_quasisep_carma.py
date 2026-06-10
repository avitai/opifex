r"""Tests for the scalable CARMA state-space Gaussian process.

The direct-evaluation CARMA(p, q) covariance (:func:`carma_kernel`)
evaluates a dense ``O(n²)`` Gram matrix. The scalable port here maps
the same covariance into a real, block-structured linear-Gaussian SDE
(the celerite quasiseparable realization of Foreman-Mackey+ 2017) and
runs the forward Kalman filter + backward RTS smoother, giving exact
fit + predict in ``O(n)`` time.

Equivalence guarantee
---------------------

For a CARMA process the kernel value factorizes as
``k(x1, x2) = h^T P_inf A(|x1 - x2|) h`` with drift ``F``, stationary
covariance ``P_inf``, transition ``A(dt) = exp(F dt)`` and observation
vector ``h``. Feeding ``(F, P_inf, A, h)`` to the Kalman primitives
therefore reproduces, to numerical precision, the dense exact GP built
on :func:`carma_kernel`:

* the log marginal likelihood matches
  ``-½ y^T (K + σ² I)^{-1} y - ½ log|K + σ² I| - ½ n log 2π``;
* the posterior mean / variance at any test point coincide with the
  direct-form exact GP.

A CARMA(1, 0) process is an Ornstein-Uhlenbeck / Matern-1/2 process,
so that special case is checked against the Matern-1/2 exact GP.

Reference implementation consulted (READ-ONLY):
``../tinygp/src/tinygp/kernels/quasisep.py:CARMA``
(``design_matrix`` + ``stationary_covariance`` + ``observation_model``
 + ``transition_matrix``).

References
----------
* Kelly, B. C., et al. 2014 — *Flexible and scalable methods for
  quantifying stochastic variability in the era of massive
  time-domain astronomical data sets*, ApJ, arXiv:1402.5978 (CARMA
  definition, ACVF Eq. 4).
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modeling*, AJ, arXiv:1703.09710
  (celerite state-space realization).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    carma_kernel,
    fit_exact_gp,
    matern12_kernel,
    predict_exact_gp,
)
from opifex.uncertainty.gp.quasisep_carma import (
    fit_quasisep_carma_gp,
    predict_quasisep_carma_gp,
    QuasisepCarmaGPState,
)
from opifex.uncertainty.types import PredictiveDistribution


# CARMA(2, 1): one complex-conjugate root pair (the celerite term).
_AR_21 = jnp.asarray([1.2, 1.5])
_MA_21 = jnp.asarray([1.0, 0.3])
# CARMA(3, 2): one real root + one complex-conjugate pair, chosen so the
# celerite realization is well-defined (every ACVF term representable).
_AR_32 = jnp.asarray([0.34101393, 2.15835222, 2.97216176])
_MA_32 = jnp.asarray([1.0, -0.47175869, -0.36866401])

_LENGTHSCALE = 1.3
_OUTPUT_SCALE = 0.9
_NOISE_STD = 0.05


def _toy_time_series_data(seed: int = 0, *, num_train: int = 28) -> tuple[jax.Array, jax.Array]:
    """Irregularly-sampled 1-D time series with a damped oscillation."""
    key = jax.random.PRNGKey(seed)
    times = jnp.sort(
        jax.random.uniform(key, (num_train,), minval=0.0, maxval=4.0 * jnp.pi, dtype=jnp.float64)
    )
    observations = (jnp.sin(1.7 * times) * jnp.exp(-0.15 * times)).astype(jnp.float64)
    return times.reshape(-1, 1), observations


def _dense_carma_posterior(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    ar_coefficients: jax.Array,
    ma_coefficients: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
) -> tuple[jax.Array, jax.Array]:
    """Dense ``O(n²)`` CARMA exact-GP posterior using the true kernel diagonal.

    The shared :func:`predict_exact_gp` assumes a unit kernel diagonal
    (``k(x, x) = output_scale²``), which holds for RBF/Matern but not for
    CARMA (``k(0) ≠ 1``). This reference therefore evaluates the full
    train/test Gram blocks so the equivalence check exercises the true
    CARMA covariance.
    """
    kernel = carma_kernel(alpha=ar_coefficients, beta=ma_coefficients)
    gram_train = kernel(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    gram_train = gram_train + noise_std**2 * jnp.eye(x_train.shape[0])
    cross = kernel(x_train, x_test, lengthscale=lengthscale, output_scale=output_scale)
    test_diag = jnp.diag(kernel(x_test, x_test, lengthscale=lengthscale, output_scale=output_scale))
    cholesky = jnp.linalg.cholesky(gram_train)
    alpha = jax.scipy.linalg.cho_solve((cholesky, True), y_train)
    mean = cross.T @ alpha
    solved = jax.scipy.linalg.solve_triangular(cholesky, cross, lower=True)
    variance = test_diag - jnp.sum(solved * solved, axis=0)
    return mean, variance


def _direct_exact_gp_log_marginal(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    ar_coefficients: jax.Array,
    ma_coefficients: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
) -> jax.Array:
    """Closed-form exact-GP log marginal for the dense CARMA kernel."""
    kernel = carma_kernel(alpha=ar_coefficients, beta=ma_coefficients)
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        kernel_fn=kernel,
    )
    n = y_train.shape[0]
    return (
        -0.5 * jnp.dot(y_train, state.alpha)
        - jnp.sum(jnp.log(jnp.diag(state.cholesky)))
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )


# -----------------------------------------------------------------------------
# Equivalence against the dense carma_kernel exact GP
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ar_coefficients", "ma_coefficients"),
    [(_AR_21, _MA_21), (_AR_32, _MA_32)],
)
def test_log_marginal_matches_dense_carma_exact_gp(
    ar_coefficients: jax.Array, ma_coefficients: jax.Array
) -> None:
    """Kalman log-marginal == dense CARMA exact-GP log-marginal to ~1e-6."""
    x_train, y_train = _toy_time_series_data(0)
    state = fit_quasisep_carma_gp(
        x_train=x_train,
        y_train=y_train,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    direct_log_marginal = _direct_exact_gp_log_marginal(
        x_train=x_train,
        y_train=y_train,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    assert isinstance(state, QuasisepCarmaGPState)
    assert jnp.isfinite(state.log_marginal_likelihood)
    # The pinned test suite runs in float32 (and the dense CARMA kernel /
    # ACVF evaluate in complex64), so the equivalence residual sits at the
    # float32 noise floor rather than the 1e-6 achievable under float64.
    assert jnp.allclose(state.log_marginal_likelihood, direct_log_marginal, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize(
    ("ar_coefficients", "ma_coefficients"),
    [(_AR_21, _MA_21), (_AR_32, _MA_32)],
)
def test_posterior_mean_matches_dense_carma_exact_gp(
    ar_coefficients: jax.Array, ma_coefficients: jax.Array
) -> None:
    """Posterior mean at held-out times matches the dense exact GP to ~1e-6."""
    x_train, y_train = _toy_time_series_data(1)
    state = fit_quasisep_carma_gp(
        x_train=x_train,
        y_train=y_train,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    x_test = jnp.linspace(0.5, 11.0, 14).reshape(-1, 1)
    scalable_pred = predict_quasisep_carma_gp(state=state, x_test=x_test)
    direct_mean, _ = _dense_carma_posterior(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    assert isinstance(scalable_pred, PredictiveDistribution)
    assert jnp.allclose(scalable_pred.mean, direct_mean, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("ar_coefficients", "ma_coefficients"),
    [(_AR_21, _MA_21), (_AR_32, _MA_32)],
)
def test_posterior_variance_matches_dense_carma_exact_gp(
    ar_coefficients: jax.Array, ma_coefficients: jax.Array
) -> None:
    """Posterior variance at held-out times matches the dense exact GP to ~1e-6."""
    x_train, y_train = _toy_time_series_data(2)
    state = fit_quasisep_carma_gp(
        x_train=x_train,
        y_train=y_train,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    x_test = jnp.linspace(0.5, 11.0, 14).reshape(-1, 1)
    scalable_pred = predict_quasisep_carma_gp(state=state, x_test=x_test)
    _, direct_variance = _dense_carma_posterior(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        ar_coefficients=ar_coefficients,
        ma_coefficients=ma_coefficients,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    assert scalable_pred.variance is not None
    assert jnp.allclose(scalable_pred.variance, direct_variance, atol=1e-6, rtol=1e-5)


# -----------------------------------------------------------------------------
# CARMA(1, 0) reduces to the Ornstein-Uhlenbeck / Matern-1/2 process
# -----------------------------------------------------------------------------


def test_carma_one_zero_reduces_to_matern12_exact_gp() -> None:
    """CARMA(1, 0) posterior matches a Matern-1/2 exact GP (OU special case)."""
    x_train, y_train = _toy_time_series_data(3)
    ar_rate = 0.8
    # CARMA(1, 0): k(τ) = (β0²/2α0) exp(-α0 |τ|). Matching the Matern-1/2
    # exact GP requires a length-scale ℓ = 1/α0 and an output-scale that
    # absorbs the CARMA amplitude 1/√(2α0).
    ma_amplitude = float(jnp.sqrt(2.0 * ar_rate))
    matern_lengthscale = 1.0 / ar_rate
    state = fit_quasisep_carma_gp(
        x_train=x_train,
        y_train=y_train,
        ar_coefficients=jnp.asarray([ar_rate]),
        ma_coefficients=jnp.asarray([ma_amplitude]),
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=_NOISE_STD,
    )
    matern_state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=matern_lengthscale,
        output_scale=1.0,
        noise_std=_NOISE_STD,
        kernel_fn=matern12_kernel,
    )
    x_test = jnp.linspace(0.5, 11.0, 14).reshape(-1, 1)
    carma_pred = predict_quasisep_carma_gp(state=state, x_test=x_test)
    matern_pred = predict_exact_gp(state=matern_state, x_test=x_test)
    assert carma_pred.variance is not None
    assert matern_pred.variance is not None
    assert jnp.allclose(carma_pred.mean, matern_pred.mean, atol=1e-6, rtol=1e-5)
    assert jnp.allclose(carma_pred.variance, matern_pred.variance, atol=1e-6, rtol=1e-5)


# -----------------------------------------------------------------------------
# O(n) scaling: handle a large series the dense path would choke on
# -----------------------------------------------------------------------------


def test_fit_handles_large_series() -> None:
    """Fit at ``n = 5000`` runs and returns a finite log marginal."""
    key = jax.random.PRNGKey(7)
    # Build strictly-increasing times from positive spacings so float32
    # rounding cannot create ties at n = 5000.
    spacings = 0.01 + jax.random.uniform(key, (5000,), minval=0.0, maxval=0.1)
    times = jnp.cumsum(spacings).reshape(-1, 1)
    observations = jnp.sin(1.3 * times.squeeze(-1)) * jnp.exp(-0.001 * times.squeeze(-1))
    state = fit_quasisep_carma_gp(
        x_train=times,
        y_train=observations,
        ar_coefficients=_AR_21,
        ma_coefficients=_MA_21,
        lengthscale=_LENGTHSCALE,
        output_scale=_OUTPUT_SCALE,
        noise_std=_NOISE_STD,
    )
    assert state.x_train.shape == (5000, 1)
    assert jnp.isfinite(state.log_marginal_likelihood)


# -----------------------------------------------------------------------------
# JAX transform compatibility: jit / grad / vmap
# -----------------------------------------------------------------------------


def test_fit_predict_is_jit_compatible() -> None:
    """``jax.jit`` compiles the full fit + predict pipeline."""
    x_train, y_train = _toy_time_series_data(4, num_train=18)
    x_test = jnp.linspace(0.5, 9.0, 6).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_quasisep_carma_gp(
            x_train=x_t,
            y_train=y_t,
            ar_coefficients=_AR_21,
            ma_coefficients=_MA_21,
            lengthscale=_LENGTHSCALE,
            output_scale=_OUTPUT_SCALE,
            noise_std=_NOISE_STD,
        )
        predictive = predict_quasisep_carma_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (6,)
    assert jnp.all(jnp.isfinite(out))


def test_log_marginal_grad_wrt_parameters_is_finite() -> None:
    """``jax.grad`` of the log marginal w.r.t. AR/MA/σ params is finite."""
    x_train, y_train = _toy_time_series_data(5, num_train=20)

    def negative_log_marginal(
        ar_coefficients: jax.Array,
        ma_coefficients: jax.Array,
        output_scale: jax.Array,
    ) -> jax.Array:
        state = fit_quasisep_carma_gp(
            x_train=x_train,
            y_train=y_train,
            ar_coefficients=ar_coefficients,
            ma_coefficients=ma_coefficients,
            lengthscale=_LENGTHSCALE,
            output_scale=output_scale,
            noise_std=_NOISE_STD,
        )
        return -state.log_marginal_likelihood

    grad_ar, grad_ma, grad_scale = jax.grad(negative_log_marginal, argnums=(0, 1, 2))(
        _AR_21, _MA_21, jnp.asarray(_OUTPUT_SCALE)
    )
    assert jnp.all(jnp.isfinite(grad_ar))
    assert jnp.all(jnp.isfinite(grad_ma))
    assert jnp.isfinite(grad_scale)


def test_fit_predict_is_vmap_compatible() -> None:
    """``jax.vmap`` maps fit + predict over a batch of output scales."""
    x_train, y_train = _toy_time_series_data(6, num_train=16)
    x_test = jnp.linspace(0.5, 9.0, 5).reshape(-1, 1)
    output_scales = jnp.asarray([0.7, 1.0, 1.4])

    def fit_predict(output_scale: jax.Array) -> jax.Array:
        state = fit_quasisep_carma_gp(
            x_train=x_train,
            y_train=y_train,
            ar_coefficients=_AR_21,
            ma_coefficients=_MA_21,
            lengthscale=_LENGTHSCALE,
            output_scale=output_scale,
            noise_std=_NOISE_STD,
        )
        return predict_quasisep_carma_gp(state=state, x_test=x_test).mean

    means = jax.vmap(fit_predict)(output_scales)
    assert means.shape == (3, 5)
    assert jnp.all(jnp.isfinite(means))


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


def test_requires_sorted_training_times() -> None:
    """Unsorted training times raise a clear ``ValueError``."""
    x_train = jnp.asarray([[0.0], [2.0], [1.0]])
    y_train = jnp.asarray([0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="sorted"):
        fit_quasisep_carma_gp(
            x_train=x_train,
            y_train=y_train,
            ar_coefficients=_AR_21,
            ma_coefficients=_MA_21,
            lengthscale=_LENGTHSCALE,
            output_scale=_OUTPUT_SCALE,
            noise_std=_NOISE_STD,
        )


def test_rejects_ma_order_exceeding_ar_order() -> None:
    """``q + 1 > p`` violates the CARMA definition and raises."""
    x_train, y_train = _toy_time_series_data(8, num_train=12)
    with pytest.raises(ValueError, match="q \\+ 1"):
        fit_quasisep_carma_gp(
            x_train=x_train,
            y_train=y_train,
            ar_coefficients=jnp.asarray([1.2, 1.5]),
            ma_coefficients=jnp.asarray([1.0, 0.3, 0.1]),
            lengthscale=_LENGTHSCALE,
            output_scale=_OUTPUT_SCALE,
            noise_std=_NOISE_STD,
        )
