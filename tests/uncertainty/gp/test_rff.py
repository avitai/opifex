r"""Tests for Random Fourier Features (Rahimi & Recht 2007).

For a shift-invariant kernel ``k(x, x') = k(x - x')`` Bochner's
theorem guarantees a spectral density ``ρ(ω)`` such that

.. math::

    k(x - x') = \int e^{i \omega^{T} (x - x')}\,\rho(\omega)\,d\omega.

Sampling ``ω_1, …, ω_{D/2} \sim ρ`` and defining the feature map

.. math::

    \phi(x) = \sqrt{\tfrac{2}{D}}\,\bigl[
        \cos(\omega_{1}^{T} x), \sin(\omega_{1}^{T} x), \ldots,
        \cos(\omega_{D/2}^{T} x), \sin(\omega_{D/2}^{T} x)
    \bigr]^{T}

yields the unbiased Monte-Carlo estimator
``φ(x)^T φ(x') ≈ k(x, x')`` (Rahimi & Recht 2007 Algorithm 1).
For the RBF kernel ``ρ`` is Gaussian with covariance ``ℓ^{-2} I``.

Once the feature map is constructed, an approximate GP regression
reduces to ridge regression on the lifted features
``Φ(X) ∈ R^{n × D}``: the predictive mean is
``φ(x*)^T (Φ^T Φ + σ² I)^{-1} Φ^T y`` and the predictive variance is
``σ² φ(x*)^T (Φ^T Φ + σ² I)^{-1} φ(x*)``.

References
----------
* Rahimi, A., Recht, B. 2007 — *Random Features for Large-Scale Kernel
  Machines*, NeurIPS, arXiv:0708.0234 (PRIMARY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.gp import (
    fit_rff_gp,
    predict_rff_gp,
    rbf_kernel,
    rbf_random_fourier_features,
)
from opifex.uncertainty.types import PredictiveDistribution


def test_rff_feature_inner_product_approximates_rbf_kernel() -> None:
    """``φ(x)^T φ(x') → k(x, x')`` as ``num_features → ∞``."""
    rngs = nnx.Rngs(0)
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
    phi = rbf_random_fourier_features(
        x=x,
        lengthscale=0.8,
        output_scale=1.0,
        num_features=4096,
        rngs=rngs,
    )
    approx = phi @ phi.T
    exact = rbf_kernel(x, x, lengthscale=0.8, output_scale=1.0)
    # MC estimator standard deviation O(1/sqrt(D)) — with D=4096 the
    # entry-wise error should be < 0.05 for unit output_scale.
    assert jnp.max(jnp.abs(approx - exact)) < 0.05


def test_rff_feature_shape_is_n_by_num_features() -> None:
    """Feature matrix has shape ``(n, num_features)``."""
    rngs = nnx.Rngs(1)
    phi = rbf_random_fourier_features(
        x=jnp.zeros((6, 3)),
        lengthscale=1.0,
        output_scale=1.0,
        num_features=128,
        rngs=rngs,
    )
    assert phi.shape == (6, 128)


def test_rff_gp_fit_predict_returns_predictive_distribution() -> None:
    """The fit/predict driver round-trips through ``PredictiveDistribution``."""
    x_train = jnp.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    state = fit_rff_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
        num_features=256,
        rngs=nnx.Rngs(2),
    )
    predictive = predict_rff_gp(state=state, x_test=x_train)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    # Approximate-GP predictive mean stays within a few noise scales at
    # training points when num_features is sufficient.
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 0.5


def test_rff_gp_predictive_converges_to_exact_gp_as_num_features_grows() -> None:
    """Increasing ``num_features`` reduces the gap to the exact-GP predictive mean."""
    from opifex.uncertainty.gp import fit_exact_gp, predict_exact_gp

    x_train = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    x_test = jnp.linspace(-0.5, 0.5, 5).reshape(-1, 1)

    exact_state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
    )
    exact_mean = predict_exact_gp(state=exact_state, x_test=x_test).mean

    def rff_error(num_features: int) -> float:
        state = fit_rff_gp(
            x_train=x_train,
            y_train=y_train,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.05,
            num_features=num_features,
            rngs=nnx.Rngs(3),
        )
        rff_mean = predict_rff_gp(state=state, x_test=x_test).mean
        return float(jnp.max(jnp.abs(rff_mean - exact_mean)))

    err_64 = rff_error(64)
    err_2048 = rff_error(2048)
    assert err_2048 < err_64


def test_rff_gp_pipeline_is_jit_compatible() -> None:
    """Fit + predict compile end-to-end under ``jax.jit`` with a raw key."""
    x_train = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    x_test = jnp.linspace(-0.5, 0.5, 3).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array, key: jax.Array) -> jax.Array:
        state = fit_rff_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=0.05,
            num_features=64,
            rngs=key,
        )
        pd = predict_rff_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x_train, y_train, x_test, jax.random.PRNGKey(4))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_rff_rejects_nonpositive_num_features() -> None:
    """``num_features`` must be a positive even integer."""
    with pytest.raises(ValueError, match="num_features"):
        rbf_random_fourier_features(
            x=jnp.zeros((3, 1)),
            lengthscale=1.0,
            output_scale=1.0,
            num_features=0,
            rngs=nnx.Rngs(5),
        )


def test_rff_rejects_odd_num_features() -> None:
    """The cos/sin pairing requires an even feature count."""
    with pytest.raises(ValueError, match="even"):
        rbf_random_fourier_features(
            x=jnp.zeros((3, 1)),
            lengthscale=1.0,
            output_scale=1.0,
            num_features=7,
            rngs=nnx.Rngs(6),
        )


def test_rff_predictive_metadata_advertises_rff() -> None:
    """Metadata records ``estimator=rff_gp``."""
    state = fit_rff_gp(
        x_train=jnp.zeros((3, 1)),
        y_train=jnp.zeros((3,)),
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=0.1,
        num_features=32,
        rngs=nnx.Rngs(7),
    )
    predictive = predict_rff_gp(state=state, x_test=jnp.zeros((2, 1)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "rff_gp"
