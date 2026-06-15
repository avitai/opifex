"""Tests for the concrete LaplaceAdapterSpec backed by curvature primitives.

The diagonal Laplace adapter wraps a pre-fitted ``LaplaceState`` (MAP
estimate, posterior precision, model function) into a
``PredictiveDistribution`` provider. Sampling draws parameters from the
diagonal Gaussian posterior ``N(θ*, diag(1/precision))`` and computes
the predictive ensemble via ``vmap`` over the model function.

Canonical reference:
* Daxberger Laplace package and bayesian-torch use this same recipe;
  opifex builds it on top of ``diagonal_laplace_posterior`` from
  ``opifex.uncertainty.curvature``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.adapters import LaplaceAdapterSpec, LaplaceState
from opifex.uncertainty.curvature import DiagonalLaplacePosterior
from opifex.uncertainty.registry import DefaultStrategy, UQCapability


def _make_capability(strategy: DefaultStrategy = DefaultStrategy.LAPLACE) -> UQCapability:
    return UQCapability(
        default_strategy=strategy,
        source_package="opifex.uncertainty.curvature",
        native_nnx_module=True,
    )


def _linear_model(parameters: jax.Array, x: jax.Array) -> jax.Array:
    return x @ parameters


def test_laplace_adapter_wrap_rejects_non_laplace_capability() -> None:
    """``wrap`` raises ``ValueError`` when the capability strategy is wrong."""
    state = LaplaceState(
        model_fn=_linear_model,
        posterior=DiagonalLaplacePosterior(mean=jnp.zeros(2), precision_diagonal=jnp.ones(2)),
    )
    adapter = LaplaceAdapterSpec()
    wrong_capability = _make_capability(strategy=DefaultStrategy.DETERMINISTIC)
    with pytest.raises(ValueError, match="LaplaceAdapterSpec requires"):
        adapter.wrap(state, wrong_capability)


def test_laplace_adapter_predict_distribution_returns_samples_and_variance() -> None:
    """``predict_distribution`` returns sample mean + epistemic variance > 0."""
    map_estimate = jnp.asarray([1.0, -0.5])
    precision = jnp.asarray([4.0, 9.0])  # variance = [0.25, 1/9]
    state = LaplaceState(
        model_fn=_linear_model,
        posterior=DiagonalLaplacePosterior(mean=map_estimate, precision_diagonal=precision),
        num_samples=128,
    )
    adapter = LaplaceAdapterSpec()
    wrapped = adapter.wrap(state, _make_capability())

    x = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    distribution = wrapped.predict_distribution(x, rngs=nnx.Rngs(params=0))

    assert distribution.samples is not None and distribution.samples.shape == (128, 2)
    assert distribution.mean.shape == (2,)
    # Sample mean should be close to MAP prediction; variance > 0.
    map_prediction = _linear_model(map_estimate, x)
    assert jnp.allclose(distribution.mean, map_prediction, atol=0.1)
    assert distribution.variance is not None
    assert jnp.all(distribution.variance > 0.0)
    assert distribution.epistemic is not None
    assert jnp.all(distribution.epistemic > 0.0)


def test_laplace_adapter_zero_variance_at_infinite_precision_limit() -> None:
    """With very large precision, posterior samples collapse to the MAP point."""
    map_estimate = jnp.asarray([2.0, -1.0])
    huge_precision = jnp.asarray([1e10, 1e10])
    state = LaplaceState(
        model_fn=_linear_model,
        posterior=DiagonalLaplacePosterior(mean=map_estimate, precision_diagonal=huge_precision),
        num_samples=64,
    )
    adapter = LaplaceAdapterSpec()
    wrapped = adapter.wrap(state, _make_capability())
    x = jnp.eye(2)
    distribution = wrapped.predict_distribution(x, rngs=nnx.Rngs(params=1))
    map_prediction = _linear_model(map_estimate, x)
    assert jnp.allclose(distribution.mean, map_prediction, atol=1e-3)
    assert distribution.variance is not None
    assert jnp.all(distribution.variance < 1e-4)


def test_laplace_adapter_predict_distribution_metadata_advertises_method() -> None:
    """Metadata identifies the strategy + source package + sample count."""
    state = LaplaceState(
        model_fn=_linear_model,
        posterior=DiagonalLaplacePosterior(mean=jnp.zeros(2), precision_diagonal=jnp.ones(2)),
        num_samples=16,
    )
    adapter = LaplaceAdapterSpec()
    wrapped = adapter.wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(jnp.eye(2), rngs=nnx.Rngs(params=2))
    metadata = dict(distribution.metadata)
    assert metadata.get("method") == DefaultStrategy.LAPLACE.value
    assert metadata.get("num_samples") == 16


def test_laplace_adapter_predict_distribution_is_jit_compatible() -> None:
    """``predict_distribution`` runs inside ``jax.jit`` end-to-end."""
    state = LaplaceState(
        model_fn=_linear_model,
        posterior=DiagonalLaplacePosterior(
            mean=jnp.asarray([0.5, 1.0]), precision_diagonal=jnp.asarray([2.0, 2.0])
        ),
        num_samples=8,
    )
    adapter = LaplaceAdapterSpec()
    wrapped = adapter.wrap(state, _make_capability())

    @jax.jit
    def call(x: jax.Array, key: jax.Array) -> jax.Array:
        variance = wrapped.predict_distribution(x, rngs=nnx.Rngs(params=key)).variance
        assert variance is not None
        return variance

    variance = call(jnp.eye(2), jax.random.PRNGKey(3))
    assert jnp.all(jnp.isfinite(variance))
