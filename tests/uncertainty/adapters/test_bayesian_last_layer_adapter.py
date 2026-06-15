"""Tests for the concrete Bayesian-last-layer (BLL) UQ adapter.

The BLL adapter applies a Bayesian treatment to ONLY the final linear
layer over frozen backbone features ``phi(x)``. A Gaussian posterior
``N(weight_mean, weight_covariance)`` over the last-layer weights yields
a CLOSED-FORM (analytic) predictive — no Monte-Carlo sampling required.

This is the GLM / linearised-Laplace pushforward for a linear head
(equivalently the neural-linear model). It is DISTINCT from the
full-network diagonal-Laplace adapter in
``opifex.uncertainty.curvature.laplace`` — that one Monte-Carlo-samples
parameters; this one evaluates the analytic last-layer predictive.

Canonical references:
* Ober & Rasmussen 2019 — *Benchmarking the Neural Linear Model*.
* Snoek et al. 2015 — *Scalable Bayesian Optimization Using DNNs*.
* Daxberger et al. 2021 — *Laplace Redux* (last-layer Laplace).
* ``../laplax/laplax/eval/pushforward.py`` (GLM ``J @ Sigma @ J^T`` cov).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    BayesianLastLayerAdapter,
    BayesianLastLayerState,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


_N_FEATURES = 4
_N_OUTPUTS = 3
_BATCH = 5


def _make_capability(
    strategy: DefaultStrategy = DefaultStrategy.BAYESIAN_LAST_LAYER,
) -> UQCapability:
    return UQCapability(
        default_strategy=strategy,
        source_package="opifex",
        native_nnx_module=True,
    )


def _fixed_feature_fn(x: jax.Array) -> jax.Array:
    """Frozen deterministic backbone: ``phi(x) = tanh(x @ W)``."""
    feature_weight = jnp.linspace(-1.0, 1.0, num=x.shape[-1] * _N_FEATURES)
    feature_weight = feature_weight.reshape(x.shape[-1], _N_FEATURES)
    return jnp.tanh(x @ feature_weight)


def _spd_covariance() -> jax.Array:
    """SPD posterior covariance ``Sigma = L @ L.T + 1e-2 * I``."""
    key = jax.random.PRNGKey(0)
    lower = jax.random.normal(key, (_N_FEATURES, _N_FEATURES))
    return lower @ lower.T + 1e-2 * jnp.eye(_N_FEATURES)


def _make_state(*, observation_noise_variance: float = 0.0) -> BayesianLastLayerState:
    weight_mean = jnp.linspace(-0.5, 0.5, num=_N_FEATURES * _N_OUTPUTS).reshape(
        _N_FEATURES, _N_OUTPUTS
    )
    return BayesianLastLayerState(
        feature_fn=_fixed_feature_fn,
        weight_mean=weight_mean,
        weight_covariance=_spd_covariance(),
        observation_noise_variance=observation_noise_variance,
    )


def _make_inputs(input_dim: int = 2) -> jax.Array:
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (_BATCH, input_dim))


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_bayesian_last_layer_adapter_satisfies_protocol() -> None:
    """``BayesianLastLayerAdapter`` is a ``ModelUncertaintyAdapterProtocol``."""
    adapter: object = BayesianLastLayerAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


# ---------------------------------------------------------------------------
# 2. Closed-form predictive correctness
# ---------------------------------------------------------------------------


def test_bayesian_last_layer_predict_distribution_matches_closed_form() -> None:
    """Analytic mean / epistemic / aleatoric / total match the reference formulas."""
    sigma_squared = 0.05
    state = _make_state(observation_noise_variance=sigma_squared)
    adapter = BayesianLastLayerAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    x = _make_inputs()
    distribution = wrapped.predict_distribution(x)
    assert isinstance(distribution, PredictiveDistribution)

    phi = _fixed_feature_fn(x)
    expected_mean = phi @ state.weight_mean
    quadratic = jnp.einsum("bi,ij,bj->b", phi, state.weight_covariance, phi)
    expected_epistemic = quadratic[:, None] * jnp.ones((1, _N_OUTPUTS))
    expected_aleatoric = sigma_squared * jnp.ones((_BATCH, _N_OUTPUTS))
    expected_total = expected_epistemic + expected_aleatoric

    # Shapes: (batch, n_outputs).
    assert distribution.mean.shape == (_BATCH, _N_OUTPUTS)
    assert distribution.epistemic is not None
    assert distribution.epistemic.shape == (_BATCH, _N_OUTPUTS)
    assert distribution.aleatoric is not None
    assert distribution.aleatoric.shape == (_BATCH, _N_OUTPUTS)
    assert distribution.total_uncertainty is not None
    assert distribution.total_uncertainty.shape == (_BATCH, _N_OUTPUTS)

    assert jnp.allclose(distribution.mean, expected_mean)
    assert jnp.allclose(distribution.epistemic, expected_epistemic)
    assert jnp.allclose(distribution.aleatoric, expected_aleatoric)
    assert jnp.allclose(distribution.total_uncertainty, expected_total)
    # variance mirrors total_uncertainty for the closed-form predictive.
    assert distribution.variance is not None
    assert jnp.allclose(distribution.variance, expected_total)
    # Epistemic strictly positive for an SPD covariance + non-degenerate phi.
    assert jnp.all(distribution.epistemic > 0.0)
    # Single representative draw equals the predictive mean.
    assert distribution.samples is not None
    assert distribution.samples.shape == (1, _BATCH, _N_OUTPUTS)
    assert jnp.allclose(distribution.samples[0], expected_mean)


def test_bayesian_last_layer_satisfies_variance_additivity_contract() -> None:
    """The closed-form output passes ``PredictiveDistribution.validate``."""
    state = _make_state(observation_noise_variance=0.1)
    adapter = BayesianLastLayerAdapter()
    wrapped = adapter.wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    distribution.validate()  # must not raise — total == epistemic + aleatoric.


def test_bayesian_last_layer_metadata_advertises_method() -> None:
    """Metadata identifies the strategy + source package."""
    state = _make_state()
    adapter = BayesianLastLayerAdapter()
    wrapped = adapter.wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    metadata = dict(distribution.metadata)
    assert metadata.get("method") == DefaultStrategy.BAYESIAN_LAST_LAYER.value
    assert metadata.get("source_package") == "opifex"


# ---------------------------------------------------------------------------
# 3. wrap rejects a non-BAYESIAN_LAST_LAYER capability
# ---------------------------------------------------------------------------


def test_bayesian_last_layer_adapter_wrap_rejects_wrong_capability() -> None:
    """``wrap`` raises ``ValueError`` mentioning the strategy on a mismatch."""
    state = _make_state()
    adapter = BayesianLastLayerAdapter()
    wrong_capability = _make_capability(strategy=DefaultStrategy.DETERMINISTIC)
    with pytest.raises(ValueError, match="BAYESIAN_LAST_LAYER"):
        adapter.wrap(state, wrong_capability)


# ---------------------------------------------------------------------------
# 4. jit / grad / vmap transform compatibility (required exit criterion)
# ---------------------------------------------------------------------------


def test_bayesian_last_layer_predict_mean_is_jit_compatible() -> None:
    """``jax.jit`` of the predictive mean returns a finite, correct-shape array."""
    state = _make_state(observation_noise_variance=0.2)
    adapter = BayesianLastLayerAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    out = predict_mean(_make_inputs())
    assert out.shape == (_BATCH, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_bayesian_last_layer_loss_is_grad_compatible_wrt_weight_mean() -> None:
    """``jax.grad`` of an MSE loss wrt ``weight_mean`` is finite + correct shape."""
    base_state = _make_state(observation_noise_variance=0.0)
    adapter = BayesianLastLayerAdapter()
    capability = _make_capability()
    x = _make_inputs()
    targets = jnp.zeros((_BATCH, _N_OUTPUTS))

    def loss_fn(weight_mean: jax.Array) -> jax.Array:
        state = dataclasses.replace(base_state, weight_mean=weight_mean)
        wrapped = adapter.wrap(state, capability)
        pred = wrapped.predict_distribution(x).mean
        return jnp.mean((pred - targets) ** 2)

    grad = jax.grad(loss_fn)(base_state.weight_mean)
    assert grad.shape == base_state.weight_mean.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_bayesian_last_layer_predict_is_vmap_compatible() -> None:
    """``jax.vmap`` over a batched input stack works end-to-end."""
    state = _make_state(observation_noise_variance=0.05)
    adapter = BayesianLastLayerAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    batched_x = _make_inputs().reshape(_BATCH, 1, 2)  # leading axis = vmap axis.
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (_BATCH, 1, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(means)))


# ---------------------------------------------------------------------------
# 5. State pytree behaviour
# ---------------------------------------------------------------------------


def test_bayesian_last_layer_state_is_pytree_with_array_leaves() -> None:
    """``tree_leaves`` is non-empty and round-trips through ``tree_map``."""
    state = _make_state(observation_noise_variance=0.3)
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves, "BayesianLastLayerState must expose at least one pytree leaf"

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2.0, state)
    assert bool(jnp.allclose(doubled.weight_mean, 2.0 * state.weight_mean))
    assert bool(jnp.allclose(doubled.weight_covariance, 2.0 * state.weight_covariance))
    # Static aux_data survives untouched (not a pytree leaf).
    assert doubled.observation_noise_variance == state.observation_noise_variance
    assert doubled.metadata == state.metadata
    for leaf in leaves:
        assert leaf is not state.metadata


def test_bayesian_last_layer_state_validate_rejects_non_square_covariance() -> None:
    """``validate`` rejects a covariance whose dim mismatches ``weight_mean``."""
    state = BayesianLastLayerState(
        feature_fn=_fixed_feature_fn,
        weight_mean=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        weight_covariance=jnp.eye(_N_FEATURES + 1),  # wrong dim.
    )
    with pytest.raises(ValueError, match="weight_covariance"):
        state.validate()


def test_bayesian_last_layer_state_validate_rejects_negative_noise() -> None:
    """``validate`` rejects a negative ``observation_noise_variance``."""
    state = BayesianLastLayerState(
        feature_fn=_fixed_feature_fn,
        weight_mean=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        weight_covariance=jnp.eye(_N_FEATURES),
        observation_noise_variance=-1.0,
    )
    with pytest.raises(ValueError, match="observation_noise_variance"):
        state.validate()
