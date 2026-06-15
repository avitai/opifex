"""Tests for the concrete Variational Bayesian Last Layer (VBLL) UQ adapter.

The VBLL adapter applies a variational Bayesian treatment to ONLY the
final linear layer over frozen backbone features ``phi(x)``. A Gaussian
posterior ``q(W) = N(weight_mean, Sigma_W)`` over the last-layer weights,
with ``Sigma_W = L @ L.T`` for a lower-triangular Cholesky factor ``L``,
yields a CLOSED-FORM (analytic) regression predictive — no Monte-Carlo
sampling required.

This is the regression closed-form predictive matching the JAX
reference (``../vbll/vbll/jax/layers/regression.py`` +
``../vbll/vbll/jax/utils/distributions.py``). Classification
(MC-softmax marginalization) is out of scope for this adapter — the JAX
reference implements regression only.

Canonical reference:
* Harrison, Willes & Snoek 2024 — *Variational Bayesian Last Layers*,
  arXiv:2404.11599.
* ``DenseNormal.covariance_weighted_inner_prod``:
  ``../vbll/vbll/jax/utils/distributions.py:157-160``.
* Closed-form predictive ``(W() @ x).squeeze + noise()``:
  ``../vbll/vbll/jax/layers/regression.py:61-62``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    ModelUncertaintyAdapterProtocol,
    VBLLAdapter,
    VBLLState,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


_N_FEATURES = 4
_N_OUTPUTS = 3
_BATCH = 5


def _make_capability(
    strategy: DefaultStrategy = DefaultStrategy.VBLL,
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


def _lower_triangular_cholesky() -> jax.Array:
    """Lower-triangular Cholesky factor ``L`` with a positive diagonal.

    ``Sigma_W = L @ L.T`` is then SPD. The off-diagonal entries are
    non-zero so the dense (correlated) covariance is exercised, not a
    diagonal special case.
    """
    key = jax.random.PRNGKey(0)
    dense = jax.random.normal(key, (_N_FEATURES, _N_FEATURES))
    lower = jnp.tril(dense)
    # Force a strictly positive diagonal so L is a valid Cholesky factor.
    positive_diagonal = jnp.abs(jnp.diagonal(dense)) + 0.5
    return lower - jnp.diag(jnp.diagonal(lower)) + jnp.diag(positive_diagonal)


def _make_state(*, observation_noise_variance: float = 0.0) -> VBLLState:
    weight_mean = jnp.linspace(-0.5, 0.5, num=_N_FEATURES * _N_OUTPUTS).reshape(
        _N_FEATURES, _N_OUTPUTS
    )
    return VBLLState(
        feature_fn=_fixed_feature_fn,
        weight_mean=weight_mean,
        weight_covariance_cholesky=_lower_triangular_cholesky(),
        observation_noise_variance=observation_noise_variance,
    )


def _make_inputs(input_dim: int = 2) -> jax.Array:
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (_BATCH, input_dim))


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_vbll_adapter_satisfies_protocol() -> None:
    """``VBLLAdapter`` is a ``ModelUncertaintyAdapterProtocol``."""
    adapter: object = VBLLAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


# ---------------------------------------------------------------------------
# 2. Closed-form predictive correctness
# ---------------------------------------------------------------------------


def test_vbll_predict_distribution_matches_closed_form() -> None:
    """Analytic mean / epistemic / aleatoric / total match the reference formulas."""
    sigma_squared = 0.05
    state = _make_state(observation_noise_variance=sigma_squared)
    adapter = VBLLAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    x = _make_inputs()
    distribution = wrapped.predict_distribution(x)
    assert isinstance(distribution, PredictiveDistribution)

    phi = _fixed_feature_fn(x)
    cholesky = state.weight_covariance_cholesky
    expected_mean = phi @ state.weight_mean

    # L-form (the reference form): epistemic_scalar = sum((phi @ L)**2).
    lt_phi = phi @ cholesky
    epistemic_scalar = jnp.sum(lt_phi**2, axis=-1)
    expected_epistemic = epistemic_scalar[:, None] * jnp.ones((1, _N_OUTPUTS))
    expected_aleatoric = sigma_squared * jnp.ones((_BATCH, _N_OUTPUTS))
    expected_total = expected_epistemic + expected_aleatoric

    # Sigma-form cross-check: phi @ (L @ L.T) @ phi.T diagonal.
    covariance = cholesky @ cholesky.T
    sigma_form_scalar = jnp.einsum("bi,ij,bj->b", phi, covariance, phi)
    assert jnp.allclose(epistemic_scalar, sigma_form_scalar)

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
    # Epistemic equals the Sigma-form quadratic broadcast across outputs.
    assert jnp.allclose(
        distribution.epistemic, sigma_form_scalar[:, None] * jnp.ones((1, _N_OUTPUTS))
    )
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


def test_vbll_satisfies_variance_additivity_contract() -> None:
    """The closed-form output passes ``PredictiveDistribution.validate``."""
    state = _make_state(observation_noise_variance=0.1)
    adapter = VBLLAdapter()
    wrapped = adapter.wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    distribution.validate()  # must not raise — total == epistemic + aleatoric.


def test_vbll_metadata_advertises_method() -> None:
    """Metadata identifies the strategy + source package."""
    state = _make_state()
    adapter = VBLLAdapter()
    wrapped = adapter.wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    metadata = dict(distribution.metadata)
    assert metadata.get("method") == DefaultStrategy.VBLL.value
    assert metadata.get("source_package") == "opifex"


# ---------------------------------------------------------------------------
# 3. wrap rejects a non-VBLL capability
# ---------------------------------------------------------------------------


def test_vbll_adapter_wrap_rejects_wrong_capability() -> None:
    """``wrap`` raises ``ValueError`` mentioning VBLL on a mismatch."""
    state = _make_state()
    adapter = VBLLAdapter()
    wrong_capability = _make_capability(strategy=DefaultStrategy.DETERMINISTIC)
    with pytest.raises(ValueError, match="VBLL"):
        adapter.wrap(state, wrong_capability)


# ---------------------------------------------------------------------------
# 4. jit / grad / vmap transform compatibility (required exit criterion)
# ---------------------------------------------------------------------------


def test_vbll_predict_mean_is_jit_compatible() -> None:
    """``jax.jit`` of the predictive mean returns a finite, correct-shape array."""
    state = _make_state(observation_noise_variance=0.2)
    adapter = VBLLAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    out = predict_mean(_make_inputs())
    assert out.shape == (_BATCH, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_vbll_loss_is_grad_compatible_wrt_weight_mean() -> None:
    """``jax.grad`` of an MSE loss wrt ``weight_mean`` is finite + correct shape."""
    base_state = _make_state(observation_noise_variance=0.0)
    adapter = VBLLAdapter()
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
    # Analytic MSE gradient: d/dW mean((phi @ W - 0)^2) = (2/N) phi^T (phi @ W).
    phi = _fixed_feature_fn(x)
    expected_grad = (2.0 / phi.shape[0]) * (phi.T @ (phi @ base_state.weight_mean))
    expected_grad = expected_grad / _N_OUTPUTS
    assert jnp.allclose(grad, expected_grad, atol=1e-5)


def test_vbll_predict_is_vmap_compatible() -> None:
    """``jax.vmap`` over a batched input stack works end-to-end."""
    state = _make_state(observation_noise_variance=0.05)
    adapter = VBLLAdapter()
    wrapped = adapter.wrap(state, _make_capability())

    batched_x = _make_inputs().reshape(_BATCH, 1, 2)  # leading axis = vmap axis.
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (_BATCH, 1, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(means)))


# ---------------------------------------------------------------------------
# 5. State pytree behaviour
# ---------------------------------------------------------------------------


def test_vbll_state_is_pytree_with_array_leaves() -> None:
    """``tree_leaves`` is non-empty and round-trips through ``tree_map``."""
    state = _make_state(observation_noise_variance=0.3)
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves, "VBLLState must expose at least one pytree leaf"

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2.0, state)
    assert bool(jnp.allclose(doubled.weight_mean, 2.0 * state.weight_mean))
    assert bool(
        jnp.allclose(
            doubled.weight_covariance_cholesky,
            2.0 * state.weight_covariance_cholesky,
        )
    )
    # Static aux_data survives untouched (not a pytree leaf).
    assert doubled.observation_noise_variance == state.observation_noise_variance
    assert doubled.metadata == state.metadata
    for leaf in leaves:
        assert leaf is not state.metadata


def test_vbll_state_validate_rejects_non_lower_triangular_cholesky() -> None:
    """``validate`` rejects a Cholesky factor that is not lower-triangular."""
    upper = jnp.triu(jnp.ones((_N_FEATURES, _N_FEATURES)))  # not lower-triangular.
    state = VBLLState(
        feature_fn=_fixed_feature_fn,
        weight_mean=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        weight_covariance_cholesky=upper,
    )
    with pytest.raises(ValueError, match="lower-triangular"):
        state.validate()


def test_vbll_state_validate_rejects_dim_mismatch() -> None:
    """``validate`` rejects a Cholesky whose dim mismatches ``weight_mean``."""
    state = VBLLState(
        feature_fn=_fixed_feature_fn,
        weight_mean=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        weight_covariance_cholesky=jnp.tril(jnp.ones((_N_FEATURES + 1, _N_FEATURES + 1))),
    )
    with pytest.raises(ValueError, match="weight_covariance_cholesky"):
        state.validate()


def test_vbll_state_validate_rejects_negative_noise() -> None:
    """``validate`` rejects a negative ``observation_noise_variance``."""
    state = VBLLState(
        feature_fn=_fixed_feature_fn,
        weight_mean=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        weight_covariance_cholesky=jnp.tril(jnp.ones((_N_FEATURES, _N_FEATURES))),
        observation_noise_variance=-1.0,
    )
    with pytest.raises(ValueError, match="observation_noise_variance"):
        state.validate()
