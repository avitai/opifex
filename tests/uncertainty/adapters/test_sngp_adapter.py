"""Tests for the concrete SNGP (Spectral-normalized Neural Gaussian Process) adapter.

SNGP (Liu et al. NeurIPS 2020, arXiv:2006.10108) replaces a deterministic
network's final layer with a random-feature Gaussian-process approximation:
a fixed random-Fourier-feature (RFF) map ``phi(x)`` feeds a Laplace-approximated
GP whose predictive *variance* is a distance-aware uncertainty signal.

The opifex :class:`SNGPAdapter` wraps an ALREADY-FITTED state carrying

* a frozen deterministic feature map ``phi(x)`` (the RFF layer is fixed, not
  trained at predict time),
* the last-layer weight matrix ``beta`` (``output_weights``), and
* the Laplace precision matrix ``Sigma^{-1}`` (``precision_matrix``).

These tests build the precision matrix via :func:`fit_sngp_precision` (a faithful
port of edward2's ``LaplaceRandomFeatureCovariance`` initial + update logic) and
assert the adapter reproduces the edward2 predictive *exactly*:

* the adapter satisfies :class:`ModelUncertaintyAdapterProtocol`;
* ``mean == phi @ output_weights`` and the per-sample epistemic variance equals
  ``ridge * sum((solve_triangular(chol, phi.T))**2, axis=0)`` cross-checked
  against ``ridge * diag(phi @ inv(precision) @ phi.T)`` — the edward2
  ``compute_predictive_covariance`` chol-solve formula;
* ``fit_sngp_precision`` (gaussian) equals ``ridge*I + phi.T @ phi`` and is SPD;
* ``wrap`` rejects a non-SNGP capability;
* the predictive is ``jit`` / ``vmap`` / ``grad`` safe;
* :class:`SNGPState` is a pytree whose precision / output-weights arrays travel
  through flatten/unflatten while ``feature_fn`` stays static;
* :func:`sngp_mean_field_logits` reproduces edward2 ``mean_field_logits``.

Reference (ported, not invented):
* Liu, J. et al. 2020 — *Simple and Principled Uncertainty Estimation with
  Deterministic Deep Learning via Distance Awareness* (SNGP), arXiv:2006.10108.
* ``../edward2/edward2/jax/nn/random_feature.py``
  (``LaplaceRandomFeatureCovariance.update_precision_matrix``:311-362 and
  ``.compute_predictive_covariance``:364-405).
* ``../edward2/edward2/jax/nn/utils.py`` (``mean_field_logits``:54-101).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    ModelUncertaintyAdapterProtocol,
    SNGPAdapter,
    SNGPState,
)
from opifex.uncertainty.adapters.model import fit_sngp_precision, sngp_mean_field_logits
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


_N_FEATURES = 6  # RFF hidden dimension D.
_N_OUTPUTS = 3
_BATCH = 5
_N_TRAIN = 20
_INPUT_DIM = 2
_RIDGE_PENALTY = 1.0


def _rff_weights() -> tuple[jax.Array, jax.Array]:
    """Fixed random-Fourier-feature projection ``W`` and phase ``b``."""
    key = jax.random.key(0)
    weight_key, bias_key = jax.random.split(key)
    feature_weight = jax.random.normal(weight_key, (_INPUT_DIM, _N_FEATURES))
    feature_bias = jax.random.uniform(bias_key, (_N_FEATURES,), minval=0.0, maxval=2.0 * jnp.pi)
    return feature_weight, feature_bias


_FEATURE_WEIGHT, _FEATURE_BIAS = _rff_weights()


def _feature_fn(x: jax.Array) -> jax.Array:
    """Frozen RFF-style feature map ``phi(x) = cos(x @ W + b)``, shape (b, D)."""
    return jnp.cos(x @ _FEATURE_WEIGHT + _FEATURE_BIAS)


def _train_features() -> jax.Array:
    """Training-feature matrix ``phi_tr`` of shape (n_train, D)."""
    key = jax.random.key(1)
    x_train = jax.random.normal(key, (_N_TRAIN, _INPUT_DIM))
    return _feature_fn(x_train)


def _output_weights() -> jax.Array:
    """Random last-layer weights ``beta`` of shape (D, n_outputs)."""
    key = jax.random.key(2)
    return jax.random.normal(key, (_N_FEATURES, _N_OUTPUTS))


def _make_capability(strategy: DefaultStrategy = DefaultStrategy.SNGP) -> UQCapability:
    return UQCapability(
        default_strategy=strategy,
        source_package="opifex",
        native_nnx_module=True,
        supports_ood_detection=True,
    )


def _make_state(*, observation_noise_variance: float = 0.0) -> SNGPState:
    precision = fit_sngp_precision(
        _train_features(), ridge_penalty=_RIDGE_PENALTY, likelihood="gaussian"
    )
    return SNGPState(
        feature_fn=_feature_fn,
        output_weights=_output_weights(),
        precision_matrix=precision,
        ridge_penalty=_RIDGE_PENALTY,
        observation_noise_variance=observation_noise_variance,
    )


def _make_inputs() -> jax.Array:
    key = jax.random.key(3)
    return jax.random.normal(key, (_BATCH, _INPUT_DIM))


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


def test_sngp_adapter_satisfies_protocol() -> None:
    """``SNGPAdapter`` is a ``ModelUncertaintyAdapterProtocol``."""
    adapter: object = SNGPAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


# ---------------------------------------------------------------------------
# 2. Predictive correctness — edward2 chol-solve formula
# ---------------------------------------------------------------------------


def test_sngp_predict_distribution_matches_edward2_chol_solve() -> None:
    """Mean / epistemic / aleatoric / total match the edward2 predictive formulas."""
    sigma_squared = 0.05
    state = _make_state(observation_noise_variance=sigma_squared)
    wrapped = SNGPAdapter().wrap(state, _make_capability())

    x = _make_inputs()
    distribution = wrapped.predict_distribution(x)
    assert isinstance(distribution, PredictiveDistribution)

    phi = _feature_fn(x)
    expected_mean = phi @ state.output_weights

    # edward2 compute_predictive_covariance (random_feature.py:393-404):
    # chol = cholesky(precision, lower); y = solve_triangular(chol, phi.T);
    # var_diag = ridge * sum(y**2, axis=0).
    chol = jax.scipy.linalg.cholesky(state.precision_matrix, lower=True)
    y = jax.scipy.linalg.solve_triangular(chol, phi.T, lower=True)
    epistemic_scalar = _RIDGE_PENALTY * jnp.sum(y**2, axis=0)
    expected_epistemic = epistemic_scalar[:, None] * jnp.ones((1, _N_OUTPUTS))
    expected_aleatoric = sigma_squared * jnp.ones((_BATCH, _N_OUTPUTS))
    expected_total = expected_epistemic + expected_aleatoric

    # Cross-check: chol-solve equals ridge * diag(phi @ inv(precision) @ phi.T).
    inverse_precision = jnp.linalg.inv(state.precision_matrix)
    dense_scalar = _RIDGE_PENALTY * jnp.diagonal(phi @ inverse_precision @ phi.T)
    assert jnp.allclose(epistemic_scalar, dense_scalar, atol=1e-5)

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
    # Epistemic equals the dense-form quadratic broadcast across outputs.
    assert jnp.allclose(
        distribution.epistemic, dense_scalar[:, None] * jnp.ones((1, _N_OUTPUTS)), atol=1e-5
    )
    assert jnp.allclose(distribution.aleatoric, expected_aleatoric)
    assert jnp.allclose(distribution.total_uncertainty, expected_total)
    # variance mirrors total_uncertainty for the closed-form predictive.
    assert distribution.variance is not None
    assert jnp.allclose(distribution.variance, expected_total)
    # Epistemic strictly positive for an SPD precision + non-degenerate phi.
    assert jnp.all(distribution.epistemic > 0.0)
    # Single representative draw equals the predictive mean.
    assert distribution.samples is not None
    assert distribution.samples.shape == (1, _BATCH, _N_OUTPUTS)
    assert jnp.allclose(distribution.samples[0], expected_mean)


def test_sngp_aleatoric_equals_observation_noise_and_total_is_sum() -> None:
    """``aleatoric == observation_noise_variance``; ``total == epistemic + aleatoric``."""
    sigma_squared = 0.2
    state = _make_state(observation_noise_variance=sigma_squared)
    wrapped = SNGPAdapter().wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    assert distribution.aleatoric is not None
    assert distribution.epistemic is not None
    assert distribution.total_uncertainty is not None
    assert jnp.allclose(distribution.aleatoric, sigma_squared)
    assert jnp.allclose(
        distribution.total_uncertainty, distribution.epistemic + distribution.aleatoric
    )


def test_sngp_satisfies_variance_additivity_contract() -> None:
    """The closed-form output passes ``PredictiveDistribution.validate``."""
    state = _make_state(observation_noise_variance=0.1)
    wrapped = SNGPAdapter().wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    distribution.validate()  # must not raise — total == epistemic + aleatoric.


def test_sngp_metadata_advertises_method() -> None:
    """Metadata identifies the strategy + source package."""
    state = _make_state()
    wrapped = SNGPAdapter().wrap(state, _make_capability())
    distribution = wrapped.predict_distribution(_make_inputs())
    metadata = dict(distribution.metadata)
    assert metadata.get("method") == DefaultStrategy.SNGP.value
    assert metadata.get("source_package") == "opifex"


# ---------------------------------------------------------------------------
# 3. fit_sngp_precision — gaussian Laplace precision
# ---------------------------------------------------------------------------


def test_fit_sngp_precision_gaussian_equals_ridge_identity_plus_gram() -> None:
    """Gaussian precision equals ``ridge*I + phi.T @ phi`` (edward2 exact update)."""
    train_features = _train_features()
    precision = fit_sngp_precision(
        train_features, ridge_penalty=_RIDGE_PENALTY, likelihood="gaussian"
    )
    expected = _RIDGE_PENALTY * jnp.eye(_N_FEATURES) + train_features.T @ train_features
    assert precision.shape == (_N_FEATURES, _N_FEATURES)
    assert jnp.allclose(precision, expected, atol=1e-5)


def test_fit_sngp_precision_is_symmetric_positive_definite() -> None:
    """The precision matrix is symmetric and Cholesky-factorisable (SPD)."""
    precision = fit_sngp_precision(
        _train_features(), ridge_penalty=_RIDGE_PENALTY, likelihood="gaussian"
    )
    assert jnp.allclose(precision, precision.T, atol=1e-5)
    chol = jax.scipy.linalg.cholesky(precision, lower=True)
    assert bool(jnp.all(jnp.isfinite(chol)))
    # Reconstruct: chol @ chol.T == precision.
    assert jnp.allclose(chol @ chol.T, precision, atol=1e-4)


def test_fit_sngp_precision_rejects_unknown_likelihood() -> None:
    """An unsupported likelihood raises ``ValueError``."""
    with pytest.raises(ValueError, match="likelihood"):
        fit_sngp_precision(_train_features(), ridge_penalty=_RIDGE_PENALTY, likelihood="bogus")


def test_fit_sngp_precision_binary_logistic_uses_prob_multiplier() -> None:
    """Binary-logistic precision weights the Gram matrix by ``p*(1-p)``."""
    train_features = _train_features()
    logits = jnp.linspace(-1.0, 1.0, _N_TRAIN).reshape(_N_TRAIN, 1)
    precision = fit_sngp_precision(
        train_features,
        ridge_penalty=_RIDGE_PENALTY,
        logits=logits,
        likelihood="binary_logistic",
    )
    probability = jax.nn.sigmoid(logits)
    multiplier = probability * (1.0 - probability)
    adjusted = jnp.sqrt(multiplier) * train_features
    expected = _RIDGE_PENALTY * jnp.eye(_N_FEATURES) + adjusted.T @ adjusted
    assert jnp.allclose(precision, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. wrap rejects a non-SNGP capability
# ---------------------------------------------------------------------------


def test_sngp_adapter_wrap_rejects_wrong_capability() -> None:
    """``wrap`` raises ``ValueError`` mentioning SNGP on a mismatch."""
    state = _make_state()
    wrong_capability = _make_capability(strategy=DefaultStrategy.DETERMINISTIC)
    with pytest.raises(ValueError, match="SNGP"):
        SNGPAdapter().wrap(state, wrong_capability)


# ---------------------------------------------------------------------------
# 5. jit / grad / vmap transform compatibility (required exit criterion)
# ---------------------------------------------------------------------------


def test_sngp_predict_mean_is_jit_compatible() -> None:
    """``jax.jit`` of the predictive mean returns a finite, correct-shape array."""
    state = _make_state(observation_noise_variance=0.2)
    wrapped = SNGPAdapter().wrap(state, _make_capability())

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    inputs = _make_inputs()
    out = predict_mean(inputs)
    assert out.shape == (_BATCH, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert jnp.allclose(out, _feature_fn(inputs) @ state.output_weights)


def test_sngp_loss_is_grad_compatible_wrt_output_weights() -> None:
    """``jax.grad`` of an MSE loss wrt ``output_weights`` is finite + correct shape."""
    base_state = _make_state(observation_noise_variance=0.0)
    capability = _make_capability()
    x = _make_inputs()
    targets = jnp.zeros((_BATCH, _N_OUTPUTS))

    def loss_fn(output_weights: jax.Array) -> jax.Array:
        state = dataclasses.replace(base_state, output_weights=output_weights)
        wrapped = SNGPAdapter().wrap(state, capability)
        pred = wrapped.predict_distribution(x).mean
        return jnp.mean((pred - targets) ** 2)

    grad = jax.grad(loss_fn)(base_state.output_weights)
    assert grad.shape == base_state.output_weights.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    # Analytic MSE gradient: d/dW mean((phi @ W)^2) = (2 / (N * n_out)) phi^T (phi @ W).
    phi = _feature_fn(x)
    expected_grad = (2.0 / (phi.shape[0] * _N_OUTPUTS)) * (
        phi.T @ (phi @ base_state.output_weights)
    )
    assert jnp.allclose(grad, expected_grad, atol=1e-5)


def test_sngp_predict_is_vmap_compatible() -> None:
    """``jax.vmap`` over a batched input stack works end-to-end."""
    state = _make_state(observation_noise_variance=0.05)
    wrapped = SNGPAdapter().wrap(state, _make_capability())

    batched_x = _make_inputs().reshape(_BATCH, 1, _INPUT_DIM)  # leading axis = vmap axis.
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (_BATCH, 1, _N_OUTPUTS)
    assert bool(jnp.all(jnp.isfinite(means)))


# ---------------------------------------------------------------------------
# 6. State pytree behaviour
# ---------------------------------------------------------------------------


def test_sngp_state_round_trips_through_tree_util() -> None:
    """``precision_matrix`` / ``output_weights`` carry through ``tree_map``; ``feature_fn`` static."""
    state = _make_state(observation_noise_variance=0.3)
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves, "SNGPState must expose at least one pytree leaf"
    # feature_fn is static — it must NOT appear among the pytree leaves.
    assert all(leaf is not _feature_fn for leaf in leaves)
    assert all(not callable(leaf) for leaf in leaves)

    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2.0, state)
    assert bool(jnp.allclose(doubled.output_weights, 2.0 * state.output_weights))
    assert bool(jnp.allclose(doubled.precision_matrix, 2.0 * state.precision_matrix))
    # Identity round-trip preserves the static feature_fn object.
    round_tripped = jax.tree_util.tree_map(lambda leaf: leaf, state)
    assert round_tripped.feature_fn is state.feature_fn
    # Static aux_data survives untouched (not a pytree leaf).
    assert doubled.ridge_penalty == state.ridge_penalty
    assert doubled.observation_noise_variance == state.observation_noise_variance
    assert doubled.metadata == state.metadata
    for leaf in leaves:
        assert leaf is not state.metadata


def test_sngp_state_validate_rejects_non_square_precision() -> None:
    """``validate`` rejects a non-square precision matrix."""
    state = SNGPState(
        feature_fn=_feature_fn,
        output_weights=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        precision_matrix=jnp.zeros((_N_FEATURES, _N_FEATURES + 1)),
    )
    with pytest.raises(ValueError, match="precision_matrix"):
        state.validate()


def test_sngp_state_validate_rejects_dim_mismatch() -> None:
    """``validate`` rejects a precision whose dim mismatches ``output_weights``."""
    state = SNGPState(
        feature_fn=_feature_fn,
        output_weights=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        precision_matrix=jnp.eye(_N_FEATURES + 1),
    )
    with pytest.raises(ValueError, match="output_weights"):
        state.validate()


def test_sngp_state_validate_rejects_non_positive_ridge() -> None:
    """``validate`` rejects a non-positive ``ridge_penalty``."""
    state = SNGPState(
        feature_fn=_feature_fn,
        output_weights=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        precision_matrix=jnp.eye(_N_FEATURES),
        ridge_penalty=0.0,
    )
    with pytest.raises(ValueError, match="ridge_penalty"):
        state.validate()


def test_sngp_state_validate_rejects_negative_noise() -> None:
    """``validate`` rejects a negative ``observation_noise_variance``."""
    state = SNGPState(
        feature_fn=_feature_fn,
        output_weights=jnp.zeros((_N_FEATURES, _N_OUTPUTS)),
        precision_matrix=jnp.eye(_N_FEATURES),
        observation_noise_variance=-1.0,
    )
    with pytest.raises(ValueError, match="observation_noise_variance"):
        state.validate()


# ---------------------------------------------------------------------------
# 7. mean_field_logits — edward2 utils.py port (classification helper)
# ---------------------------------------------------------------------------


def test_sngp_mean_field_logits_logistic_scales_by_sqrt_factor() -> None:
    """Logistic mean-field scaling: ``logits / sqrt(1 + var*factor)`` (edward2 utils:95)."""
    logits = jnp.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    variances = jnp.array([0.4, 1.2])
    factor = 0.5
    adjusted = sngp_mean_field_logits(
        logits, variances, mean_field_factor=factor, likelihood="logistic"
    )
    scale = jnp.sqrt(1.0 + variances * factor)
    expected = logits / scale[:, None]
    assert adjusted.shape == logits.shape
    assert jnp.allclose(adjusted, expected)


def test_sngp_mean_field_logits_poisson_scales_by_exp() -> None:
    """Poisson mean-field scaling: ``logits / exp(-var*factor/2)`` (edward2 utils:93)."""
    logits = jnp.array([[1.0, -2.0], [0.0, 3.0]])
    variances = jnp.array([0.4, 1.2])
    factor = 0.5
    adjusted = sngp_mean_field_logits(
        logits, variances, mean_field_factor=factor, likelihood="poisson"
    )
    scale = jnp.exp(-variances * factor / 2.0)
    expected = logits / scale[:, None]
    assert jnp.allclose(adjusted, expected)


def test_sngp_mean_field_logits_negative_factor_is_identity() -> None:
    """A negative ``mean_field_factor`` returns the logits unchanged (edward2 utils:85-86)."""
    logits = jnp.array([[1.0, -2.0, 0.5]])
    variances = jnp.array([0.4])
    adjusted = sngp_mean_field_logits(
        logits, variances, mean_field_factor=-1.0, likelihood="logistic"
    )
    assert jnp.allclose(adjusted, logits)


def test_sngp_mean_field_logits_rejects_unknown_likelihood() -> None:
    """An unsupported likelihood raises ``ValueError``."""
    logits = jnp.array([[1.0, -2.0]])
    variances = jnp.array([0.4])
    with pytest.raises(ValueError, match="likelihood"):
        sngp_mean_field_logits(logits, variances, mean_field_factor=0.5, likelihood="bogus")
