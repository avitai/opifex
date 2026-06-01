"""Test-time augmentation (TTA) model→UQ adapter.

Mirrors ``test_ensemble_adapters.py``: ``wrap`` validates the fitted-state
container and the capability's ``default_strategy``, then returns a wrapped
model whose ``predict_distribution`` produces an
:class:`opifex.uncertainty.types.PredictiveDistribution` in the same field
shape as :class:`_WrappedDeepEnsembleModel`.

TTA is conceptually an ensemble over a fixed tuple of deterministic input
augmentations: forward each augmented copy of the input through the same
deterministic model and aggregate the cross-augmentation mean / variance.

Reference (PyTorch torch-uncertainty):

* Aggregation —
  ``../torch-uncertainty/src/torch_uncertainty/routines/classification.py``
  (lines 439/446: ``rearrange(logits, "(m b) c -> b m c")`` then
  ``probs_per_est.mean(dim=1)`` — average over the augmentation axis).
* Epistemic via mutual information —
  ``../torch-uncertainty/src/torch_uncertainty/metrics/classification/mutual_information.py``
  (lines 89-93: ``MI = H(mean_m p_m) − mean_m H(p_m)``). This adapter uses
  the regression mean+across-augmentation-variance form (consistent with
  the other model adapters); the classification analogue is the MI form.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    ModelUncertaintyAdapterProtocol,
    TestTimeAugmentationAdapter,
    TestTimeAugmentationState,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


def _model_fn(x: jax.Array) -> jax.Array:
    """Tiny deterministic predictor: ``y = sum(x, axis=-1)``."""
    return jnp.sum(x, axis=-1, keepdims=True)


def _augmentations() -> tuple:
    """Three deterministic augmentations: identity, scale, shift."""
    return (lambda x: x, lambda x: 2.0 * x, lambda x: x + 1.0)


def _make_tta_state() -> TestTimeAugmentationState:
    return TestTimeAugmentationState(model_fn=_model_fn, augmentations=_augmentations())


def _tta_capability() -> UQCapability:
    return UQCapability(
        supports_calibration=True, default_strategy=DefaultStrategy.TEST_TIME_AUGMENTATION
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_tta_adapter_satisfies_protocol() -> None:
    adapter: object = TestTimeAugmentationAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


# ---------------------------------------------------------------------------
# Aggregation correctness — mean / variance across augmentations
# ---------------------------------------------------------------------------


def test_tta_adapter_aggregates_mean_and_variance_across_augmentations() -> None:
    """TTA mean/variance equal the mean/variance over augmented forward passes."""
    state = _make_tta_state()
    adapter = TestTimeAugmentationAdapter()
    wrapped = adapter.wrap(model=state, capability=_tta_capability())
    x = jnp.ones((2, 3))
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)

    augmented = jnp.stack([aug_fn(x) for aug_fn in _augmentations()], axis=0)
    expected_preds = jnp.stack([_model_fn(a) for a in augmented], axis=0)
    expected_mean = jnp.mean(expected_preds, axis=0)
    expected_var = jnp.var(expected_preds, axis=0)

    assert dist.mean.shape == (2, 1)
    assert bool(jnp.allclose(dist.mean, expected_mean))
    assert dist.variance is not None
    assert bool(jnp.allclose(dist.variance, expected_var))
    assert dist.epistemic is not None
    assert bool(jnp.allclose(dist.epistemic, expected_var))
    assert bool(jnp.all(dist.epistemic > 0.0))
    # Aleatoric is zero; total == epistemic for this regression mean+var form.
    assert dist.aleatoric is not None
    assert bool(jnp.all(dist.aleatoric == 0.0))
    assert dist.total_uncertainty is not None
    assert bool(jnp.allclose(dist.total_uncertainty, dist.epistemic))
    # Samples carry the per-augmentation predictions: (n_aug, batch, n_out).
    assert dist.samples is not None
    assert dist.samples.shape == (3, 2, 1)
    # Variance-additivity invariant must hold.
    dist.validate()
    meta = dist.metadata_dict()
    assert meta["method"] == "test_time_augmentation"
    assert meta["source_package"] == "opifex"
    assert int(meta["num_augmentations"]) == 3


# ---------------------------------------------------------------------------
# Capability + state validation
# ---------------------------------------------------------------------------


def test_tta_adapter_rejects_wrong_strategy() -> None:
    state = _make_tta_state()
    adapter = TestTimeAugmentationAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.ENSEMBLE)
    with pytest.raises(ValueError, match="TEST_TIME_AUGMENTATION"):
        adapter.wrap(model=state, capability=capability)


def test_tta_state_rejects_fewer_than_two_augmentations() -> None:
    """A single augmentation yields no spread; ``validate`` must reject it."""
    state = TestTimeAugmentationState(model_fn=_model_fn, augmentations=(lambda x: x,))
    with pytest.raises(ValueError, match="at least 2"):
        state.validate()


def test_tta_adapter_validates_state_on_wrap() -> None:
    """``wrap`` must invoke ``validate`` and reject a single-augmentation state."""
    state = TestTimeAugmentationState(model_fn=_model_fn, augmentations=(lambda x: x,))
    adapter = TestTimeAugmentationAdapter()
    with pytest.raises(ValueError, match="at least 2"):
        adapter.wrap(model=state, capability=_tta_capability())


# ---------------------------------------------------------------------------
# JAX transform compatibility — jit / grad / vmap (pure-array kernel)
# ---------------------------------------------------------------------------


def test_tta_adapter_is_jit_compatible() -> None:
    """Predict mean traces under ``jax.jit`` and returns a finite array."""
    state = _make_tta_state()
    adapter = TestTimeAugmentationAdapter()
    wrapped = adapter.wrap(model=state, capability=_tta_capability())

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    out = predict_mean(jnp.ones((2, 3)))
    assert out.shape == (2, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_tta_adapter_is_grad_compatible() -> None:
    """Gradient of an MSE loss wrt a weight threaded through ``model_fn`` is finite."""
    adapter = TestTimeAugmentationAdapter()
    capability = _tta_capability()

    def loss_fn(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        state = TestTimeAugmentationState(model_fn=lambda z: z @ w, augmentations=_augmentations())
        wrapped = adapter.wrap(model=state, capability=capability)
        pred = wrapped.predict_distribution(x).mean
        return jnp.mean((pred - y) ** 2)

    w = jnp.ones((3, 1))
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 1))
    grad = jax.grad(loss_fn)(w, x, y)
    assert grad.shape == w.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_tta_adapter_is_vmap_compatible() -> None:
    """``predict_distribution`` composes with ``jax.vmap`` over a batched input."""
    state = _make_tta_state()
    adapter = TestTimeAugmentationAdapter()
    wrapped = adapter.wrap(model=state, capability=_tta_capability())

    batched_x = jnp.ones((5, 2, 3))  # leading dim = vmap axis
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (5, 2, 1)
    assert bool(jnp.all(jnp.isfinite(means)))


# ---------------------------------------------------------------------------
# Pytree contract
# ---------------------------------------------------------------------------


def test_tta_state_is_pytree() -> None:
    """``TestTimeAugmentationState`` registers as a pytree and round-trips.

    ``model_fn`` and ``augmentations`` are static (``pytree_node=False``)
    deterministic callables, so the state carries no array leaves — every
    field rides in the static treedef aux_data. The meaningful pytree
    contract is therefore that the state flattens and unflattens cleanly
    through ``jax.tree_util`` with its callables preserved, exactly as the
    sibling all-static ``MCDropoutState`` does.
    """
    state = _make_tta_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    # All fields are static aux_data; no array leaves are exposed.
    assert leaves == []
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, TestTimeAugmentationState)
    assert rebuilt.model_fn is _model_fn
    assert len(rebuilt.augmentations) == 3
    # The rebuilt state still produces the same predictive mean.
    x = jnp.ones((2, 3))
    adapter = TestTimeAugmentationAdapter()
    expected = adapter.wrap(model=state, capability=_tta_capability()).predict_distribution(x)
    actual = adapter.wrap(model=rebuilt, capability=_tta_capability()).predict_distribution(x)
    assert bool(jnp.allclose(actual.mean, expected.mean))
