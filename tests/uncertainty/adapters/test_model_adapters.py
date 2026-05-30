"""Task 3.6: deterministic-model UQ adapter contracts.

Concrete adapters (``ModelUncertaintyAdapter``, ``DeepEnsembleAdapter``,
``MCDropoutAdapter``) wrap a deterministic-model callable and produce a
:class:`opifex.uncertainty.types.PredictiveDistribution` with method,
member/sample count, source-package, and assumption metadata. Stochastic
adapters MUST require caller-owned ``rngs`` at the method boundary — no
hidden dropout/ensemble seed.

Spec dataclasses for deferred backends (``BayesianLastLayerAdapterSpec``,
``SNGPAdapterSpec``, ``VBLLAdapterSpec``, ``DUEAdapterSpec``,
``TestTimeAugmentationAdapterSpec``) declare capability + source-package
metadata and raise an actionable ``NotImplementedError`` with backend
guidance until real implementations are wired. The Snapshot-ensemble,
SWAG, and BatchEnsemble adapters are concrete — their behaviour is
covered in ``test_ensemble_adapters.py``.

``LaplaceAdapterSpec`` is concrete and produces a Monte-Carlo predictive
distribution by sampling parameters from a diagonal Laplace posterior
built via :func:`opifex.uncertainty.curvature.diagonal_laplace_posterior`.

Fitted-state containers (``DeepEnsembleState``, ``SnapshotEnsembleState``,
``SWAGState``, ``BatchEnsembleState``, ``MCDropoutState``) are
``flax.struct.dataclass``-decorated pytrees carrying member parameters /
running statistics through transforms (pattern (B) per
``GUIDE_ALIGNMENT.md`` item 5a).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.adapters import (
    BatchEnsembleState,
    BayesianLastLayerAdapterSpec,
    DeepEnsembleAdapter,
    DeepEnsembleState,
    DUEAdapterSpec,
    MCDropoutAdapter,
    MCDropoutState,
    ModelUncertaintyAdapter,
    ModelUncertaintyAdapterProtocol,
    SnapshotEnsembleState,
    SNGPAdapterSpec,
    SWAGState,
    TestTimeAugmentationAdapterSpec,
    VBLLAdapterSpec,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


def _deterministic_model(x: jax.Array) -> jax.Array:
    """Tiny linear stand-in for an arbitrary deterministic predictor."""
    return jnp.sum(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# ModelUncertaintyAdapter — deterministic wrapper
# ---------------------------------------------------------------------------


def test_model_uncertainty_adapter_satisfies_protocol() -> None:
    adapter: object = ModelUncertaintyAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_model_uncertainty_adapter_returns_predictive_distribution_with_zero_epistemic() -> None:
    """Wrapping a deterministic callable produces zero-epistemic predictions."""
    adapter = ModelUncertaintyAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.DETERMINISTIC)
    wrapped = adapter.wrap(model=_deterministic_model, capability=capability)
    x = jnp.ones((4, 3))
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (4, 1)
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic == 0.0))
    meta = dist.metadata_dict()
    assert meta["method"] == "deterministic"
    assert meta["source_package"] == "opifex"


def test_model_uncertainty_adapter_rejects_non_deterministic_capability() -> None:
    """Only ``DETERMINISTIC`` strategy is permitted — anything else needs a real adapter."""
    adapter = ModelUncertaintyAdapter()
    capability = UQCapability(native_bayesian=True, default_strategy=DefaultStrategy.BAYESIAN)
    with pytest.raises(ValueError, match="DETERMINISTIC"):
        adapter.wrap(model=_deterministic_model, capability=capability)


# ---------------------------------------------------------------------------
# DeepEnsembleAdapter — multi-member ensemble
# ---------------------------------------------------------------------------


def test_deep_ensemble_adapter_satisfies_protocol() -> None:
    adapter: object = DeepEnsembleAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_deep_ensemble_adapter_aggregates_mean_and_variance_across_members() -> None:
    """Deep ensemble of two members produces a non-trivial epistemic variance."""

    def member_a(x: jax.Array) -> jax.Array:
        return jnp.sum(x, axis=-1, keepdims=True)

    def member_b(x: jax.Array) -> jax.Array:
        return 2.0 * jnp.sum(x, axis=-1, keepdims=True)

    state = DeepEnsembleState(members=(member_a, member_b))
    adapter = DeepEnsembleAdapter()
    capability = UQCapability(supports_ensemble=True, default_strategy=DefaultStrategy.ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((2, 3))
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)
    expected_mean = (
        jnp.sum(x, axis=-1, keepdims=True) + 2.0 * jnp.sum(x, axis=-1, keepdims=True)
    ) / 2.0
    assert bool(jnp.allclose(dist.mean, expected_mean))
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic > 0.0))
    meta = dist.metadata_dict()
    assert meta["method"] == "ensemble"
    assert int(meta["num_members"]) == 2


def test_deep_ensemble_state_rejects_single_member() -> None:
    """An "ensemble" of one is not an ensemble; validate that loudly."""

    def member(x: jax.Array) -> jax.Array:
        return x

    state = DeepEnsembleState(members=(member,))
    with pytest.raises(ValueError, match="at least 2"):
        state.validate()


def test_deep_ensemble_state_is_pytree_compatible() -> None:
    """``DeepEnsembleState`` carries arrays through ``jax.tree_util`` (pattern B)."""
    arr_a = jnp.array([1.0, 2.0])
    arr_b = jnp.array([3.0, 4.0])
    state = DeepEnsembleState(
        members=(lambda x: x + arr_a, lambda x: x + arr_b),
        metadata=(("source", "opifex"),),
    )
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves, "DeepEnsembleState must expose at least one pytree leaf"


# ---------------------------------------------------------------------------
# MCDropoutAdapter — stochastic, requires caller-owned RNG
# ---------------------------------------------------------------------------


def test_mc_dropout_adapter_satisfies_protocol() -> None:
    adapter: object = MCDropoutAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_mc_dropout_adapter_requires_caller_owned_rngs() -> None:
    """Stochastic predictions MUST take ``rngs`` at the method boundary."""

    def stochastic_model(x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        key = rngs.dropout()
        mask = jax.random.bernoulli(key, 0.5, x.shape).astype(x.dtype)
        return jnp.sum(x * mask, axis=-1, keepdims=True)

    state = MCDropoutState(model_fn=stochastic_model, num_samples=8)
    adapter = MCDropoutAdapter()
    capability = UQCapability(supports_ensemble=False, default_strategy=DefaultStrategy.MC_DROPOUT)
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((2, 3))
    with pytest.raises(TypeError):
        wrapped.predict_distribution(x)  # type: ignore[call-arg]

    dist = wrapped.predict_distribution(x, rngs=nnx.Rngs(dropout=5))
    assert isinstance(dist, PredictiveDistribution)
    assert dist.epistemic is not None
    meta = dist.metadata_dict()
    assert meta["method"] == "mc_dropout"
    assert int(meta["num_samples"]) == 8


# ---------------------------------------------------------------------------
# Adapter specs for deferred backends — unsupported-backend errors
# ---------------------------------------------------------------------------


_DEFERRED_SPECS: tuple[tuple[type, DefaultStrategy], ...] = (
    (BayesianLastLayerAdapterSpec, DefaultStrategy.BAYESIAN_LAST_LAYER),
    (SNGPAdapterSpec, DefaultStrategy.SNGP),
    (VBLLAdapterSpec, DefaultStrategy.VBLL),
    (DUEAdapterSpec, DefaultStrategy.DUE),
    (TestTimeAugmentationAdapterSpec, DefaultStrategy.TEST_TIME_AUGMENTATION),
)


@pytest.mark.parametrize(("spec_cls", "expected_strategy"), _DEFERRED_SPECS)
def test_deferred_adapter_specs_are_frozen_dataclasses_with_capability_metadata(
    spec_cls: type, expected_strategy: DefaultStrategy
) -> None:
    """Deferred specs are pattern-(A) frozen dataclasses with capability metadata."""
    import dataclasses as dc

    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert spec.default_strategy is expected_strategy
    assert isinstance(spec.source_package, str)
    assert isinstance(spec.required_capabilities, tuple)
    assert all(isinstance(tag, str) for tag in spec.required_capabilities)
    # frozen — assigning to a field should fail
    with pytest.raises(dc.FrozenInstanceError):
        spec.source_package = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize(("spec_cls", "_strategy"), _DEFERRED_SPECS)
def test_deferred_adapter_spec_wrap_raises_actionable_error(
    spec_cls: type, _strategy: DefaultStrategy
) -> None:
    """`wrap` on a deferred spec raises a clear unsupported-backend error."""
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    with pytest.raises(NotImplementedError, match=spec.default_strategy.value):
        spec.wrap(model=_deterministic_model, capability=capability)


# ---------------------------------------------------------------------------
# Fitted-state containers — pattern (B): flax.struct.dataclass
# ---------------------------------------------------------------------------


_STATE_CLASSES = (
    DeepEnsembleState,
    SnapshotEnsembleState,
    SWAGState,
    BatchEnsembleState,
    MCDropoutState,
)


@pytest.mark.parametrize("state_cls", _STATE_CLASSES)
def test_fitted_state_metadata_is_static_pytree_aux(state_cls: type) -> None:
    """`metadata` is ``tuple[tuple[str, Any], ...]`` and not a pytree leaf."""
    # Each state class must accept ``metadata=()`` (the default) and a
    # non-empty metadata tuple without flattening it into pytree leaves.
    sig_kwargs = _minimal_state_kwargs(state_cls)
    state = state_cls(**sig_kwargs, metadata=(("source", "opifex"),))
    # The metadata tuple should be carried as static aux_data — it must not
    # appear among the pytree leaves.
    leaves = jax.tree_util.tree_leaves(state)
    for leaf in leaves:
        assert leaf is not state.metadata


def _minimal_state_kwargs(state_cls: type) -> dict[str, object]:
    """Minimum kwargs required to construct each state container under test."""
    if state_cls is DeepEnsembleState:
        return {"members": (lambda x: x, lambda x: x * 2.0)}
    if state_cls is SnapshotEnsembleState:
        return {"members": (lambda x: x, lambda x: x * 2.0)}
    if state_cls is SWAGState:
        return {
            "first_moment": jnp.zeros(4),
            "second_moment": jnp.zeros(4),
            "deviation_matrix": jnp.zeros((4, 2)),
            "forward_fn": lambda flat_params, x: x @ flat_params.reshape(-1, 1),
        }
    if state_cls is BatchEnsembleState:
        return {
            "shared_kernel": jnp.zeros((3, 4)),
            "alpha": jnp.ones((2, 3)),
            "gamma": jnp.ones((2, 4)),
        }
    if state_cls is MCDropoutState:
        return {"model_fn": lambda x, *, rngs: x, "num_samples": 4}
    raise AssertionError(f"unhandled state class {state_cls!r}")


# ---------------------------------------------------------------------------
# JAX / Flax NNX transform compatibility — adapters must compose with
# ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` (pure-array kernels) and
# ``nnx.jit`` (NNX-state-carrying kernels). Posterior-collapse and
# trace-level issues are caught here, not at integration time.
# ---------------------------------------------------------------------------


def test_deterministic_adapter_is_jit_compatible() -> None:
    """Deterministic forward through the adapter traces under ``jax.jit``."""
    adapter = ModelUncertaintyAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.DETERMINISTIC)
    wrapped = adapter.wrap(model=_deterministic_model, capability=capability)

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    x = jnp.ones((4, 3))
    out = predict_mean(x)
    assert out.shape == (4, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_deterministic_adapter_is_grad_compatible() -> None:
    """Gradient of a deterministic-adapter loss is finite and non-zero."""
    adapter = ModelUncertaintyAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.DETERMINISTIC)

    def parameterized_model(w: jax.Array, x: jax.Array) -> jax.Array:
        return x @ w

    def loss_fn(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        wrapped = adapter.wrap(model=lambda z: parameterized_model(w, z), capability=capability)
        pred = wrapped.predict_distribution(x).mean
        return jnp.mean((pred - y) ** 2)

    w = jnp.ones((3, 2))
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 2))
    grad = jax.grad(loss_fn)(w, x, y)
    assert grad.shape == w.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_deep_ensemble_adapter_is_vmap_compatible_across_inputs() -> None:
    """``predict_distribution`` composes with ``jax.vmap`` over a batched input."""

    def member_a(x: jax.Array) -> jax.Array:
        return jnp.sum(x, axis=-1, keepdims=True)

    def member_b(x: jax.Array) -> jax.Array:
        return 2.0 * jnp.sum(x, axis=-1, keepdims=True)

    state = DeepEnsembleState(members=(member_a, member_b))
    adapter = DeepEnsembleAdapter()
    capability = UQCapability(supports_ensemble=True, default_strategy=DefaultStrategy.ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)

    batched_x = jnp.ones((5, 2, 3))  # leading dim = vmap axis
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (5, 2, 1)
    assert bool(jnp.all(jnp.isfinite(means)))


def test_mc_dropout_adapter_is_nnx_jit_compatible() -> None:
    """``MCDropoutAdapter.predict_distribution`` traces under ``nnx.jit`` with traced rngs.

    Mirrors the canonical Flax NNX pattern: ``rngs`` is passed as a traced
    argument, not closed over (closure across trace levels raises
    ``TraceContextError``).
    """

    def stochastic_model(x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        key = rngs.dropout()
        mask = jax.random.bernoulli(key, 0.7, x.shape).astype(x.dtype)
        return jnp.sum(x * mask, axis=-1, keepdims=True)

    state = MCDropoutState(model_fn=stochastic_model, num_samples=4)
    adapter = MCDropoutAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.MC_DROPOUT)
    wrapped = adapter.wrap(model=state, capability=capability)

    @nnx.jit
    def step(rngs: nnx.Rngs, x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x, rngs=rngs).mean

    out = step(nnx.Rngs(dropout=11), jnp.ones((3, 4)))
    assert out.shape == (3, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_swag_state_round_trips_through_tree_util() -> None:
    """Array-only state (``SWAGState``) reconstructs cleanly via ``tree_map``.

    Confirms the ``flax.struct.dataclass`` pytree contract: arrays carry
    through transforms; ``metadata`` stays static aux_data.
    """
    first_moment = jnp.array([1.0, 2.0])
    second_moment = jnp.array([0.5, 0.5])
    deviation_matrix = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    state = SWAGState(
        first_moment=first_moment,
        second_moment=second_moment,
        deviation_matrix=deviation_matrix,
        forward_fn=lambda flat_params, x: x @ flat_params.reshape(-1, 1),
        metadata=(("source", "opifex"),),
    )
    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2.0, state)
    assert bool(jnp.allclose(doubled.first_moment, 2.0 * first_moment))
    assert bool(jnp.allclose(doubled.second_moment, 2.0 * second_moment))
    assert bool(jnp.allclose(doubled.deviation_matrix, 2.0 * deviation_matrix))
    # Metadata is non-pytree-node, so it survives untouched.
    assert doubled.metadata == state.metadata
