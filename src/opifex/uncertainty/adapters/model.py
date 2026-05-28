"""Concrete model-uncertainty adapters for deterministic-model UQ.

Includes deterministic and MC-dropout adapters, plus the deferred
Bayesian-last-layer / SNGP / VBLL specs. The concrete diagonal-Laplace
adapter (``LaplaceAdapterSpec`` + ``LaplaceState``) lives in
:mod:`opifex.uncertainty.curvature.laplace` — it is co-located with the
curvature primitives it consumes.

The deterministic adapter is the bookkeeping wrapper that produces a
zero-epistemic :class:`PredictiveDistribution` from a deterministic
callable. The MC-dropout adapter exposes a stochastic model through
caller-owned ``nnx.Rngs`` — no hidden dropout seed — and returns the
sample-mean and sample-variance.

Spec dataclasses for backends that are not yet wired (Bayesian last
layer, SNGP, VBLL) declare their capability metadata and raise an
actionable :class:`NotImplementedError` from ``wrap`` until the
underlying implementation lands.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

from opifex.uncertainty.adapters._specs import _DeferredAdapterSpec
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


if TYPE_CHECKING:
    from collections.abc import Callable


_MCDROPOUT_STREAMS = ("dropout", "sample", "default")


# ---------------------------------------------------------------------------
# Wrapped-model objects (the value ``adapter.wrap(...)`` returns)
# ---------------------------------------------------------------------------


class _WrappedDeterministicModel:
    """Bookkeeping wrapper around a deterministic callable.

    ``predict_distribution`` runs one forward pass and returns a
    :class:`PredictiveDistribution` whose ``epistemic`` field is zero by
    construction (the deterministic strategy advertises no model
    uncertainty).
    """

    def __init__(self, model: Callable[[jax.Array], jax.Array], capability: UQCapability) -> None:
        self._model = model
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        pred = self._model(x)
        zeros = jnp.zeros_like(pred)
        return PredictiveDistribution(
            mean=pred,
            samples=pred[None, ...],
            variance=zeros,
            epistemic=zeros,
            total_uncertainty=zeros,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_samples", 1),),
            ),
        )


class _WrappedMCDropoutModel:
    """Bookkeeping wrapper around an MC-dropout-style stochastic callable."""

    def __init__(self, state: MCDropoutState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        # Caller-owned RNG: extract a single key and split into per-sample
        # keys. No hidden default — passing nothing raises TypeError at
        # the method boundary.
        key = extract_rng_key(
            rngs, streams=_MCDROPOUT_STREAMS, context="MCDropoutAdapter.predict_distribution"
        )
        sample_keys = jax.random.split(key, self._state.num_samples)

        def _draw(_carry: None, sample_key: jax.Array) -> tuple[None, jax.Array]:
            pred = self._state.model_fn(x, rngs=nnx.Rngs(dropout=sample_key))
            return None, pred

        _, samples = jax.lax.scan(_draw, None, sample_keys)
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        return PredictiveDistribution(
            mean=mean,
            samples=samples,
            variance=variance,
            epistemic=variance,
            total_uncertainty=variance,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_samples", int(self._state.num_samples)),),
            ),
        )


# ---------------------------------------------------------------------------
# Adapters (the objects with ``.wrap(model, capability) -> Wrapped*``)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelUncertaintyAdapter:
    """Deterministic-strategy adapter for an arbitrary callable.

    Wrapping is the explicit way to advertise "this surface advertises
    no epistemic uncertainty"; the alternative is a real adapter (deep
    ensemble, MC-dropout, Laplace, SNGP, etc.) — see
    :class:`DeepEnsembleAdapter` and :class:`MCDropoutAdapter`.
    """

    def wrap(
        self, model: Callable[[jax.Array], jax.Array], capability: UQCapability
    ) -> _WrappedDeterministicModel:
        """Wrap a deterministic callable; rejects any non-deterministic capability."""
        if capability.default_strategy is not DefaultStrategy.DETERMINISTIC:
            raise ValueError(
                f"ModelUncertaintyAdapter requires default_strategy="
                f"{DefaultStrategy.DETERMINISTIC!r}; got "
                f"{capability.default_strategy!r}. Use a real adapter "
                f"(DeepEnsembleAdapter / MCDropoutAdapter / LaplaceAdapterSpec / "
                f"SNGPAdapterSpec / VBLLAdapterSpec / BayesianLastLayerAdapterSpec) "
                f"to advertise epistemic uncertainty."
            )
        return _WrappedDeterministicModel(model=model, capability=capability)


class _MCDropoutModelProtocol(Protocol):
    """Callable signature required by MCDropoutAdapter members."""

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array: ...


@struct.dataclass(slots=True, kw_only=True)
class MCDropoutState:
    """Fitted-state pytree for MC-dropout inference.

    ``model_fn`` is a stochastic callable accepting caller-owned
    ``rngs`` and the input batch; ``num_samples`` is the MC sample
    budget per ``predict_distribution`` call.
    """

    model_fn: _MCDropoutModelProtocol = struct.field(pytree_node=False)
    num_samples: int = struct.field(pytree_node=False, default=10)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        if self.num_samples <= 1:
            raise ValueError(
                f"MCDropoutState.num_samples must be > 1 to yield a non-trivial "
                f"variance estimate; got {self.num_samples!r}."
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MCDropoutAdapter:
    """MC-dropout adapter; caller-owned RNG at the predict-time boundary."""

    def wrap(self, model: MCDropoutState, capability: UQCapability) -> _WrappedMCDropoutModel:
        """Wrap an :class:`MCDropoutState`; rejects non-``MC_DROPOUT`` capabilities."""
        if capability.default_strategy is not DefaultStrategy.MC_DROPOUT:
            raise ValueError(
                f"MCDropoutAdapter requires default_strategy="
                f"{DefaultStrategy.MC_DROPOUT!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedMCDropoutModel(state=model, capability=capability)


# ---------------------------------------------------------------------------
# Spec stubs for deferred backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BayesianLastLayerAdapterSpec(_DeferredAdapterSpec):
    """Bayesian-only on the final layer of an otherwise deterministic backbone."""

    default_strategy: DefaultStrategy = DefaultStrategy.BAYESIAN_LAST_LAYER
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SNGPAdapterSpec(_DeferredAdapterSpec):
    """Spectral-Normalized Neural Gaussian Process last layer."""

    default_strategy: DefaultStrategy = DefaultStrategy.SNGP
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VBLLAdapterSpec(_DeferredAdapterSpec):
    """Variational Bayesian Last Layer."""

    default_strategy: DefaultStrategy = DefaultStrategy.VBLL
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


__all__ = [
    "BayesianLastLayerAdapterSpec",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "SNGPAdapterSpec",
    "VBLLAdapterSpec",
]
