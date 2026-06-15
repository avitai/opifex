"""Ensemble adapters for deterministic-model UQ.

Includes the working deep-ensemble adapter plus snapshot / SWAG / batch
ensemble / DUE / TTA specs (deferred).

The deep ensemble adapter aggregates the mean and (sample) variance
across a fixed tuple of deterministic-member callables. Member tuples
travel as a ``flax.struct.dataclass`` pytree (pattern (B)) so they can
ride through ``jax.tree_util``-aware transforms.

Spec dataclasses for backends that are not yet wired (snapshot ensemble,
SWAG, BatchEnsemble, DUE, TTA) declare their capability metadata and
raise an actionable :class:`NotImplementedError` from ``wrap`` until
the underlying implementation lands.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.adapters._specs import _DeferredAdapterSpec
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Fitted-state containers
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class DeepEnsembleState:
    """Fixed tuple of independently-trained deterministic-member callables.

    Each member is a pure callable ``x -> y`` (typically a thin wrapper
    around an NNX ``Module`` that closed over its own parameters at
    fitting time). The tuple itself travels as a pytree leaf; the
    flax.struct decorator preserves the structure across transforms.
    """

    members: tuple[Callable[[jax.Array], jax.Array], ...]
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; ensures the ensemble is non-trivial."""
        if len(self.members) < 2:
            raise ValueError(
                f"DeepEnsembleState requires at least 2 members; got {len(self.members)}."
            )


@struct.dataclass(slots=True, kw_only=True)
class SnapshotEnsembleState:
    """Cyclic-LR snapshot ensemble — same shape as DeepEnsembleState."""

    members: tuple[Callable[[jax.Array], jax.Array], ...]
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Raise if fewer than two snapshots have been collected."""
        if len(self.members) < 2:
            raise ValueError(
                f"SnapshotEnsembleState requires at least 2 snapshots; got {len(self.members)}."
            )


@struct.dataclass(slots=True, kw_only=True)
class SWAGState:
    """Stochastic Weight Averaging Gaussian fitted state.

    The first/second moments of the SGD trajectory plus a low-rank
    deviation matrix define a Gaussian posterior over weights. The
    exact reconstruction (and the eventual SWAG forward pass) lands
    with the ``SWAGAdapterSpec`` implementation.
    """

    first_moment: jax.Array
    second_moment: jax.Array
    deviation_matrix: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Raise if first/second moments disagree on shape."""
        if self.first_moment.shape != self.second_moment.shape:
            raise ValueError("SWAGState.first_moment and second_moment must share shape.")


@struct.dataclass(slots=True, kw_only=True)
class BatchEnsembleState:
    """Wen et al. BatchEnsemble — shared kernel + per-member rank-1 factors.

    ``alpha`` and ``gamma`` have shape ``(num_members, ...)`` so that each
    ensemble member's effective weight is ``shared_kernel * outer(alpha_m, gamma_m)``.
    The eventual BatchEnsemble forward pass lands with the spec
    implementation.
    """

    shared_kernel: jax.Array
    alpha: jax.Array
    gamma: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Raise if ``alpha`` and ``gamma`` disagree on the member axis."""
        if self.alpha.shape[0] != self.gamma.shape[0]:
            raise ValueError("BatchEnsembleState.alpha and gamma must agree on the member axis.")


# ---------------------------------------------------------------------------
# Wrapped ensemble model
# ---------------------------------------------------------------------------


class _WrappedDeepEnsembleModel:
    """Predict via per-member forward pass + cross-member mean/variance."""

    def __init__(self, state: DeepEnsembleState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        member_outputs = jnp.stack([member(x) for member in self._state.members], axis=0)
        mean = jnp.mean(member_outputs, axis=0)
        variance = jnp.var(member_outputs, axis=0)
        return PredictiveDistribution(
            mean=mean,
            samples=member_outputs,
            variance=variance,
            epistemic=variance,
            total_uncertainty=variance,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_members", len(self._state.members)),),
            ),
        )


# ---------------------------------------------------------------------------
# DeepEnsembleAdapter (working) + spec stubs for deferred backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepEnsembleAdapter:
    """Deep ensemble adapter: aggregate mean/variance over fixed members."""

    def wrap(self, model: DeepEnsembleState, capability: UQCapability) -> _WrappedDeepEnsembleModel:
        """Wrap a :class:`DeepEnsembleState`; rejects non-``ENSEMBLE`` capabilities."""
        if capability.default_strategy is not DefaultStrategy.ENSEMBLE:
            raise ValueError(
                f"DeepEnsembleAdapter requires default_strategy="
                f"{DefaultStrategy.ENSEMBLE!r}; got "
                f"{capability.default_strategy!r}."
            )
        model.validate()
        return _WrappedDeepEnsembleModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotEnsembleAdapterSpec(_DeferredAdapterSpec):
    """Snapshot ensemble (cyclic-LR snapshots of a single training run)."""

    default_strategy: DefaultStrategy = DefaultStrategy.SNAPSHOT_ENSEMBLE
    required_capabilities: tuple[str, ...] = ("supports_ensemble",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SWAGAdapterSpec(_DeferredAdapterSpec):
    """SWAG: Stochastic Weight Averaging Gaussian posterior."""

    default_strategy: DefaultStrategy = DefaultStrategy.SWAG
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BatchEnsembleAdapterSpec(_DeferredAdapterSpec):
    """BatchEnsemble: rank-1 per-member perturbations on a shared kernel."""

    default_strategy: DefaultStrategy = DefaultStrategy.BATCH_ENSEMBLE
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DUEAdapterSpec(_DeferredAdapterSpec):
    """DUE: Deep kernel + spectral-normalized feature extractor + GP head."""

    default_strategy: DefaultStrategy = DefaultStrategy.DUE
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TestTimeAugmentationAdapterSpec(_DeferredAdapterSpec):
    """Test-time augmentation: average predictions across input augmentations."""

    default_strategy: DefaultStrategy = DefaultStrategy.TEST_TIME_AUGMENTATION
    required_capabilities: tuple[str, ...] = ()


__all__ = [
    "BatchEnsembleAdapterSpec",
    "BatchEnsembleState",
    "DUEAdapterSpec",
    "DeepEnsembleAdapter",
    "DeepEnsembleState",
    "SWAGAdapterSpec",
    "SWAGState",
    "SnapshotEnsembleAdapterSpec",
    "SnapshotEnsembleState",
    "TestTimeAugmentationAdapterSpec",
]
