"""Ensemble adapters for deterministic-model UQ.

Includes the deep-ensemble, snapshot-ensemble, SWAG, BatchEnsemble, and
test-time-augmentation (TTA) adapters. The DUE (Deterministic Uncertainty
Estimation) adapter is concrete and lives in
:mod:`opifex.uncertainty.adapters.model` alongside the other deep-kernel /
last-layer adapters it shares the wrapped-model contract with.

The deep / snapshot ensemble adapters aggregate the mean and (sample)
variance across a fixed tuple of deterministic-member callables. SWAG
samples weights from a low-rank-plus-diagonal Gaussian posterior and
forwards each draw; BatchEnsemble forwards over ``M`` rank-1 fast-weight
members of a single shared kernel. Test-time augmentation forwards a single
deterministic model over a fixed tuple of deterministic input augmentations
and aggregates across the augmentation axis (an ensemble over
augmentations). Fitted-state containers travel as ``flax.struct.dataclass``
pytrees (pattern (B)) so they can ride through ``jax.tree_util``-aware
transforms.

References:
    * Deep ensembles — Lakshminarayanan, Pritzel, Blundell, "Simple and
      Scalable Predictive Uncertainty Estimation using Deep Ensembles",
      NeurIPS 2017 (arXiv:1612.01474).
    * Snapshot ensembles — Huang, Li, Pleiss, Liu, Hopcroft, Weinberger,
      "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017
      (arXiv:1704.00109).
    * SWAG — Maddox, Garipov, Izmailov, Vetrov, Wilson, "A Simple
      Baseline for Bayesian Uncertainty in Deep Learning", NeurIPS 2019
      (arXiv:1902.02476).
    * BatchEnsemble — Wen, Tran, Ba, "BatchEnsemble: An Alternative
      Approach to Efficient Ensemble and Lifelong Learning", ICLR 2020
      (arXiv:2002.06715).
    * Test-time augmentation — Wang, Aitchison, Rutherford, …, "Aleatoric
      uncertainty estimation with test-time augmentation for medical image
      segmentation with convolutional neural networks", Neurocomputing 2019.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

from opifex.uncertainty._predictive import ensemble_predictive
from opifex.uncertainty.registry import DefaultStrategy, UQCapability


if TYPE_CHECKING:
    from collections.abc import Callable

    from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


_SWAG_STREAMS = ("sample", "default")


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


class _SWAGForwardProtocol(Protocol):
    """Map a flat weight vector + input batch to a prediction.

    ``forward_fn(flat_params, x) -> y`` reconstructs the model parameters
    from the flattened SWAG weight draw and runs the forward pass. The
    caller owns the reshape/unflatten (typically ``nnx.split`` /
    ``ravel_pytree``) so this adapter stays a pure-array kernel.
    """

    def __call__(self, flat_params: jax.Array, x: jax.Array) -> jax.Array: ...


@struct.dataclass(slots=True, kw_only=True)
class SWAGState:
    """Stochastic Weight Averaging Gaussian fitted state.

    The first/second moments of the SGD trajectory plus a low-rank
    deviation matrix define a Gaussian posterior over weights
    (Maddox et al. NeurIPS 2019, arXiv:1902.02476):
    ``θ ~ N(θ_SWA, ½(Σ_diag + Σ_lowrank))`` where
    ``Σ_diag = diag(second_moment - first_moment²)`` and
    ``Σ_lowrank = D Dᵀ / (K - 1)`` with ``D = deviation_matrix`` of
    shape ``(num_params, K)``.

    ``forward_fn`` maps a sampled flat weight vector + input to a
    prediction; ``num_samples`` is the Monte-Carlo weight-draw budget per
    ``predict_distribution`` call.
    """

    first_moment: jax.Array
    second_moment: jax.Array
    deviation_matrix: jax.Array
    forward_fn: _SWAGForwardProtocol = struct.field(pytree_node=False)
    num_samples: int = struct.field(pytree_node=False, default=30)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Raise if moments disagree on shape or the sample budget is trivial."""
        if self.first_moment.shape != self.second_moment.shape:
            raise ValueError("SWAGState.first_moment and second_moment must share shape.")
        if self.deviation_matrix.shape[0] != self.first_moment.shape[0]:
            raise ValueError(
                "SWAGState.deviation_matrix leading axis must match the parameter "
                f"count {self.first_moment.shape[0]}; got {self.deviation_matrix.shape[0]}."
            )
        if self.num_samples <= 1:
            raise ValueError(
                "SWAGState.num_samples must be > 1 to yield a non-trivial variance "
                f"estimate; got {self.num_samples!r}."
            )


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


@struct.dataclass(slots=True, kw_only=True)
class TestTimeAugmentationState:
    """Deterministic test-time augmentation (TTA) fitted state.

    A single deterministic model is evaluated on a fixed tuple of
    deterministic input augmentations (identity, flips, scales, …) and the
    predictions are aggregated across augmentations — conceptually an
    ensemble over augmentations (Wang et al., Neurocomputing 2019). Both
    ``model_fn`` and the augmentation callables are static
    (``pytree_node=False``) and deterministic, so the wrapper carries no RNG
    and composes with ``jit`` / ``grad`` / ``vmap``.

    ``augmentations`` each map ``x -> augmented_x`` (same shape as ``x``);
    ``model_fn`` maps ``x -> y``.
    """

    model_fn: Callable[[jax.Array], jax.Array] = struct.field(pytree_node=False)
    augmentations: tuple[Callable[[jax.Array], jax.Array], ...] = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Raise if fewer than two augmentations are configured.

        A single augmentation produces a degenerate one-member ensemble with
        zero spread; require at least two to yield a non-trivial variance.
        """
        if len(self.augmentations) < 2:
            raise ValueError(
                "TestTimeAugmentationState requires at least 2 augmentations to yield a "
                f"non-trivial across-augmentation variance; got {len(self.augmentations)}. "
                "Supply two or more augmentation callables (e.g. identity plus flips/scales)."
            )


# ---------------------------------------------------------------------------
# Shared aggregation helper (Rule 1: DRY)
# ---------------------------------------------------------------------------


def _predictive_from_member_samples(
    *,
    samples: jax.Array,
    capability: UQCapability,
    extra_metadata: MetadataItems,
    include_zero_aleatoric: bool = False,
) -> PredictiveDistribution:
    """Aggregate a stack of member predictions into a :class:`PredictiveDistribution`.

    Members live on ``axis=0`` (shape ``(num_members, batch, ...)``). The
    capability supplies the ``method`` (its ``default_strategy`` value) and
    ``source_package`` provenance; the actual mean / variance reduction and
    variance decomposition are delegated to the shared
    :func:`opifex.uncertainty._predictive.ensemble_predictive` constructor so
    the aggregation lives in exactly one place (Rule 1 — DRY) and is shared
    with the quantum-chemistry UQ surfaces.

    When ``include_zero_aleatoric`` is set, an explicit zero ``aleatoric``
    array is emitted so that the variance-additivity invariant
    ``total_uncertainty == epistemic + aleatoric`` is satisfied with a
    materialised aleatoric term (used by the test-time-augmentation wrapper).
    """
    return ensemble_predictive(
        samples,
        method=capability.default_strategy.value,
        source_package=capability.source_package,
        extra_metadata=extra_metadata,
        include_zero_aleatoric=include_zero_aleatoric,
    )


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
        return _predictive_from_member_samples(
            samples=member_outputs,
            capability=self._capability,
            extra_metadata=(("num_members", len(self._state.members)),),
        )


class _WrappedSnapshotEnsembleModel:
    """Predict by averaging over cyclic-LR snapshots (Huang et al. 2017).

    Algorithmically identical to a deep ensemble at inference time — the
    distinction is that the snapshots come from a single training run with
    a cyclic learning-rate schedule rather than independent runs.
    """

    def __init__(self, state: SnapshotEnsembleState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        snapshot_outputs = jnp.stack([snapshot(x) for snapshot in self._state.members], axis=0)
        return _predictive_from_member_samples(
            samples=snapshot_outputs,
            capability=self._capability,
            extra_metadata=(("num_snapshots", len(self._state.members)),),
        )


class _WrappedSWAGModel:
    """Predict by sampling weights from the SWAG Gaussian and forwarding each.

    Implements the SWAG predictive draw (Maddox et al. NeurIPS 2019,
    arXiv:1902.02476, eq. 1), cross-checked against
    ``../torch-uncertainty/src/torch_uncertainty/methods/swag.py``
    (``SWAG._fullrank_sample``):

    ``θ̃ = θ_SWA + (1/√2)·√Σ_diag·z₁ + (1/√(2(K-1)))·D·z₂``,
    ``z₁ ~ N(0, I_P)``, ``z₂ ~ N(0, I_K)``.

    The ``½`` scaling on both covariance terms is the published default
    (``swa_gaussian`` repo). torch-uncertainty folds the diagonal ``1/√2``
    into a tunable ``scale`` defaulting to ``1.0``; we keep the canonical
    ``1/√2`` to match the paper's ``½(Σ_diag + Σ_lowrank)`` posterior.
    """

    def __init__(self, state: SWAGState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def _sample_weights(self, key: jax.Array) -> jax.Array:
        """Draw ``num_samples`` weight vectors from the SWAG Gaussian."""
        first_moment = self._state.first_moment
        diag_variance = jnp.clip(self._state.second_moment - first_moment**2, min=0.0)
        deviation = self._state.deviation_matrix
        rank = deviation.shape[-1]
        diag_key, lowrank_key = jax.random.split(key)

        num_params = first_moment.shape[0]
        diag_noise = jax.random.normal(diag_key, (self._state.num_samples, num_params))
        lowrank_noise = jax.random.normal(lowrank_key, (self._state.num_samples, rank))

        # Diagonal term: (1/√2) √Σ_diag z₁.
        diag_term = jnp.sqrt(0.5 * diag_variance) * diag_noise
        # Low-rank term: (1/√(2(K-1))) D z₂. Guard K == 1 (no low-rank info).
        lowrank_scale = jnp.where(rank > 1, 1.0 / jnp.sqrt(2.0 * jnp.maximum(rank - 1, 1)), 0.0)
        lowrank_term = lowrank_scale * (lowrank_noise @ deviation.T)
        return first_moment[None, :] + diag_term + lowrank_term

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        key = extract_rng_key(
            rngs, streams=_SWAG_STREAMS, context="SWAGAdapter.predict_distribution"
        )
        weight_samples = self._sample_weights(key)

        def _forward(_carry: None, flat_params: jax.Array) -> tuple[None, jax.Array]:
            return None, self._state.forward_fn(flat_params, x)

        _, samples = jax.lax.scan(_forward, None, weight_samples)
        return _predictive_from_member_samples(
            samples=samples,
            capability=self._capability,
            extra_metadata=(("num_samples", int(self._state.num_samples)),),
        )


class _WrappedBatchEnsembleModel:
    """Predict over ``M`` rank-1 fast-weight members (Wen et al. 2020).

    Each member applies the shared kernel ``W`` with per-member rank-1 fast
    weights ``r_m`` (``alpha``) and ``s_m`` (``gamma``):
    ``y_m = ((x ∘ r_m) W) ∘ s_m`` (arXiv:2002.06715, eq. 1), cross-checked
    against ``../torch-uncertainty/src/torch_uncertainty/layers/batch_ensemble.py``
    (``BatchLinear.forward``). The predictive mean/variance aggregate over
    the member axis exactly like a deep ensemble.
    """

    def __init__(self, state: BatchEnsembleState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        # Vectorise the rank-1 forward over the member axis (alpha/gamma rows).
        def _member_forward(alpha_m: jax.Array, gamma_m: jax.Array) -> jax.Array:
            return ((x * alpha_m) @ self._state.shared_kernel) * gamma_m

        member_outputs = jax.vmap(_member_forward)(self._state.alpha, self._state.gamma)
        return _predictive_from_member_samples(
            samples=member_outputs,
            capability=self._capability,
            extra_metadata=(("num_members", int(self._state.alpha.shape[0])),),
        )


class _WrappedTestTimeAugmentationModel:
    """Predict by forwarding each input augmentation + cross-augmentation mean/variance.

    Forwards the deterministic ``model_fn`` over every augmented copy of the
    input and aggregates the predictive mean / variance across the
    augmentation axis — the same member-aggregation as a deep ensemble
    (torch-uncertainty
    ``routines/classification.py`` lines 439/446: ``rearrange(logits,
    "(m b) c -> b m c")`` then ``probs_per_est.mean(dim=1)``).

    This is the **regression** mean+across-augmentation-variance form,
    consistent with the other model adapters in this module; ``aleatoric``
    is identically zero (the deterministic model contributes no
    observation-noise term) so ``total_uncertainty == epistemic``. The
    **classification** analogue is the predictive-entropy / mutual-information
    decomposition ``MI = H(mean_m p_m) − mean_m H(p_m)`` (torch-uncertainty
    ``metrics/classification/mutual_information.py`` lines 89-93); no
    classification path is fabricated here.
    """

    def __init__(self, state: TestTimeAugmentationState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        preds = jnp.stack(
            [self._state.model_fn(aug(x)) for aug in self._state.augmentations], axis=0
        )
        return _predictive_from_member_samples(
            samples=preds,
            capability=self._capability,
            extra_metadata=(("num_augmentations", len(self._state.augmentations)),),
            include_zero_aleatoric=True,
        )


# ---------------------------------------------------------------------------
# Ensemble adapters (working) + spec stubs for deferred backends
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
class SnapshotEnsembleAdapter:
    """Snapshot ensemble (cyclic-LR snapshots of a single training run).

    Huang, Li, Pleiss, Liu, Hopcroft, Weinberger, "Snapshot Ensembles:
    Train 1, Get M for Free", ICLR 2017 (arXiv:1704.00109). At inference
    time, average the snapshot forward passes — identical aggregation to
    :class:`DeepEnsembleAdapter`.
    """

    def wrap(
        self, model: SnapshotEnsembleState, capability: UQCapability
    ) -> _WrappedSnapshotEnsembleModel:
        """Wrap a :class:`SnapshotEnsembleState`; rejects non-``SNAPSHOT_ENSEMBLE``."""
        if capability.default_strategy is not DefaultStrategy.SNAPSHOT_ENSEMBLE:
            raise ValueError(
                f"SnapshotEnsembleAdapter requires default_strategy="
                f"{DefaultStrategy.SNAPSHOT_ENSEMBLE!r}; got "
                f"{capability.default_strategy!r}."
            )
        model.validate()
        return _WrappedSnapshotEnsembleModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SWAGAdapter:
    """SWAG: Stochastic Weight Averaging Gaussian posterior over weights.

    Maddox, Garipov, Izmailov, Vetrov, Wilson, "A Simple Baseline for
    Bayesian Uncertainty in Deep Learning", NeurIPS 2019
    (arXiv:1902.02476). Sample weights from the low-rank-plus-diagonal
    Gaussian, forward each draw, and aggregate the predictive
    mean/variance. RNG is caller-owned at the predict-time boundary.
    """

    def wrap(self, model: SWAGState, capability: UQCapability) -> _WrappedSWAGModel:
        """Wrap a :class:`SWAGState`; rejects non-``SWAG`` capabilities."""
        if capability.default_strategy is not DefaultStrategy.SWAG:
            raise ValueError(
                f"SWAGAdapter requires default_strategy="
                f"{DefaultStrategy.SWAG!r}; got {capability.default_strategy!r}."
            )
        model.validate()
        return _WrappedSWAGModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BatchEnsembleAdapter:
    """BatchEnsemble: rank-1 per-member fast weights on a shared kernel.

    Wen, Tran, Ba, "BatchEnsemble: An Alternative Approach to Efficient
    Ensemble and Lifelong Learning", ICLR 2020 (arXiv:2002.06715). Forward
    over the ``M`` rank-1 members ``y_m = ((x ∘ r_m) W) ∘ s_m`` and
    aggregate the predictive mean/variance over the member axis.
    """

    def wrap(
        self, model: BatchEnsembleState, capability: UQCapability
    ) -> _WrappedBatchEnsembleModel:
        """Wrap a :class:`BatchEnsembleState`; rejects non-``BATCH_ENSEMBLE``."""
        if capability.default_strategy is not DefaultStrategy.BATCH_ENSEMBLE:
            raise ValueError(
                f"BatchEnsembleAdapter requires default_strategy="
                f"{DefaultStrategy.BATCH_ENSEMBLE!r}; got "
                f"{capability.default_strategy!r}."
            )
        model.validate()
        return _WrappedBatchEnsembleModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TestTimeAugmentationAdapter:
    """Test-time augmentation: average predictions across input augmentations.

    Wang, Aitchison, Rutherford, … (Neurocomputing 2019). Forward the
    deterministic ``model_fn`` over a fixed tuple of deterministic input
    augmentations and aggregate the predictive mean/variance across the
    augmentation axis — identical aggregation to :class:`DeepEnsembleAdapter`.
    No RNG: the wrapper is jit/grad/vmap-friendly.
    """

    def wrap(
        self, model: TestTimeAugmentationState, capability: UQCapability
    ) -> _WrappedTestTimeAugmentationModel:
        """Wrap a :class:`TestTimeAugmentationState`; rejects non-``TEST_TIME_AUGMENTATION``."""
        if capability.default_strategy is not DefaultStrategy.TEST_TIME_AUGMENTATION:
            raise ValueError(
                f"TestTimeAugmentationAdapter requires default_strategy="
                f"{DefaultStrategy.TEST_TIME_AUGMENTATION!r}; got "
                f"{capability.default_strategy!r}."
            )
        model.validate()
        return _WrappedTestTimeAugmentationModel(state=model, capability=capability)


__all__ = [
    "BatchEnsembleAdapter",
    "BatchEnsembleState",
    "DeepEnsembleAdapter",
    "DeepEnsembleState",
    "SWAGAdapter",
    "SWAGState",
    "SnapshotEnsembleAdapter",
    "SnapshotEnsembleState",
    "TestTimeAugmentationAdapter",
    "TestTimeAugmentationState",
]
