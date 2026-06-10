"""Neural-operator UQ adapter declarations.

Operator-mediated UQ adapter specs for the FNO and DeepONet families.
These specs declare which UQ strategy each family is compatible with
(conformal, deep ensemble, MC-dropout), the operator's function-space
axis metadata (spatial / spectral), and the supported metric set
(L2 / H1 / spatial-coverage / spectral-coverage) — without claiming
native Bayesian support.

``wrap()`` delegates by strategy to the existing concrete adapters and
enriches the resulting :class:`PredictiveDistribution.metadata` with the
operator's function-space provenance (family, spatial / spectral axes,
supported metrics) so downstream consumers can identify the output
topology:

* **ENSEMBLE** — the ``model`` argument is the tuple of operator ensemble
  members; the spec packages it as a
  :class:`~opifex.uncertainty.adapters.ensemble.DeepEnsembleState` and
  delegates to
  :class:`~opifex.uncertainty.adapters.ensemble.DeepEnsembleAdapter`.
* **MC_DROPOUT** — the ``model`` argument is an
  :class:`~opifex.uncertainty.adapters.model.MCDropoutState`; the spec
  delegates to
  :class:`~opifex.uncertainty.adapters.model.MCDropoutAdapter` and
  preserves the caller-owned ``rngs`` at predict-time.
* **CONFORMAL** — conformal UQ for operators does **not** fit the
  ``wrap(model, capability) -> predict_distribution`` adapter contract:
  the conformal calibrators consume a held-out calibration set
  (``predictions`` / ``targets``) at fit time and emit a
  :class:`~opifex.uncertainty.types.PredictionInterval` (a scalar
  threshold), not a per-input predictive distribution. ``wrap()``
  therefore raises an actionable :class:`NotImplementedError` redirecting
  callers to the dedicated conformal surfaces —
  :class:`~opifex.uncertainty.conformal.fields.FieldSplitConformalRegressor`
  for a deterministic FNO / DeepONet field output, or
  :class:`opifex.neural.operators.specialized.uqno.UncertaintyQuantificationNeuralOperator`
  for the native UQNO conformal operator. This is an honest, documented
  boundary, not a generic stub.

Each spec is:

* a pattern-(A) frozen dataclass (``@dataclass(frozen=True,
  slots=True, kw_only=True)``);
* hashable (used as a static argument in jit boundaries);
* honest — ``recommended_capability()`` returns a
  :class:`UQCapability` with ``native_bayesian=False`` and the
  matching strategy + capability flag; ``wrap()`` rejects any capability
  that falsely claims ``native_bayesian=True``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opifex.uncertainty.adapters.ensemble import DeepEnsembleAdapter, DeepEnsembleState
from opifex.uncertainty.adapters.model import MCDropoutAdapter
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import (  # noqa: TC001 — eager per convention
    MetadataItems,
    PredictiveDistribution,
)


_FNO_SPATIAL_AXES = (1, 2)
_FNO_SPECTRAL_AXES = (1, 2)
_DEEPONET_SPATIAL_AXES = (1,)

_CONFORMAL_METRICS = ("l2", "h1", "spatial_coverage", "spectral_coverage")
_ENSEMBLE_METRICS = ("l2", "h1", "spatial_coverage")
_MCDROPOUT_METRICS = ("l2", "spatial_coverage")

# Operator-family identifier strings used as ``operator_family`` values.
_FNO_FAMILY = "fno"
_DEEPONET_FAMILY = "deeponet"

# Required-capability tag tuple every operator adapter spec demands.
_OPERATOR_REQUIRED_CAPABILITIES: tuple[str, ...] = (
    "native_nnx_module",
    "supports_function_space",
)


def _enrich_with_function_space_metadata(
    distribution: PredictiveDistribution, function_space_metadata: MetadataItems
) -> PredictiveDistribution:
    """Return ``distribution`` with operator function-space provenance appended.

    The concrete adapter (deep-ensemble / MC-dropout) owns the ``method`` /
    ``source_package`` provenance plus the mean / epistemic arrays; this
    helper leaves all of those untouched and only appends the operator
    function-space keys that are not already present (idempotent on repeat
    application). Single source of truth for the metadata enrichment shared
    by every operator strategy wrapper (Rule 1: DRY).
    """
    existing = distribution.metadata
    existing_keys = {key for key, _ in existing}
    appended = tuple(
        (key, value) for key, value in function_space_metadata if key not in existing_keys
    )
    if not appended:
        return distribution
    return dataclasses.replace(distribution, metadata=existing + appended)


class _OperatorFunctionSpaceWrapper:
    """Delegate ``predict_distribution`` and append function-space provenance.

    Wraps the object returned by a concrete adapter
    (:class:`DeepEnsembleAdapter` / :class:`MCDropoutAdapter`). The mean and
    epistemic / total-uncertainty terms come from the concrete adapter
    unchanged; only :attr:`PredictiveDistribution.metadata` is enriched with
    the operator family / spatial-axis / spectral-axis / supported-metric
    provenance. ``predict_distribution`` forwards every positional and
    keyword argument (so caller-owned ``rngs`` for the MC-dropout path is
    preserved) and is transform-safe — :func:`dataclasses.replace` over a
    ``flax.struct`` pytree composes with ``jit`` / ``grad`` / ``vmap``.
    """

    def __init__(self, *, wrapped: Any, function_space_metadata: MetadataItems) -> None:
        """Store the wrapped operator and its function-space metadata."""
        self._wrapped = wrapped
        self._function_space_metadata = function_space_metadata

    def predict_distribution(self, *args: Any, **kwargs: Any) -> PredictiveDistribution:
        """Forward to the delegate and append function-space provenance metadata."""
        distribution = self._wrapped.predict_distribution(*args, **kwargs)
        return _enrich_with_function_space_metadata(distribution, self._function_space_metadata)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OperatorAdapterSpec:
    """Base class for operator-family UQ adapter specs.

    Fields:

    * ``operator_family`` — lowercase family name (``"fno"``, ``"deeponet"``).
    * ``default_strategy`` — :class:`DefaultStrategy` enum value advertising
      which adapter the spec configures (CONFORMAL, ENSEMBLE, MC_DROPOUT).
    * ``source_package`` — owning package name (always ``"opifex"`` here).
    * ``spatial_axes`` — tuple of input/output axes treated as spatial by
      the operator (used by function-space metrics and the calibrator).
    * ``spectral_axes`` — subset of ``spatial_axes`` that participate in a
      Fourier-spectral kernel; ``None`` for non-spectral operators (e.g.
      DeepONet).
    * ``supported_metrics`` — tuple of metric names the eventual calibrator
      can compute against this spec (``"l2"``, ``"h1"``,
      ``"spatial_coverage"``, ``"spectral_coverage"``).
    * ``required_capabilities`` — capability tags the operator must satisfy
      before the spec is wired (e.g. ``("native_nnx_module",)``).
    """

    operator_family: str
    default_strategy: DefaultStrategy
    source_package: str = "opifex"
    spatial_axes: tuple[int, ...] = ()
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate that ``spectral_axes`` is a subset of ``spatial_axes``."""
        if self.spectral_axes is not None and not set(self.spectral_axes).issubset(
            set(self.spatial_axes)
        ):
            raise ValueError(
                f"spectral_axes must be a subset of spatial_axes; got "
                f"spectral_axes={self.spectral_axes!r}, spatial_axes={self.spatial_axes!r}"
            )

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    def recommended_capability(self) -> UQCapability:
        """Return an honest :class:`UQCapability` for this spec.

        ``native_bayesian`` is always ``False`` — adapter-mediated UQ on
        a deterministic operator is not native Bayesian. The matching
        strategy capability flag (``supports_conformal`` /
        ``supports_ensemble``) is set to ``True``;
        ``supports_function_space`` is always ``True`` for operator
        adapters.
        """
        flag_map: dict[DefaultStrategy, str] = {
            DefaultStrategy.CONFORMAL: "supports_conformal",
            DefaultStrategy.ENSEMBLE: "supports_ensemble",
            DefaultStrategy.MC_DROPOUT: "supports_ensemble",
        }
        flag_name = flag_map.get(self.default_strategy)
        if flag_name is None:
            raise ValueError(
                f"OperatorAdapterSpec.default_strategy={self.default_strategy!r} "
                f"is not a recognised adapter strategy "
                f"(supported: CONFORMAL / ENSEMBLE / MC_DROPOUT)."
            )
        kwargs: dict[str, Any] = {
            "default_strategy": self.default_strategy,
            "source_package": self.source_package,
            "supports_function_space": True,
            flag_name: True,
        }
        return UQCapability(**kwargs)

    def function_space_metadata(self) -> MetadataItems:
        """Return the operator function-space provenance as metadata pairs.

        Records ``operator_family``, ``spatial_axes``, the supported metric
        tuple, and — for spectral operators only — ``spectral_axes``. This
        provenance is merged into every wrapped predictive distribution so
        downstream consumers can identify the output topology (and which
        function-space metrics are admissible) without re-deriving it from
        the spec.
        """
        items: list[tuple[str, Any]] = [
            ("operator_family", self.operator_family),
            ("spatial_axes", self.spatial_axes),
            ("supported_metrics", self.supported_metrics),
        ]
        if self.spectral_axes is not None:
            items.append(("spectral_axes", self.spectral_axes))
        return tuple(items)

    # ------------------------------------------------------------------
    # Adapter-wiring boundary — delegate by strategy
    # ------------------------------------------------------------------

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Wire the spec to its concrete adapter, dispatching on ``default_strategy``.

        ENSEMBLE packages ``model`` (the operator-member tuple) as a
        :class:`DeepEnsembleState` and delegates to
        :class:`DeepEnsembleAdapter`; MC_DROPOUT delegates an
        :class:`MCDropoutState` to :class:`MCDropoutAdapter`. The wrapped
        object's ``predict_distribution`` output is enriched with this
        spec's :meth:`function_space_metadata`. CONFORMAL raises an
        actionable redirect to the dedicated conformal calibrators (the
        conformal contract takes calibration data, not a model). A
        capability falsely claiming ``native_bayesian=True`` is rejected.
        """
        if capability.native_bayesian:
            raise ValueError(
                f"{type(self).__name__} declines capabilities with "
                f"native_bayesian=True. Adapter-mediated UQ on a "
                f"{self.operator_family!r} operator is not native Bayesian — "
                f"use the UQNO conformal path or wrap a real "
                f"BayesianFNO / BayesianDeepONet implementation directly."
            )
        if self.default_strategy is DefaultStrategy.ENSEMBLE:
            state = DeepEnsembleState(members=model)
            wrapped = DeepEnsembleAdapter().wrap(state, capability)
            return _OperatorFunctionSpaceWrapper(
                wrapped=wrapped, function_space_metadata=self.function_space_metadata()
            )
        if self.default_strategy is DefaultStrategy.MC_DROPOUT:
            wrapped = MCDropoutAdapter().wrap(model, capability)
            return _OperatorFunctionSpaceWrapper(
                wrapped=wrapped, function_space_metadata=self.function_space_metadata()
            )
        if self.default_strategy is DefaultStrategy.CONFORMAL:
            raise NotImplementedError(
                f"Conformal UQ for the {self.operator_family!r} operator family "
                f"(strategy={self.default_strategy.value!r}) is not exposed through "
                f"the wrap(model, capability) adapter contract: conformal "
                f"calibration consumes a held-out calibration set "
                f"(predictions/targets) and yields a PredictionInterval, not a "
                f"per-input predictive distribution. Use "
                f"opifex.uncertainty.conformal.fields.FieldSplitConformalRegressor "
                f"to conformalize a deterministic {self.operator_family!r} field "
                f"output (supported metrics: {self.supported_metrics!r}), or "
                f"opifex.neural.operators.specialized.uqno."
                f"UncertaintyQuantificationNeuralOperator for the native UQNO "
                f"conformal operator."
            )
        raise ValueError(
            f"{type(self).__name__} has unrecognised default_strategy="
            f"{self.default_strategy!r}; expected CONFORMAL / ENSEMBLE / MC_DROPOUT."
        )


# ---------------------------------------------------------------------------
# FNO family adapter specs (spectral)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNOConformalAdapterSpec(OperatorAdapterSpec):
    """FNO + conformal-calibration adapter spec (pre-UQNO-rewrite path)."""

    operator_family: str = _FNO_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.CONFORMAL
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _CONFORMAL_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNODeepEnsembleAdapterSpec(OperatorAdapterSpec):
    """FNO + deep-ensemble adapter spec (member tuple via DeepEnsembleState)."""

    operator_family: str = _FNO_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.ENSEMBLE
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _ENSEMBLE_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNOMCDropoutAdapterSpec(OperatorAdapterSpec):
    """FNO + MC-dropout adapter spec (caller-owned rngs at predict-time)."""

    operator_family: str = _FNO_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.MC_DROPOUT
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _MCDROPOUT_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


# ---------------------------------------------------------------------------
# DeepONet family adapter specs (no spectral kernel)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetConformalAdapterSpec(OperatorAdapterSpec):
    """DeepONet + conformal-calibration adapter spec."""

    operator_family: str = _DEEPONET_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.CONFORMAL
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _CONFORMAL_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetDeepEnsembleAdapterSpec(OperatorAdapterSpec):
    """DeepONet + deep-ensemble adapter spec."""

    operator_family: str = _DEEPONET_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.ENSEMBLE
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _ENSEMBLE_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetMCDropoutAdapterSpec(OperatorAdapterSpec):
    """DeepONet + MC-dropout adapter spec (caller-owned rngs at predict-time)."""

    operator_family: str = _DEEPONET_FAMILY
    default_strategy: DefaultStrategy = DefaultStrategy.MC_DROPOUT
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _MCDROPOUT_METRICS
    required_capabilities: tuple[str, ...] = _OPERATOR_REQUIRED_CAPABILITIES


__all__ = [
    "DeepONetConformalAdapterSpec",
    "DeepONetDeepEnsembleAdapterSpec",
    "DeepONetMCDropoutAdapterSpec",
    "FNOConformalAdapterSpec",
    "FNODeepEnsembleAdapterSpec",
    "FNOMCDropoutAdapterSpec",
    "OperatorAdapterSpec",
]
