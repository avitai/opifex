"""Neural-operator UQ adapter declarations.

Operator-mediated UQ adapter specs for the FNO and DeepONet families.
These specs declare which UQ strategy each family is compatible with
(conformal, deep ensemble, MC-dropout), the operator's function-space
axis metadata (spatial / spectral), and the supported metric set
(L2 / H1 / spatial-coverage / spectral-coverage) — without claiming
native Bayesian support.

UQNO's native conformal path is handled separately via
:class:`opifex.neural.operators.specialized.uqno.UncertaintyQuantificationNeuralOperator`.
The adapter specs in this module cover the *deterministic-FNO* and
*deterministic-DeepONet* families when bolted to one of the
strategy-specific adapters (conformal calibrator, deep ensemble,
MC-dropout) from :mod:`opifex.uncertainty.adapters.model` and
:mod:`opifex.uncertainty.adapters.ensemble`.

Each spec is:

* a pattern-(A) frozen dataclass (``@dataclass(frozen=True,
  slots=True, kw_only=True)``);
* hashable (used as a static argument in jit boundaries);
* honest — ``recommended_capability()`` returns a
  :class:`UQCapability` with ``native_bayesian=False`` and the
  matching strategy + capability flag;
* actionable — ``wrap()`` raises :class:`NotImplementedError` naming
  the operator family + missing strategy backend until a concrete
  adapter implementation lands (the spec carries the metadata; the
  fitted wrapper will reuse :class:`DeepEnsembleAdapter` /
  :class:`MCDropoutAdapter` from the sibling modules).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_FNO_SPATIAL_AXES = (1, 2)
_FNO_SPECTRAL_AXES = (1, 2)
_DEEPONET_SPATIAL_AXES = (1,)

_CONFORMAL_METRICS = ("l2", "h1", "spatial_coverage", "spectral_coverage")
_ENSEMBLE_METRICS = ("l2", "h1", "spatial_coverage")
_MCDROPOUT_METRICS = ("l2", "spatial_coverage")


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

    # ------------------------------------------------------------------
    # Adapter-wiring boundary (unsupported until a concrete adapter lands)
    # ------------------------------------------------------------------

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Raise an actionable error naming the missing operator-adapter wiring."""
        del model
        if capability.native_bayesian:
            raise ValueError(
                f"{type(self).__name__} declines capabilities with "
                f"native_bayesian=True. Adapter-mediated UQ on a "
                f"{self.operator_family!r} operator is not native Bayesian — "
                f"use the UQNO conformal path or wrap a real "
                f"BayesianFNO / BayesianDeepONet implementation directly."
            )
        raise NotImplementedError(
            f"Operator-adapter wiring for family={self.operator_family!r} + "
            f"strategy={self.default_strategy.value!r} is not yet implemented. "
            f"Required capabilities: {self.required_capabilities!r}; supported "
            f"metrics: {self.supported_metrics!r}. The spec carries the metadata "
            f"a future adapter (DeepEnsembleAdapter / MCDropoutAdapter / a "
            f"conformal calibrator) will consume."
        )


# ---------------------------------------------------------------------------
# FNO family adapter specs (spectral)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNOConformalAdapterSpec(OperatorAdapterSpec):
    """FNO + conformal-calibration adapter spec (pre-UQNO-rewrite path)."""

    operator_family: str = "fno"
    default_strategy: DefaultStrategy = DefaultStrategy.CONFORMAL
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _CONFORMAL_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNODeepEnsembleAdapterSpec(OperatorAdapterSpec):
    """FNO + deep-ensemble adapter spec (member tuple via DeepEnsembleState)."""

    operator_family: str = "fno"
    default_strategy: DefaultStrategy = DefaultStrategy.ENSEMBLE
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _ENSEMBLE_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FNOMCDropoutAdapterSpec(OperatorAdapterSpec):
    """FNO + MC-dropout adapter spec (caller-owned rngs at predict-time)."""

    operator_family: str = "fno"
    default_strategy: DefaultStrategy = DefaultStrategy.MC_DROPOUT
    spatial_axes: tuple[int, ...] = _FNO_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = _FNO_SPECTRAL_AXES
    supported_metrics: tuple[str, ...] = _MCDROPOUT_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


# ---------------------------------------------------------------------------
# DeepONet family adapter specs (no spectral kernel)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetConformalAdapterSpec(OperatorAdapterSpec):
    """DeepONet + conformal-calibration adapter spec."""

    operator_family: str = "deeponet"
    default_strategy: DefaultStrategy = DefaultStrategy.CONFORMAL
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _CONFORMAL_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetDeepEnsembleAdapterSpec(OperatorAdapterSpec):
    """DeepONet + deep-ensemble adapter spec."""

    operator_family: str = "deeponet"
    default_strategy: DefaultStrategy = DefaultStrategy.ENSEMBLE
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _ENSEMBLE_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeepONetMCDropoutAdapterSpec(OperatorAdapterSpec):
    """DeepONet + MC-dropout adapter spec (caller-owned rngs at predict-time)."""

    operator_family: str = "deeponet"
    default_strategy: DefaultStrategy = DefaultStrategy.MC_DROPOUT
    spatial_axes: tuple[int, ...] = _DEEPONET_SPATIAL_AXES
    spectral_axes: tuple[int, ...] | None = None
    supported_metrics: tuple[str, ...] = _MCDROPOUT_METRICS
    required_capabilities: tuple[str, ...] = ("native_nnx_module", "supports_function_space")


__all__ = [
    "DeepONetConformalAdapterSpec",
    "DeepONetDeepEnsembleAdapterSpec",
    "DeepONetMCDropoutAdapterSpec",
    "FNOConformalAdapterSpec",
    "FNODeepEnsembleAdapterSpec",
    "FNOMCDropoutAdapterSpec",
    "OperatorAdapterSpec",
]
