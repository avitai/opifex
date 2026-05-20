"""Adapter protocols + spec containers.

Two protocols and one capability-spec container:

* :class:`DistributionAdapterProtocol` — wraps backend distributions (Artifex
  primary, Distrax secondary, TFP / bijx / FlowJAX / GPJax / NumPyro through
  unsupported metadata).
* :class:`ModelUncertaintyAdapterProtocol` — wraps deterministic / ensemble /
  dropout / Laplace-style models with capability metadata.
* :class:`DistributionAdapterSpec` — capability declaration with pinned
  ``resolution_order`` tuple.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING


if TYPE_CHECKING:
    from opifex.uncertainty.registry import UQCapability
    from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


def compose_method_metadata(
    *, method: str, source_package: str, extra: MetadataItems = ()
) -> MetadataItems:
    """Compose the standard adapter metadata tuple.

    Single source of truth used by every adapter (model.py, ensemble.py,
    …) to attach ``method`` + ``source_package`` provenance to wrapped
    artefacts.
    """
    base: list[tuple[str, Any]] = [
        ("method", method),
        ("source_package", source_package),
    ]
    base.extend(extra)
    return tuple(base)


@runtime_checkable
class DistributionAdapterProtocol(Protocol):
    """Wrap a backend distribution into an Opifex :class:`PredictiveDistribution`.

    Primary adapter target: ``artifex.generative_models.core.distributions.base.Distribution``
    (verified at ``../artifex/src/artifex/generative_models/core/distributions/base.py``).
    Secondary target: Distrax-like objects exposing ``sample``, ``log_prob``,
    ``mean``, ``variance``.
    """

    def from_distribution(self, distribution: Any) -> PredictiveDistribution:
        """Wrap ``distribution`` into a :class:`PredictiveDistribution`."""
        ...


@runtime_checkable
class ModelUncertaintyAdapterProtocol(Protocol):
    """Wrap a model surface with declared UQ capability metadata.

    Concrete implementations cover ``ModelUncertaintyAdapter``,
    ``DeepEnsembleAdapter``, ``MCDropoutAdapter``,
    ``BayesianLastLayerAdapterSpec``, ``LaplaceAdapterSpec``, etc.
    """

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Wrap ``model`` and record adapter strategy + source-package metadata."""
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DistributionAdapterSpec:
    """Capability declaration for a distribution adapter.

    Frozen, slotted, hashable. ``resolution_order`` is a tuple to remain
    hashable; order is binding — the router walks the tuple left-to-right and
    selects the first installed backend.
    """

    name: str
    primary_target: str
    resolution_order: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that ``primary_target`` is contained in ``resolution_order``."""
        if not isinstance(self.resolution_order, tuple):
            raise TypeError("resolution_order must be a tuple.")
        if self.primary_target not in self.resolution_order:
            raise ValueError(
                f"primary_target {self.primary_target!r} must appear in "
                f"resolution_order {self.resolution_order!r}."
            )


__all__ = [
    "DistributionAdapterProtocol",
    "DistributionAdapterSpec",
    "ModelUncertaintyAdapterProtocol",
    "compose_method_metadata",
]
