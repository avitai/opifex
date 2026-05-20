"""Shared base for deferred-backend adapter specs (pattern A).

Every "deferred" adapter (Bayesian-last-layer, Laplace, SNGP, VBLL,
Snapshot-ensemble, SWAG, BatchEnsemble, DUE, TTA) ships a frozen
dataclass that:

* declares the :class:`DefaultStrategy` it represents,
* declares the source package / required upstream capabilities, and
* raises a clear :class:`NotImplementedError` from ``wrap`` until the
  real implementation lands.

The shared base eliminates per-class duplication while keeping every
spec a distinct, frozen dataclass type so ``isinstance``-based dispatch
still works at the call site.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _DeferredAdapterSpec:
    """Base for adapter specs awaiting a backend implementation.

    Subclasses override ``default_strategy`` (and may extend
    ``required_capabilities``) to declare which UQ behaviour they would
    deliver once wired. ``wrap`` always raises
    :class:`NotImplementedError` referencing the strategy name and
    source package so the failure message is actionable.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.UNSUPPORTED
    source_package: str = "opifex"
    required_capabilities: tuple[str, ...] = ()

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Raise :class:`NotImplementedError`; deferred until backend lands."""
        del model, capability
        raise NotImplementedError(
            f"Adapter strategy {self.default_strategy.value!r} is not yet "
            f"wired (source_package={self.source_package!r}). Required "
            f"capabilities: {self.required_capabilities!r}. Wire the backend "
            f"before calling .wrap()."
        )


__all__ = ["_DeferredAdapterSpec"]
