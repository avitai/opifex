"""UQ capability declarations for the MLOps surfaces.

``opifex.mlops`` is a metric-publication surface: ``ExperimentTracker`` forwards
UQ-flavoured metrics (Brier, ECE, NLL, coverage) to a run, but computes no
uncertainty itself, so its capability is ``UNSUPPORTED``. Registration is
explicit: importing the package registers nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


if TYPE_CHECKING:
    from opifex.uncertainty.registry import UQRegistry


_EXPERIMENT_TRACKER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "ExperimentTracker publishes UQ-flavoured metrics (Brier / ECE / "
        "NLL / coverage / interval width) to MLflow or other registered "
        "backends. It does not own any uncertainty computation, which "
        "lives in opifex.uncertainty.monitoring and the calibration and "
        "conformal adapters."
    ),
)


MLOPS_CAPABILITIES: dict[str, UQCapability] = {
    "mlops:ExperimentTracker": _EXPERIMENT_TRACKER_CAPABILITY,
}


def register_mlops_capabilities(registry: UQRegistry) -> None:
    """Register the MLOps capabilities in ``registry``; already-registered names are kept."""
    for name, capability in MLOPS_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


__all__ = ["MLOPS_CAPABILITIES", "register_mlops_capabilities"]
