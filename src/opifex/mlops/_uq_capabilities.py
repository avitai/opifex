"""UQ capability declarations for the MLOps surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.mlops.__init__``.

``opifex.mlops`` is a metric-publication surface: ``ExperimentTracker``
forwards UQ-flavoured metrics (Brier / ECE / NLL / coverage) to MLflow
or other registered backends, but it does not compute uncertainty
itself. Capability is therefore ``UNSUPPORTED`` — the honest reading
per the plan's "UNSUPPORTED placeholder if monitoring is
metric-publication-only" rule.

Plan reference: ``07-phase-registry-docs-examples.md`` lines 624-627.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_EXPERIMENT_TRACKER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "ExperimentTracker publishes UQ-flavoured metrics (Brier / ECE / "
        "NLL / coverage / interval width) to MLflow or other registered "
        "backends. It does not own any uncertainty computation — that "
        "lives in opifex.uncertainty.monitoring + the calibration / "
        "conformal adapters. Declared UNSUPPORTED for the registry per "
        "the plan's 'metric-publication-only' rule."
    ),
)


MLOPS_CAPABILITIES: dict[str, UQCapability] = {
    "mlops:ExperimentTracker": _EXPERIMENT_TRACKER_CAPABILITY,
}


__all__ = ["MLOPS_CAPABILITIES"]
