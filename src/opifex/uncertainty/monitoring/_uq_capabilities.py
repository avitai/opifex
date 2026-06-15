"""UQ capability declarations for the monitoring / reporting surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.uncertainty.monitoring.__init__``.

The Phase 5 monitoring layer is a builder + container surface: it
assembles already-computed metrics into a validated
:class:`UQReliabilityReport`. It does not compute the metrics itself —
those come from the calibration / conformal / scoring adapters. Declared
as ``CALIBRATION`` (the dominant downstream consumer) with
``native_jax_kernel=True`` since the builder is pure-function and the
inputs are ``jax.Array``.

Plan reference: ``07-phase-registry-docs-examples.md`` lines 624-627 +
659-664 (monitoring/reporting registry coverage).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_BUILD_RELIABILITY_REPORT_CAPABILITY = UQCapability(
    supports_calibration=True,
    default_strategy=DefaultStrategy.CALIBRATION,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "build_reliability_report assembles already-computed metrics "
        "(ECE / NLL / Brier / coverage / interval width / interval "
        "score / CRPS / energy score / spread-skill / OOD-AUROC / AURC) "
        "into a validated UQReliabilityReport. Pure-function builder "
        "over jax.Array inputs; no uncertainty computation of its own."
    ),
)


_MONITORING_INPUTS_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.CALIBRATION,
    source_package="opifex",
    notes=(
        "MonitoringInputs is the Pattern A frozen provenance container "
        "(model name / method / source package / dataset / split / "
        "run id / assumption warnings) threaded into "
        "build_reliability_report. No computation; metadata only."
    ),
)


MONITORING_CAPABILITIES: dict[str, UQCapability] = {
    "monitoring:build_reliability_report": _BUILD_RELIABILITY_REPORT_CAPABILITY,
    "monitoring:MonitoringInputs": _MONITORING_INPUTS_CAPABILITY,
}


__all__ = ["MONITORING_CAPABILITIES"]
