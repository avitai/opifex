"""UQ capability declarations for the monitoring and reporting surfaces.

Static, module-level constants, collected into :data:`opifex.capabilities.UQ_CAPABILITIES`;
importing this module changes no global state. The monitoring layer assembles
already-computed metrics into a validated :class:`UQReliabilityReport` and computes none of
them itself; they come from the calibration, conformal and scoring adapters. It is declared
``CALIBRATION``, its dominant downstream consumer, with ``native_jax_kernel=True`` since the
builder is a pure function over ``jax.Array`` inputs.
"""

from __future__ import annotations

from opifex.core.capabilities import DefaultStrategy, UQCapability


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
