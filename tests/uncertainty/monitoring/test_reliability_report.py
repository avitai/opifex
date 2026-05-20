"""Monitoring-side reliability report tests.

The monitoring subsystem builds a `UQReliabilityReport` from a typed
`MonitoringInputs` value object — the inputs come from a streaming /
evaluation context (model name, dataset, split, run-id, etc.) and the
already-computed metric values are passed through unchanged.
"""

from __future__ import annotations

import dataclasses as dc

import jax.numpy as jnp
import pytest


def _import_monitoring():
    from opifex.uncertainty import monitoring

    return monitoring


def test_monitoring_inputs_is_frozen_dataclass() -> None:
    monitoring = _import_monitoring()
    inputs = monitoring.MonitoringInputs(
        model_name="fno_darcy",
        method="split_conformal",
        source_package="opifex.uncertainty.conformal",
        dataset="darcy_test",
        split="test",
        run_id="run-0001",
    )
    assert inputs.model_name == "fno_darcy"
    with pytest.raises(dc.FrozenInstanceError):
        inputs.model_name = "other"  # type: ignore[misc]


def test_build_reliability_report_populates_metadata_from_inputs() -> None:
    monitoring = _import_monitoring()
    inputs = monitoring.MonitoringInputs(
        model_name="fno_darcy",
        method="split_conformal",
        source_package="opifex.uncertainty.conformal",
        dataset="darcy_test",
        split="test",
        run_id="run-0001",
    )
    report = monitoring.build_reliability_report(
        inputs=inputs,
        calibration_ece=jnp.asarray(0.05),
        empirical_coverage=jnp.asarray(0.91),
    )
    md = dict(report.metadata)
    assert md["model_name"] == "fno_darcy"
    assert md["method"] == "split_conformal"
    assert md["source_package"] == "opifex.uncertainty.conformal"
    assert md["dataset"] == "darcy_test"
    assert md["split"] == "test"
    assert md["run_id"] == "run-0001"


def test_build_reliability_report_propagates_assumption_warnings() -> None:
    monitoring = _import_monitoring()
    inputs = monitoring.MonitoringInputs(
        model_name="m",
        method="conformal",
        source_package="opifex.uncertainty.conformal",
        dataset="d",
        split="test",
        run_id="r",
        assumption_warnings=("ks_residual_shift_detected:p=0.001",),
    )
    report = monitoring.build_reliability_report(
        inputs=inputs,
        empirical_coverage=jnp.asarray(0.7),
    )
    md = dict(report.metadata)
    assert tuple(md["assumption_warnings"]) == ("ks_residual_shift_detected:p=0.001",)


def test_build_reliability_report_does_not_mutate_inputs() -> None:
    monitoring = _import_monitoring()
    inputs = monitoring.MonitoringInputs(
        model_name="m",
        method="conformal",
        source_package="opifex.uncertainty",
        dataset="d",
        split="test",
        run_id="r",
    )
    original_model = inputs.model_name
    _ = monitoring.build_reliability_report(inputs=inputs, empirical_coverage=jnp.asarray(0.9))
    assert inputs.model_name == original_model


def test_build_reliability_report_validates_payload() -> None:
    """No metric supplied → validation fails (no silent half-empty reports)."""
    monitoring = _import_monitoring()
    inputs = monitoring.MonitoringInputs(
        model_name="m",
        method="conformal",
        source_package="opifex.uncertainty",
        dataset="d",
        split="test",
        run_id="r",
    )
    with pytest.raises(ValueError, match=r"(?i)(metric|empty|populated)"):
        monitoring.build_reliability_report(inputs=inputs)


def test_public_monitoring_surface() -> None:
    monitoring = _import_monitoring()
    expected = {"MonitoringInputs", "build_reliability_report"}
    missing = expected - set(dir(monitoring))
    assert not missing, f"missing public monitoring symbols: {sorted(missing)}"
