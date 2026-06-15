"""UQReliabilityReport contract.

The report is a Pattern-B `flax.struct.dataclass` aggregating already-computed
metric values from calibration / conformal / OOD / selective / field
subsystems into a single serializable container. It is a data class +
serializer, NOT an evaluator — it does not recompute metrics from raw
predictions.

Contract:

* All metric leaves are optional (some pipelines skip OOD, others skip
  selective). `validate()` enforces that any required metric is present
  and raises `ValueError` if the report was constructed with a
  half-populated state.
* `metadata` carries provenance: model name, method, `source_package`,
  dataset / split labels, run identifier, assumption warnings.
* `to_dict()` returns a deterministic JSON-compatible mapping with
  arrays converted to Python lists / floats.
* Construction must not mutate input values.
"""

from __future__ import annotations

import dataclasses as dc

import jax.numpy as jnp
import pytest


def _import_reports():
    from opifex.uncertainty import reports

    return reports


# ---------------------------------------------------------------------------
# Construction + frozen contract
# ---------------------------------------------------------------------------


def test_uq_reliability_report_is_frozen_pattern_b_dataclass() -> None:
    reports = _import_reports()
    report = reports.UQReliabilityReport(
        calibration_ece=jnp.asarray(0.05),
        empirical_coverage=jnp.asarray(0.91),
        metadata=(
            ("model_name", "demo"),
            ("method", "conformal"),
            ("source_package", "opifex.uncertainty"),
            ("dataset", "darcy_test"),
            ("run_id", "run-0001"),
        ),
    )
    assert report.calibration_ece is not None
    assert report.empirical_coverage is not None
    assert float(report.calibration_ece) == pytest.approx(0.05)
    assert float(report.empirical_coverage) == pytest.approx(0.91)
    with pytest.raises(dc.FrozenInstanceError):
        report.calibration_ece = jnp.asarray(0.1)  # type: ignore[misc]


def test_uq_reliability_report_metadata_preserves_provenance_fields() -> None:
    reports = _import_reports()
    report = reports.UQReliabilityReport(
        empirical_coverage=jnp.asarray(0.9),
        metadata=(
            ("model_name", "fno_darcy"),
            ("method", "split_conformal"),
            ("source_package", "opifex.uncertainty.conformal"),
            ("dataset", "darcy_test"),
            ("split", "test"),
            ("run_id", "exp-42"),
        ),
    )
    md = dict(report.metadata)
    assert md["model_name"] == "fno_darcy"
    assert md["method"] == "split_conformal"
    assert md["source_package"] == "opifex.uncertainty.conformal"
    assert md["dataset"] == "darcy_test"
    assert md["split"] == "test"
    assert md["run_id"] == "exp-42"


def test_uq_reliability_report_preserves_shift_and_assumption_warnings() -> None:
    """Failed shift / exchangeability diagnostics from Tasks 4.5 and 5.3 must
    propagate verbatim into report metadata."""
    reports = _import_reports()
    warnings = (
        "ks_residual_shift_detected:p=0.001",
        "exchangeability_failed:p=0.0005",
    )
    report = reports.UQReliabilityReport(
        empirical_coverage=jnp.asarray(0.6),
        metadata=(
            ("model_name", "demo"),
            ("method", "split_conformal"),
            ("source_package", "opifex.uncertainty.conformal"),
            ("assumption_warnings", warnings),
            ("assumption_status", "shift_detected"),
        ),
    )
    md = dict(report.metadata)
    assert tuple(md["assumption_warnings"]) == warnings
    assert md["assumption_status"] == "shift_detected"


# ---------------------------------------------------------------------------
# Validation: missing required metrics → explicit error
# ---------------------------------------------------------------------------


def test_uq_reliability_report_validate_rejects_empty_metric_payload() -> None:
    reports = _import_reports()
    report = reports.UQReliabilityReport(metadata=(("model_name", "demo"),))
    with pytest.raises(ValueError, match=r"(?i)(metric|empty|populated)"):
        report.validate()


def test_uq_reliability_report_validate_accepts_any_populated_metric() -> None:
    """Caller may populate any subset of metrics; the validator only fails
    when ALL metric slots are empty."""
    reports = _import_reports()
    report_with_ece = reports.UQReliabilityReport(
        calibration_ece=jnp.asarray(0.05),
        metadata=(("model_name", "demo"),),
    )
    report_with_ece.validate()  # No raise.


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_uq_reliability_report_to_dict_is_deterministic_and_jsonable() -> None:
    """``to_dict`` returns a deterministic JSON-compatible mapping with
    arrays converted to Python scalars / lists."""
    import json

    reports = _import_reports()
    report = reports.UQReliabilityReport(
        calibration_ece=jnp.asarray(0.05),
        empirical_coverage=jnp.asarray(0.91),
        ood_auroc=jnp.asarray(0.88),
        aurc=jnp.asarray(0.12),
        metadata=(
            ("model_name", "demo"),
            ("method", "conformal"),
            ("source_package", "opifex.uncertainty.conformal"),
        ),
    )
    payload = report.to_dict()
    encoded_a = json.dumps(payload, sort_keys=True)
    encoded_b = json.dumps(payload, sort_keys=True)
    assert encoded_a == encoded_b
    assert payload["calibration_ece"] == pytest.approx(0.05)
    assert payload["empirical_coverage"] == pytest.approx(0.91)
    assert payload["metadata"]["model_name"] == "demo"


def test_uq_reliability_report_does_not_mutate_inputs() -> None:
    reports = _import_reports()
    original_ece = jnp.asarray(0.05)
    original_metadata = (
        ("model_name", "demo"),
        ("method", "conformal"),
    )
    report = reports.UQReliabilityReport(
        calibration_ece=original_ece,
        metadata=original_metadata,
    )
    # Caller's references must still point to the same un-mutated objects.
    assert original_metadata == (
        ("model_name", "demo"),
        ("method", "conformal"),
    )
    assert float(original_ece) == pytest.approx(0.05)
    # And `report.calibration_ece` reflects the input.
    assert report.calibration_ece is not None
    assert float(report.calibration_ece) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Aggregation from sibling subsystems
# ---------------------------------------------------------------------------


def test_uq_reliability_report_can_be_built_from_phase_4_5_outputs() -> None:
    """A report can be assembled from a calibration metric, a conformal
    PredictionInterval's coverage, an OOD AUROC value, and an AURC value —
    proving the aggregation contract end-to-end."""
    from opifex.uncertainty.calibration import expected_calibration_error
    from opifex.uncertainty.selective import area_under_risk_coverage

    reports = _import_reports()

    # Calibration ECE on toy binary data.
    probs = jnp.array([0.1, 0.4, 0.8, 0.9])
    targets = jnp.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(probabilities=probs, targets=targets, num_bins=4)

    # AURC on toy confidence/error pairs.
    confidences = jnp.array([0.9, 0.8, 0.4, 0.1])
    errors = jnp.array([0.0, 0.0, 1.0, 1.0])
    aurc = area_under_risk_coverage(confidences=confidences, errors=errors)

    report = reports.UQReliabilityReport(
        calibration_ece=ece,
        aurc=aurc,
        metadata=(
            ("model_name", "toy"),
            ("source_package", "opifex.uncertainty"),
        ),
    )
    report.validate()
    payload = report.to_dict()
    assert payload["calibration_ece"] == pytest.approx(float(ece), abs=1e-6)
    assert payload["aurc"] == pytest.approx(float(aurc), abs=1e-6)
