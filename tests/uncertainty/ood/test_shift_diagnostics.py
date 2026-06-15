"""Shift diagnostic contract.

Reuses the two-sample Kolmogorov-Smirnov machinery from
:mod:`opifex.uncertainty.conformal.exchangeability` and packages the
output as a typed report. Diagnostics return explicit status +
metadata, never a bare boolean.
"""

from __future__ import annotations

import dataclasses as dc

import jax.numpy as jnp
import numpy as np
import pytest


def _import_diag():
    from opifex.uncertainty.ood import shift_diagnostics

    return shift_diagnostics


def test_shift_report_is_frozen_dataclass() -> None:
    diag = _import_diag()
    report = diag.ShiftReport(p_value=jnp.asarray(0.5), passes=True)
    assert float(report.p_value) == pytest.approx(0.5)
    assert report.passes is True
    with pytest.raises(dc.FrozenInstanceError):
        report.passes = False  # type: ignore[misc]


def test_residual_shift_diagnostic_passes_on_iid_residuals() -> None:
    diag = _import_diag()
    rng = np.random.default_rng(0)
    reference = jnp.asarray(rng.standard_normal(512))
    observed = jnp.asarray(rng.standard_normal(512))
    report = diag.residual_shift_diagnostic(
        reference_residuals=reference, observed_residuals=observed, alpha=0.05
    )
    assert report.passes
    assert float(report.p_value) > 0.05
    md = dict(report.metadata)
    assert md["method"] == "ks_two_sample_residual"
    assert md["alpha"] == pytest.approx(0.05)


def test_residual_shift_diagnostic_fails_on_distribution_shift() -> None:
    diag = _import_diag()
    rng = np.random.default_rng(1)
    reference = jnp.asarray(rng.standard_normal(512))
    observed = jnp.asarray(2.0 + rng.standard_normal(512))  # location shift
    report = diag.residual_shift_diagnostic(
        reference_residuals=reference, observed_residuals=observed, alpha=0.05
    )
    assert not report.passes
    assert float(report.p_value) < 0.05
    md = dict(report.metadata)
    assert md["status"] == "shift_detected"


def test_residual_shift_diagnostic_records_sample_sizes_in_metadata() -> None:
    diag = _import_diag()
    rng = np.random.default_rng(2)
    reference = jnp.asarray(rng.standard_normal(256))
    observed = jnp.asarray(rng.standard_normal(128))
    report = diag.residual_shift_diagnostic(
        reference_residuals=reference, observed_residuals=observed, alpha=0.05
    )
    md = dict(report.metadata)
    assert int(md["reference_size"]) == 256
    assert int(md["observed_size"]) == 128


def test_shift_diagnostic_does_not_claim_conformal_validity_on_failure() -> None:
    """When the diagnostic fails, metadata must flag that distribution-free
    coverage no longer holds — never silently pass."""
    diag = _import_diag()
    rng = np.random.default_rng(3)
    reference = jnp.asarray(rng.standard_normal(256))
    observed = jnp.asarray(3.0 + rng.standard_normal(256))
    report = diag.residual_shift_diagnostic(
        reference_residuals=reference, observed_residuals=observed, alpha=0.05
    )
    md = dict(report.metadata)
    assert md["assumption_status"] == "shift_detected"
