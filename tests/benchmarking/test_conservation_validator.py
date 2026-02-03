"""Tests for ConservationValidator."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from opifex.benchmarking.validators.conservation import (
    ConservationReport,
    ConservationValidator,
)


@pytest.fixture
def validator() -> ConservationValidator:
    """Create a default ConservationValidator."""
    return ConservationValidator()


class TestConservationValidator:
    """Tests for ConservationValidator.validate()."""

    def test_perfect_conservation(self, validator: ConservationValidator) -> None:
        """No violations when prediction matches truth exactly."""
        y_pred = jnp.array([[1.0, 2.0, 3.0]])
        y_true = jnp.array([[1.0, 2.0, 3.0]])
        report = validator.validate(y_pred, y_true)
        assert report.all_conserved is True

    def test_energy_violation_detected(self, validator: ConservationValidator) -> None:
        """Energy violation detected when prediction differs significantly."""
        y_pred = jnp.array([[10.0, 20.0, 30.0]])
        y_true = jnp.array([[1.0, 2.0, 3.0]])
        report = validator.validate(y_pred, y_true)
        assert "energy" in report.violations
        assert report.violations["energy"] > 0

    def test_momentum_violation_detected(
        self, validator: ConservationValidator
    ) -> None:
        """Momentum violation detected when totals differ."""
        y_pred = jnp.array([[5.0, 0.0, 0.0]])
        y_true = jnp.array([[0.0, 0.0, 5.0]])
        report = validator.validate(y_pred, y_true)
        assert "momentum" in report.violations

    def test_mass_with_target(self) -> None:
        """Mass conservation checks against a target value."""
        validator = ConservationValidator(
            laws=["mass"], mass_target=1.0, mass_tolerance=1e-4
        )
        y_pred = jnp.array([[0.5, 0.5]])
        report = validator.validate(y_pred, y_pred)
        assert "mass" in report.violations
        # Total mass = 1.0 which matches target
        assert report.violations["mass"] == pytest.approx(0.0)

    def test_worst_violation_tracked(self, validator: ConservationValidator) -> None:
        """Report tracks the worst violation across all laws."""
        y_pred = jnp.array([[10.0, 20.0, 30.0]])
        y_true = jnp.array([[1.0, 2.0, 3.0]])
        report = validator.validate(y_pred, y_true)
        assert report.worst_violation >= 0.0

    def test_report_is_frozen_dataclass(self, validator: ConservationValidator) -> None:
        """ConservationReport is a frozen dataclass."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.0, 2.0]])
        report = validator.validate(y_pred, y_true)
        assert isinstance(report, ConservationReport)
        with pytest.raises(AttributeError):
            report.all_conserved = False  # type: ignore[misc]

    def test_selective_laws(self) -> None:
        """Only check specified conservation laws."""
        validator = ConservationValidator(laws=["energy"])
        y_pred = jnp.array([[5.0, 0.0]])
        y_true = jnp.array([[0.0, 5.0]])
        report = validator.validate(y_pred, y_true)
        assert "energy" in report.violations
        assert "momentum" not in report.violations

    def test_custom_tolerances(self) -> None:
        """Custom tolerances affect violation detection."""
        # Very loose tolerance — should not detect violations
        validator = ConservationValidator(laws=["energy"], energy_tolerance=1e6)
        y_pred = jnp.array([[10.0, 20.0]])
        y_true = jnp.array([[1.0, 2.0]])
        report = validator.validate(y_pred, y_true)
        assert report.violations["energy"] == pytest.approx(0.0)


class TestConservationReportInterop:
    """Tests for ConservationReport.to_validation_report()."""

    def test_to_validation_report_passing(self) -> None:
        """A passing report converts to a passing ValidationReport."""
        report = ConservationReport(
            violations={"energy": 0.0, "momentum": 0.0},
            all_conserved=True,
            worst_violation=0.0,
        )
        vr = report.to_validation_report()
        assert vr.passed is True
        assert vr.name == "conservation_check"
        assert vr.reference == "physics_laws"
        assert vr.accuracy_metrics == {"energy": 0.0, "momentum": 0.0}
        assert vr.violations == ()

    def test_to_validation_report_failing(self) -> None:
        """A failing report converts with violation descriptions."""
        report = ConservationReport(
            violations={"energy": 1.5e-3, "momentum": 0.0},
            all_conserved=False,
            worst_violation=1.5e-3,
        )
        vr = report.to_validation_report()
        assert vr.passed is False
        assert len(vr.violations) == 1
        assert "energy" in vr.violations[0]
        assert "worst_violation" in vr.notes

    def test_to_validation_report_roundtrip_data(self) -> None:
        """Violation magnitudes survive the conversion."""
        report = ConservationReport(
            violations={"energy": 2.0e-4, "momentum": 3.0e-5},
            all_conserved=False,
            worst_violation=2.0e-4,
        )
        vr = report.to_validation_report()
        assert vr.accuracy_metrics["energy"] == pytest.approx(2.0e-4)
        assert vr.accuracy_metrics["momentum"] == pytest.approx(3.0e-5)


class TestConservationConvergence:
    """Tests for ConservationValidator.validate_convergence()."""

    def test_convergence_with_improving_series(self) -> None:
        """Convergence detected when violations decrease across resolutions."""
        validator = ConservationValidator(laws=["energy"])
        # Simulate multi-resolution: violations decrease
        predictions = [
            jnp.array([[5.0, 5.0]]),
            jnp.array([[2.0, 3.0]]),
            jnp.array([[1.1, 2.1]]),
        ]
        truths = [
            jnp.array([[1.0, 2.0]]),
            jnp.array([[1.0, 2.0]]),
            jnp.array([[1.0, 2.0]]),
        ]
        result = validator.validate_convergence(
            predictions, truths, tolerances=[1.0, 0.1]
        )
        assert result.rates is not None
        assert len(result.rates) > 0
