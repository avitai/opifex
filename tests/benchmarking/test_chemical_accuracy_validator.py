"""Tests for ChemicalAccuracyValidator."""

from __future__ import annotations

import pytest
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from opifex.benchmarking.validators.chemical_accuracy import (
    ChemicalAccuracyAssessment,
    ChemicalAccuracyValidator,
)


@pytest.fixture
def validator() -> ChemicalAccuracyValidator:
    """Create a default ChemicalAccuracyValidator."""
    return ChemicalAccuracyValidator()


class TestChemicalAccuracyValidator:
    """Tests for ChemicalAccuracyValidator."""

    def test_quantum_computing_pass(self, validator: ChemicalAccuracyValidator) -> None:
        """Passes when error is below quantum threshold (1e-3 Hartree)."""
        result = _make_result(relative_error=5e-4)
        assessment = validator.assess(result, "quantum_computing")
        assert assessment.passed is True

    def test_quantum_computing_fail(self, validator: ChemicalAccuracyValidator) -> None:
        """Fails when error exceeds quantum threshold."""
        result = _make_result(relative_error=5e-3)
        assessment = validator.assess(result, "quantum_computing")
        assert assessment.passed is False

    def test_materials_science_pass(self, validator: ChemicalAccuracyValidator) -> None:
        """Passes when error is below materials threshold (5e-2 eV/atom)."""
        result = _make_result(relative_error=1e-2)
        assessment = validator.assess(result, "materials_science")
        assert assessment.passed is True

    def test_materials_science_fail(self, validator: ChemicalAccuracyValidator) -> None:
        """Fails when error exceeds materials threshold."""
        result = _make_result(relative_error=1e-1)
        assessment = validator.assess(result, "materials_science")
        assert assessment.passed is False

    def test_molecular_dynamics_pass(
        self, validator: ChemicalAccuracyValidator
    ) -> None:
        """Passes when error is below molecular dynamics threshold (1e-2 eV)."""
        result = _make_result(relative_error=5e-3)
        assessment = validator.assess(result, "molecular_dynamics")
        assert assessment.passed is True

    def test_molecular_dynamics_fail(
        self, validator: ChemicalAccuracyValidator
    ) -> None:
        """Fails when error exceeds molecular dynamics threshold."""
        result = _make_result(relative_error=5e-2)
        assessment = validator.assess(result, "molecular_dynamics")
        assert assessment.passed is False

    def test_auto_detection_from_tags(self) -> None:
        """Auto-detects domain from BenchmarkResult tags."""
        validator = ChemicalAccuracyValidator()
        result = _make_result(relative_error=5e-4, domain="quantum_computing")
        assessment = validator.assess(result)
        assert assessment.passed is True

    def test_custom_threshold_override(self) -> None:
        """Custom threshold overrides domain default."""
        validator = ChemicalAccuracyValidator(thresholds={"quantum_computing": 1e-1})
        result = _make_result(relative_error=5e-2)
        assessment = validator.assess(result, "quantum_computing")
        assert assessment.passed is True

    def test_margin_positive_when_passing(
        self, validator: ChemicalAccuracyValidator
    ) -> None:
        """Margin is positive when result passes (headroom)."""
        result = _make_result(relative_error=5e-4)
        assessment = validator.assess(result, "quantum_computing")
        assert assessment.margin > 0

    def test_assessment_wraps_accuracy_result(
        self, validator: ChemicalAccuracyValidator
    ) -> None:
        """ChemicalAccuracyAssessment wraps calibrax AccuracyResult."""
        result = _make_result(relative_error=5e-4)
        assessment = validator.assess(result, "quantum_computing")
        assert isinstance(assessment, ChemicalAccuracyAssessment)
        assert assessment.accuracy_result is not None
        assert assessment.domain == "quantum_computing"

    def test_recommendations_on_failure(
        self, validator: ChemicalAccuracyValidator
    ) -> None:
        """Provides recommendations when assessment fails."""
        result = _make_result(relative_error=5e-3)
        assessment = validator.assess(result, "quantum_computing")
        assert assessment.passed is False
        assert len(assessment.recommendations) > 0

    def test_unknown_domain_raises(self, validator: ChemicalAccuracyValidator) -> None:
        """Raises ValueError for unknown domain without auto-detection."""
        result = _make_result(relative_error=0.01)
        with pytest.raises(ValueError, match="Unknown domain"):
            validator.assess(result, "astrophysics")


def _make_result(
    relative_error: float = 0.01,
    domain: str = "",
    name: str = "TestModel",
) -> BenchmarkResult:
    """Create a minimal BenchmarkResult for testing."""
    tags: dict[str, str] = {}
    if domain:
        tags["domain"] = domain

    return BenchmarkResult(
        name=name,
        domain=domain,
        tags=tags,
        metrics={"relative_error": Metric(value=relative_error)},
    )
