"""Tests for benchmarking shared constants and utilities."""

import pytest
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric

from opifex.benchmarking._shared import (
    ACCURACY_METRIC_KEYS,
    CHEMICAL_ACCURACY_THRESHOLDS,
    extract_metric_value,
    infer_domain,
    LOWER_IS_BETTER,
)


class TestConstants:
    """Tests for shared constants consistency."""

    def test_lower_is_better_contains_standard_error_metrics(self):
        """Standard error metrics are marked as lower-is-better."""
        assert "mse" in LOWER_IS_BETTER
        assert "mae" in LOWER_IS_BETTER
        assert "rmse" in LOWER_IS_BETTER
        assert "relative_error" in LOWER_IS_BETTER

    def test_accuracy_keys_are_tuple(self):
        """Accuracy metric keys are an ordered tuple."""
        assert isinstance(ACCURACY_METRIC_KEYS, tuple)
        assert "mse" in ACCURACY_METRIC_KEYS
        assert "r2_score" in ACCURACY_METRIC_KEYS

    def test_chemical_thresholds_have_expected_domains(self):
        """Chemical thresholds cover quantum, materials, and molecular."""
        assert "quantum_computing" in CHEMICAL_ACCURACY_THRESHOLDS
        assert "materials_science" in CHEMICAL_ACCURACY_THRESHOLDS
        assert "molecular_dynamics" in CHEMICAL_ACCURACY_THRESHOLDS


class TestInferDomain:
    """Tests for domain inference from dataset names."""

    def test_fluid_dynamics_keywords(self):
        """Fluid-related keywords map to fluid_dynamics."""
        assert infer_domain("PDEBench_2D_DarcyFlow") == "fluid_dynamics"
        assert infer_domain("burgers_equation") == "fluid_dynamics"
        assert infer_domain("navier_stokes_2d") == "fluid_dynamics"

    def test_quantum_keywords(self):
        """Quantum-related keywords map to quantum_computing."""
        assert infer_domain("molecular_dynamics_h2o") == "quantum_computing"
        assert infer_domain("dft_silicon") == "quantum_computing"

    def test_unknown_defaults_to_general(self):
        """Unknown dataset names default to 'general'."""
        assert infer_domain("random_dataset_xyz") == "general"

    def test_case_insensitive(self):
        """Domain inference is case-insensitive."""
        assert infer_domain("BURGERS_Equation") == "fluid_dynamics"


class TestExtractMetricValue:
    """Tests for metric value extraction from BenchmarkResult."""

    def test_extracts_existing_metric(self):
        """Extracts value from an existing metric."""
        result = BenchmarkResult(
            name="test",
            metrics={"mse": Metric(value=0.001)},
        )
        assert extract_metric_value(result, "mse") == pytest.approx(0.001)

    def test_returns_default_for_missing_metric(self):
        """Returns default when metric doesn't exist."""
        result = BenchmarkResult(name="test", metrics={})
        assert extract_metric_value(result, "mse") == float("inf")

    def test_custom_default(self):
        """Respects custom default value."""
        result = BenchmarkResult(name="test", metrics={})
        assert extract_metric_value(result, "mse", default=-1.0) == -1.0
