"""Tests for benchmarking validation framework functionality.

This module tests the ValidationFramework class, covering:
- Accuracy metrics (MSE, MAE, R² score)
- Tolerance checking for different domains (quantum, fluid dynamics, materials)
- Convergence rate analysis
- Error analysis (skewness, kurtosis)
- ValidationReport and AccuracyResult dataclasses
"""

import jax.numpy as jnp
import numpy as np
import pytest
from calibrax.core.models import Metric
from calibrax.validation import AccuracyResult, ConvergenceResult

from opifex.benchmarking.evaluation_engine import BenchmarkResult
from opifex.benchmarking.validation_framework import (
    ErrorAnalysis,
    ValidationFramework,
    ValidationReport,
)


class TestValidationReportDataclass:
    """Test ValidationReport dataclass."""

    def test_validation_report_creation(self):
        """Test basic ValidationReport creation."""
        report = ValidationReport(
            benchmark_name="TestBenchmark",
            reference_method="FEM",
            accuracy_metrics={"mse": 0.01, "mae": 0.05},
            convergence_metrics={"rate": 1.5},
        )

        assert report.benchmark_name == "TestBenchmark"
        assert report.reference_method == "FEM"
        assert report.accuracy_metrics["mse"] == 0.01
        # Validation should pass with no violations
        assert report.validation_passed is True

    def test_validation_report_fails_with_violations(self):
        """Test that validation fails when there are tolerance violations."""
        report = ValidationReport(
            benchmark_name="TestBenchmark",
            reference_method="FEM",
            accuracy_metrics={"mse": 0.5},
            convergence_metrics={},
            tolerance_violations=["MSE exceeds tolerance"],
        )

        assert report.validation_passed is False

    def test_validation_report_with_chemical_accuracy(self):
        """Test validation with chemical accuracy requirements."""
        # Passes when chemical accuracy is achieved
        report_pass = ValidationReport(
            benchmark_name="QuantumTest",
            reference_method="DFT",
            accuracy_metrics={"mse": 0.001},
            convergence_metrics={},
            chemical_accuracy_status=True,
        )
        assert report_pass.validation_passed is True

        # Fails when chemical accuracy is not achieved
        report_fail = ValidationReport(
            benchmark_name="QuantumTest",
            reference_method="DFT",
            accuracy_metrics={"mse": 0.1},
            convergence_metrics={},
            chemical_accuracy_status=False,
        )
        assert report_fail.validation_passed is False


class TestAccuracyResultDataclass:
    """Test AccuracyResult dataclass from calibrax."""

    def test_accuracy_result_creation(self):
        """Test basic AccuracyResult creation."""
        assessment = AccuracyResult(
            target=0.001,
            achieved=0.0005,
            metric_type="chemical_accuracy",
            units="Hartree",
            passed=True,
            margin=0.0005,
        )

        assert assessment.target == 0.001
        assert assessment.achieved == 0.0005
        assert assessment.passed is True

    def test_accuracy_result_margin_calculation(self):
        """Test that margin is correctly set."""
        assessment = AccuracyResult(
            target=0.01,
            achieved=0.005,
            metric_type="chemical_accuracy",
            units="Hartree",
            passed=True,
            margin=0.005,
        )

        # margin = target - achieved = 0.01 - 0.005 = 0.005
        assert assessment.margin == pytest.approx(0.005, rel=1e-5)


class TestValidationFrameworkInitialization:
    """Test ValidationFramework initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        framework = ValidationFramework()

        assert framework.default_tolerances == [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        assert framework.reference_methods == {}

    def test_custom_tolerances(self):
        """Test initialization with custom tolerances."""
        custom_tolerances = [1e-1, 1e-2, 1e-3]
        framework = ValidationFramework(default_tolerances=custom_tolerances)

        assert framework.default_tolerances == custom_tolerances

    def test_chemical_accuracy_thresholds_defined(self):
        """Test that chemical accuracy thresholds are defined."""
        framework = ValidationFramework()

        assert "quantum_computing" in framework.chemical_accuracy_thresholds
        assert "materials_science" in framework.chemical_accuracy_thresholds
        assert "molecular_dynamics" in framework.chemical_accuracy_thresholds


class TestAccuracyMetrics:
    """Test accuracy metrics computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_compute_mse(self):
        """Test MSE computation with predictions and ground truth."""
        from opifex.benchmarking.validation_framework import _compute_accuracy_metrics

        predictions = jnp.array([1.0, 2.0, 3.0])
        ground_truth = jnp.array([1.1, 2.0, 2.9])

        metrics = _compute_accuracy_metrics(predictions, ground_truth)

        assert "mse" in metrics
        # MSE of [1.0, 2.0, 3.0] vs [1.1, 2.0, 2.9] = mean([0.01, 0.0, 0.01])
        assert metrics["mse"] == pytest.approx(0.00666, abs=1e-4)

    def test_compute_mae(self):
        """Test MAE computation with predictions and ground truth."""
        from opifex.benchmarking.validation_framework import _compute_accuracy_metrics

        predictions = jnp.array([1.0, 2.0, 3.0])
        ground_truth = jnp.array([1.1, 2.0, 2.9])

        metrics = _compute_accuracy_metrics(predictions, ground_truth)

        assert "mae" in metrics
        # MAE of [1.0, 2.0, 3.0] vs [1.1, 2.0, 2.9] = mean([0.1, 0.0, 0.1])
        assert metrics["mae"] == pytest.approx(0.0666, abs=1e-3)

    def test_compute_r2_score(self):
        """Test R² score computation with perfect predictions."""
        from opifex.benchmarking.validation_framework import _compute_accuracy_metrics

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = _compute_accuracy_metrics(predictions, ground_truth)

        assert "r_squared" in metrics
        # Perfect predictions should have R² = 1.0
        assert metrics["r_squared"] == pytest.approx(1.0, rel=1e-5)


class TestToleranceChecking:
    """Test tolerance checking for different domains."""

    def setup_method(self):
        """Set up test fixtures."""
        from opifex.benchmarking.validation_framework import _check_tolerance_violations

        self.framework = ValidationFramework()
        self._check_violations = _check_tolerance_violations

    def test_quantum_computing_tolerance_pass(self):
        """Test quantum computing tolerance - passing case."""
        metrics = {"mse": 1e-5, "relative_error": 1e-4}
        violations = self._check_violations(
            metrics, "QuantumChemistry", self.framework._infer_domain
        )

        assert len(violations) == 0

    def test_quantum_computing_tolerance_fail(self):
        """Test quantum computing tolerance - failing case."""
        metrics = {"mse": 0.01, "relative_error": 0.1}
        violations = self._check_violations(
            metrics, "QuantumChemistry", self.framework._infer_domain
        )

        # Should have violations for both MSE and relative error
        assert len(violations) == 2
        assert any("quantum" in v.lower() for v in violations)

    def test_fluid_dynamics_tolerance_pass(self):
        """Test fluid dynamics tolerance - passing case."""
        metrics = {"mse": 1e-3, "relative_error": 0.05}
        violations = self._check_violations(
            metrics, "NavierStokes2D", self.framework._infer_domain
        )

        assert len(violations) == 0

    def test_fluid_dynamics_tolerance_fail(self):
        """Test fluid dynamics tolerance - failing case."""
        metrics = {"mse": 0.5, "relative_error": 0.5}
        violations = self._check_violations(
            metrics, "NavierStokes2D", self.framework._infer_domain
        )

        assert len(violations) == 2
        assert any("fluid" in v.lower() for v in violations)

    def test_materials_science_tolerance_pass(self):
        """Test materials science tolerance - passing case."""
        metrics = {"mse": 1e-4, "relative_error": 0.01}
        violations = self._check_violations(
            metrics, "CrystalEnergies", self.framework._infer_domain
        )

        assert len(violations) == 0

    def test_materials_science_tolerance_fail(self):
        """Test materials science tolerance - failing case."""
        metrics = {"mse": 0.01, "relative_error": 0.1}
        violations = self._check_violations(
            metrics, "CrystalEnergies", self.framework._infer_domain
        )

        assert len(violations) == 2
        assert any("materials" in v.lower() for v in violations)


class TestDomainInference:
    """Test domain inference from dataset names."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_quantum_computing_domain(self):
        """Test inference of quantum computing domain."""
        assert self.framework._infer_domain("QuantumChemistry") == "quantum_computing"
        assert self.framework._infer_domain("dft_energies") == "quantum_computing"
        assert (
            self.framework._infer_domain("molecular_structures") == "quantum_computing"
        )

    def test_fluid_dynamics_domain(self):
        """Test inference of fluid dynamics domain."""
        assert self.framework._infer_domain("NavierStokes2D") == "fluid_dynamics"
        assert self.framework._infer_domain("Burgers1D") == "fluid_dynamics"
        assert self.framework._infer_domain("DarcyFlow") == "fluid_dynamics"

    def test_materials_science_domain(self):
        """Test inference of materials science domain."""
        assert self.framework._infer_domain("CrystalEnergies") == "materials_science"
        assert self.framework._infer_domain("SolidMechanics") == "materials_science"

    def test_general_domain_fallback(self):
        """Test fallback to general domain."""
        assert self.framework._infer_domain("UnknownDataset") == "general"


class TestValidateAgainstReference:
    """Test validation against reference methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_validate_with_reference_data_and_predictions(self):
        """Test validation with explicit reference data and predictions."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "TestDataset"},
            metrics={"mse": Metric(value=0.01)},
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )
        predictions = jnp.array([1.0, 2.0, 3.0])
        reference_data = jnp.array([1.1, 2.0, 2.9])

        report = self.framework.validate_against_reference(
            result, "FEM", reference_data=reference_data, predictions=predictions
        )

        assert isinstance(report, ValidationReport)
        assert report.benchmark_name == "TestDataset"
        assert report.reference_method == "FEM"
        assert "mse" in report.accuracy_metrics
        # Should compute actual error, not use result metrics
        assert report.accuracy_metrics["mse"] > 0

    def test_validate_with_reference_data_no_predictions(self):
        """Test validation falls back to result metrics without predictions."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "TestDataset"},
            metrics={"mse": Metric(value=0.01)},
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )
        reference_data = jnp.array([1.0, 2.0, 3.0])

        report = self.framework.validate_against_reference(
            result, "FEM", reference_data=reference_data
        )

        # Without predictions, should fall back to result metrics
        assert report.accuracy_metrics["mse"] == 0.01

    def test_validate_without_reference_data(self):
        """Test validation using metrics from results."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "TestDataset"},
            metrics={"mse": Metric(value=0.01), "mae": Metric(value=0.05)},
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )

        report = self.framework.validate_against_reference(
            result, "FEM", reference_data=None
        )

        # Should use metrics from benchmark results (extracted as floats)
        assert report.accuracy_metrics["mse"] == 0.01
        assert report.accuracy_metrics["mae"] == 0.05


class TestConvergenceRateAnalysis:
    """Test convergence rate analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_convergence_analysis_structure(self):
        """Test convergence analysis returns proper structure."""
        results = [
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.1), "mae": Metric(value=0.2)},
                metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
            ),
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.01), "mae": Metric(value=0.05)},
                metadata={"execution_time": 2.0, "framework_version": "flax_nnx"},
            ),
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.001), "mae": Metric(value=0.01)},
                metadata={"execution_time": 3.0, "framework_version": "flax_nnx"},
            ),
        ]

        analysis = self.framework.check_convergence_rates(results)

        assert isinstance(analysis, ConvergenceResult)
        assert "mse" in analysis.rates
        assert "mae" in analysis.rates

    def test_convergence_rate_positive_for_improving_metrics(self):
        """Test that convergence rate is positive for improving (decreasing) metrics."""
        results = [
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=1.0)},
                metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
            ),
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.1)},
                metadata={"execution_time": 2.0, "framework_version": "flax_nnx"},
            ),
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.01)},
                metadata={"execution_time": 3.0, "framework_version": "flax_nnx"},
            ),
        ]

        analysis = self.framework.check_convergence_rates(results)

        # Convergence rate should be positive (log-reduction per step)
        assert analysis.rates["mse"] > 0

    def test_convergence_uses_custom_tolerances(self):
        """Test that custom tolerances are used."""
        results = [
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.1)},
                metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
            ),
            BenchmarkResult(
                name="TestModel",
                tags={"dataset": "TestDataset"},
                metrics={"mse": Metric(value=0.01)},
                metadata={"execution_time": 2.0, "framework_version": "flax_nnx"},
            ),
        ]

        custom_tolerances = [1e-1, 1e-2]
        analysis = self.framework.check_convergence_rates(
            results, tolerances=custom_tolerances
        )

        # achieved dict keys encode metric_tolerance pairs
        assert isinstance(analysis, ConvergenceResult)
        assert "mse" in analysis.rates


class TestChemicalAccuracyAssessment:
    """Test chemical accuracy assessment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_chemical_accuracy_pass(self):
        """Test chemical accuracy assessment - passing case."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "QuantumChemistry"},
            metrics={"mse": Metric(value=1e-4)},  # Below threshold
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )

        assessment = self.framework.assess_chemical_accuracy(result)

        assert isinstance(assessment, AccuracyResult)
        assert assessment.passed is True
        assert assessment.metric_type == "chemical_accuracy"

    def test_chemical_accuracy_fail(self):
        """Test chemical accuracy assessment - failing case."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "QuantumChemistry"},
            metrics={"mse": Metric(value=0.1)},  # Above threshold
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )

        assessment = self.framework.assess_chemical_accuracy(result)

        assert assessment.passed is False

    def test_force_accuracy_assessment(self):
        """Test force accuracy assessment."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "MolecularDynamics"},
            metrics={"mae": Metric(value=1e-3)},
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )

        assessment = self.framework.assess_chemical_accuracy(
            result, accuracy_type="force_accuracy"
        )

        assert assessment.metric_type == "force_accuracy"
        assert assessment.units == "eV/Ang"

    def test_custom_target_accuracy(self):
        """Test assessment with custom target accuracy."""
        result = BenchmarkResult(
            name="TestModel",
            tags={"dataset": "TestDataset"},
            metrics={"mse": Metric(value=0.005)},
            metadata={"execution_time": 1.0, "framework_version": "flax_nnx"},
        )

        assessment = self.framework.assess_chemical_accuracy(
            result, target_accuracy=0.01
        )

        assert assessment.target == 0.01
        assert assessment.passed is True  # 0.005 < 0.01


class TestErrorAnalysis:
    """Test comprehensive error analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ValidationFramework()

    def test_error_analysis_structure(self):
        """Test that error analysis returns proper structure."""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = jnp.array([1.1, 2.0, 2.9, 4.1, 4.9])

        analysis = self.framework.generate_error_analysis(predictions, ground_truth)

        assert isinstance(analysis, ErrorAnalysis)
        assert "mse" in analysis.global_errors
        assert "mae" in analysis.global_errors
        assert "max_error" in analysis.global_errors
        assert "rmse" in analysis.global_errors

    def test_error_distribution_percentiles(self):
        """Test that error distribution includes percentiles."""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = jnp.array([1.1, 2.0, 2.9, 4.1, 4.9])

        analysis = self.framework.generate_error_analysis(predictions, ground_truth)

        assert "percentiles" in analysis.error_distribution
        percentiles = analysis.error_distribution["percentiles"]
        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p99" in percentiles

    def test_error_analysis_moments(self):
        """Test that error analysis includes statistical moments."""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = jnp.array([1.1, 1.9, 3.0, 4.1, 4.9])

        analysis = self.framework.generate_error_analysis(predictions, ground_truth)

        assert "moments" in analysis.error_distribution
        moments = analysis.error_distribution["moments"]
        assert "mean" in moments
        assert "variance" in moments
        assert "skewness" in moments
        assert "kurtosis" in moments

    def test_outlier_analysis(self):
        """Test outlier analysis structure and computation."""
        # Create data with varying errors
        predictions = jnp.array([1.0, 2.1, 3.0, 4.2, 5.0])
        ground_truth = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        analysis = self.framework.generate_error_analysis(predictions, ground_truth)

        # Verify outlier analysis structure exists
        assert "outlier_count" in analysis.outlier_analysis
        assert "outlier_percentage" in analysis.outlier_analysis
        assert "outlier_threshold" in analysis.outlier_analysis
        assert "max_outlier_error" in analysis.outlier_analysis
        # Values should be valid numbers
        assert isinstance(analysis.outlier_analysis["outlier_count"], int)
        assert analysis.outlier_analysis["outlier_percentage"] >= 0

    def test_local_errors_computed(self):
        """Test that local error maps are computed."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        ground_truth = jnp.array([1.1, 2.0, 2.9])

        analysis = self.framework.generate_error_analysis(predictions, ground_truth)

        assert "absolute_errors" in analysis.local_errors
        assert "relative_errors" in analysis.local_errors
        assert "squared_errors" in analysis.local_errors


class TestSkewnessKurtosis:
    """Test skewness and kurtosis computation."""

    def test_skewness_symmetric_distribution(self):
        """Test skewness for approximately symmetric distribution."""
        from opifex.benchmarking.validation_framework import _compute_skewness

        # Symmetric distribution should have skewness near 0
        data = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        skewness = _compute_skewness(data)

        assert abs(float(skewness)) < 0.5

    def test_kurtosis_normal_like_distribution(self):
        """Test kurtosis computation."""
        from opifex.benchmarking.validation_framework import _compute_kurtosis

        # Generate some data
        np.random.seed(42)
        data = jnp.array(np.random.randn(100))
        kurtosis = _compute_kurtosis(data)

        # Excess kurtosis of normal distribution should be near 0
        # Allow some tolerance since sample is finite
        assert abs(float(kurtosis)) < 1.0
