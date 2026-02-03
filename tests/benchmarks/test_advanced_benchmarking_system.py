"""Tests for Advanced Benchmarking System

Comprehensive tests for the new benchmarking infrastructure including
BenchmarkRegistry, ValidationFramework, AnalysisEngine, ResultsManager,
and BenchmarkRunner components.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from opifex.benchmarking.analysis_engine import AnalysisEngine, ComparisonReport
from opifex.benchmarking.benchmark_registry import (
    BenchmarkConfig,
    BenchmarkRegistry,
    DomainConfig,
)
from opifex.benchmarking.benchmark_runner import BenchmarkRunner
from opifex.benchmarking.evaluation_engine import BenchmarkResult
from opifex.benchmarking.results_manager import ResultsManager
from opifex.benchmarking.validation_framework import (
    ValidationFramework,
    ValidationReport,
)


class TestBenchmarkRegistry:
    """Test the BenchmarkRegistry component."""

    def test_registry_initialization(self):
        """Test registry initializes with default domains."""
        registry = BenchmarkRegistry()

        domains = registry.list_available_domains()
        assert "fluid_dynamics" in domains
        assert "quantum_computing" in domains
        assert "materials_science" in domains

    def test_domain_config_creation(self):
        """Test domain configuration with defaults."""
        config = DomainConfig("fluid_dynamics")

        assert config.name == "fluid_dynamics"
        assert "mse" in config.tolerance_ranges
        assert "mae" in config.required_metrics
        assert len(config.default_problem_sizes) > 0

    def test_benchmark_registration(self):
        """Test benchmark registration."""
        registry = BenchmarkRegistry()

        benchmark = BenchmarkConfig(
            name="test_darcy",
            domain="fluid_dynamics",
            problem_type="pde_solving",
            input_shape=(64, 64),
            output_shape=(64, 64),
        )

        registry.register_benchmark(benchmark)

        benchmarks = registry.list_available_benchmarks()
        assert "test_darcy" in benchmarks

        retrieved = registry.get_benchmark_config("test_darcy")
        assert retrieved.name == "test_darcy"
        assert retrieved.domain == "fluid_dynamics"

    def test_domain_specific_suite(self):
        """Test domain-specific benchmark suite retrieval."""
        registry = BenchmarkRegistry()

        # Add multiple benchmarks for fluid dynamics
        for i in range(3):
            benchmark = BenchmarkConfig(
                name=f"fluid_test_{i}",
                domain="fluid_dynamics",
                problem_type="pde_solving",
                input_shape=(64, 64),
                output_shape=(64, 64),
            )
            registry.register_benchmark(benchmark)

        fluid_benchmarks = registry.get_benchmark_suite("fluid_dynamics")
        assert len(fluid_benchmarks) == 3
        assert all(b.domain == "fluid_dynamics" for b in fluid_benchmarks)

    def test_registry_persistence(self):
        """Test registry configuration save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_registry.json"

            # Create registry and add benchmark
            registry = BenchmarkRegistry(str(config_path))
            benchmark = BenchmarkConfig(
                name="persist_test",
                domain="quantum_computing",
                problem_type="dft",
                input_shape=(32, 32),
                output_shape=(32, 32),
            )
            registry.register_benchmark(benchmark)
            registry.save_registry()

            # Load new registry and verify persistence
            registry2 = BenchmarkRegistry(str(config_path))
            benchmarks = registry2.list_available_benchmarks()
            assert "persist_test" in benchmarks


class TestValidationFramework:
    """Test the ValidationFramework component."""

    def test_framework_initialization(self):
        """Test validation framework initialization."""
        validator = ValidationFramework()

        assert len(validator.default_tolerances) > 0
        assert "quantum_computing" in validator.chemical_accuracy_thresholds

    def test_domain_inference(self):
        """Test domain inference from dataset names."""
        validator = ValidationFramework()

        assert validator._infer_domain("darcy_flow_dataset") == "fluid_dynamics"
        assert validator._infer_domain("quantum_dft_molecule") == "quantum_computing"
        assert (
            validator._infer_domain("materials_crystal_structure")
            == "materials_science"
        )
        assert validator._infer_domain("unknown_dataset") == "general"

    def test_tolerance_violation_checking(self):
        """Test tolerance violation detection."""
        validator = ValidationFramework()

        # High error metrics should trigger violations
        high_error_metrics = {
            "mse": 1.0,  # Very high MSE
            "relative_error": 0.5,  # 50% relative error
        }

        violations = validator._check_tolerance_violations(
            high_error_metrics, "fluid_dynamics_test"
        )

        assert len(violations) > 0
        assert any("MSE exceeds" in v for v in violations)

    def test_validation_report_creation(self):
        """Test validation report creation."""
        validator = ValidationFramework()

        result = BenchmarkResult(
            model_name="test_fno",
            dataset_name="test_darcy",
            metrics={"mse": 1e-4, "mae": 1e-3},
            execution_time=1.0,
        )

        report = validator.validate_against_reference(result, "synthetic_reference")

        assert isinstance(report, ValidationReport)
        assert report.benchmark_name == "test_darcy"
        assert report.reference_method == "synthetic_reference"
        assert "mse" in report.accuracy_metrics

    def test_chemical_accuracy_assessment(self):
        """Test chemical accuracy assessment."""
        validator = ValidationFramework()

        # Good accuracy result
        good_result = BenchmarkResult(
            model_name="test_dft",
            dataset_name="quantum_molecule",
            metrics={"mse": 1e-4},  # Better than chemical accuracy threshold
            execution_time=1.0,
        )

        assessment = validator.assess_chemical_accuracy(good_result)
        assert assessment.passed
        assert assessment.achieved_accuracy == 1e-4

        # Poor accuracy result
        poor_result = BenchmarkResult(
            model_name="test_dft",
            dataset_name="quantum_molecule",
            metrics={"mse": 1e-2},  # Worse than chemical accuracy threshold
            execution_time=1.0,
        )

        assessment = validator.assess_chemical_accuracy(poor_result)
        assert not assessment.passed

    def test_error_analysis(self):
        """Test comprehensive error analysis."""
        validator = ValidationFramework()

        # Create test data
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = jnp.array([1.1, 2.1, 2.9, 4.2, 4.8])

        analysis = validator.generate_error_analysis(predictions, ground_truth)

        assert "mse" in analysis.global_errors
        assert "mae" in analysis.global_errors
        assert analysis.global_errors["mse"] > 0
        assert analysis.outlier_analysis["outlier_count"] >= 0


class TestAnalysisEngine:
    """Test the AnalysisEngine component."""

    def test_engine_initialization(self):
        """Test analysis engine initialization."""
        analyzer = AnalysisEngine()

        assert analyzer.significance_threshold == 0.05
        assert "fluid_dynamics" in analyzer.metric_weights

    def test_operator_comparison(self):
        """Test multi-operator comparison."""
        analyzer = AnalysisEngine()

        # Create test results for comparison
        results = {
            "fno": BenchmarkResult(
                model_name="fno",
                dataset_name="test_benchmark",
                metrics={"mse": 1e-3, "mae": 1e-2},
                execution_time=1.0,
            ),
            "deeponet": BenchmarkResult(
                model_name="deeponet",
                dataset_name="test_benchmark",
                metrics={"mse": 2e-3, "mae": 1.5e-2},
                execution_time=2.0,
            ),
        }

        comparison = analyzer.compare_operators(results)

        assert isinstance(comparison, ComparisonReport)
        assert comparison.benchmark_name == "test_benchmark"
        assert len(comparison.operators_compared) == 2
        assert "fno" in comparison.operators_compared
        assert "deeponet" in comparison.operators_compared
        assert comparison.overall_winner in ["fno", "deeponet"]

    def test_performance_insights(self):
        """Test performance insights generation."""
        analyzer = AnalysisEngine()

        # Excellent performance result
        excellent_result = BenchmarkResult(
            model_name="efficient_fno",
            dataset_name="test_problem",
            metrics={"mse": 1e-6, "mae": 1e-5, "relative_error": 1e-4},
            execution_time=0.1,
        )

        insights = analyzer.generate_performance_insights(excellent_result)

        assert len(insights.key_insights) > 0
        assert insights.confidence_level > 0
        assert (
            "exceptional" in " ".join(insights.key_insights).lower()
            or "excellent" in " ".join(insights.key_insights).lower()
        )

    def test_operator_recommendations(self):
        """Test operator recommendations."""
        analyzer = AnalysisEngine()

        recommendations = analyzer.create_operator_recommendations(
            "pde_solving", "fluid_dynamics"
        )

        assert recommendations.problem_type == "pde_solving"
        assert recommendations.domain == "fluid_dynamics"
        assert len(recommendations.recommended_operators) > 0
        assert len(recommendations.implementation_considerations) > 0


class TestResultsManager:
    """Test the ResultsManager component."""

    def test_manager_initialization(self):
        """Test results manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)

            assert manager.storage_path.exists()
            assert manager.plots_path.exists()
            assert manager.tables_path.exists()

    def test_result_persistence(self):
        """Test benchmark result saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)

            result = BenchmarkResult(
                model_name="test_model",
                dataset_name="test_dataset",
                metrics={"mse": 1e-3, "mae": 1e-2},
                execution_time=1.5,
            )

            # Save result
            result_id = manager.save_benchmark_results(result)
            assert result_id is not None

            # Load result
            loaded = manager.load_results(result_id)
            assert loaded is not None
            assert loaded.model_name == "test_model"
            assert loaded.dataset_name == "test_dataset"
            assert loaded.metrics["mse"] == 1e-3

    def test_database_statistics(self):
        """Test database statistics generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique database path to avoid interference
            db_path = Path(tmpdir) / "test_stats_db.json"
            manager = ResultsManager(storage_path=tmpdir, database_path=str(db_path))

            # Add some test results
            for i in range(3):
                result = BenchmarkResult(
                    model_name=f"stats_model_{i}",
                    dataset_name="stats_test_dataset",
                    metrics={"mse": 1e-3 * (i + 1)},
                    execution_time=1.0 + i,
                )
                manager.save_benchmark_results(result)

            stats = manager.get_database_statistics()

            assert stats["total_results"] == 3
            assert stats["unique_models"] == 3
            assert stats["unique_datasets"] == 1

    def test_query_functionality(self):
        """Test database querying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique database path within the temporary directory
            db_path = Path(tmpdir) / "test_database.json"
            manager = ResultsManager(storage_path=tmpdir, database_path=str(db_path))

            # Add test results
            result1 = BenchmarkResult(
                model_name="fast_model",
                dataset_name="dataset_a",
                metrics={"mse": 1e-4},
                execution_time=0.5,
            )
            result2 = BenchmarkResult(
                model_name="slow_model",
                dataset_name="dataset_b",
                metrics={"mse": 1e-2},
                execution_time=5.0,
            )

            manager.save_benchmark_results(result1)
            manager.save_benchmark_results(result2)

            # Query by model name
            fast_results = manager.query_results(model_name="fast_model")
            assert len(fast_results) == 1
            assert fast_results[0]["model_name"] == "fast_model"

            # Query by metric filter
            accurate_results = manager.query_results(metric_filter={"mse": (0.0, 1e-3)})
            assert len(accurate_results) == 1


class TestBenchmarkRunner:
    """Test the BenchmarkRunner orchestration component."""

    def test_runner_initialization(self):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner()

        assert runner.registry is not None
        assert runner.evaluator is not None
        assert runner.validator is not None
        assert runner.analyzer is not None
        assert runner.results_manager is not None

    def test_runner_with_custom_components(self):
        """Test runner with custom component initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BenchmarkRegistry()
            results_manager = ResultsManager(storage_path=tmpdir)

            runner = BenchmarkRunner(
                registry=registry,
                results_manager=results_manager,
                output_dir=tmpdir,
            )

            assert runner.registry is registry
            assert runner.results_manager is results_manager

    def test_single_benchmark_execution(self):
        """Test single benchmark execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir)

            # Add a test benchmark
            benchmark = BenchmarkConfig(
                name="exec_test_benchmark",
                domain="fluid_dynamics",
                problem_type="pde_solving",
                input_shape=(32, 32),
                output_shape=(32, 32),
            )
            runner.registry.register_benchmark(benchmark)

            # Mock operator for testing
            class MockFNO:
                def __init__(self):
                    pass

            # Register with correct name reference
            MockFNO.__name__ = "MockFNO"
            runner.registry.register_operator(MockFNO)

            # Verify registration worked
            operators = runner.registry.list_available_operators()
            assert "MockFNO" in operators

            # Run single benchmark
            result = runner._run_single_benchmark("MockFNO", benchmark)

            assert isinstance(result, BenchmarkResult)
            assert result.model_name == "MockFNO"
            assert result.dataset_name == "exec_test_benchmark"
            assert "mse" in result.metrics

    def test_database_update(self):
        """Test benchmark database update."""
        runner = BenchmarkRunner()

        summary = runner.update_benchmark_database()

        assert "database_path" in summary
        assert "total_results" in summary
        assert isinstance(summary["total_results"], int)


class TestIntegrationWorkflows:
    """Test complete integration workflows."""

    def test_end_to_end_benchmarking_workflow(self):
        """Test complete end-to-end benchmarking workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize runner with temporary storage
            runner = BenchmarkRunner(
                results_manager=ResultsManager(storage_path=tmpdir),
                output_dir=tmpdir,
            )

            # Setup test benchmarks
            benchmark = BenchmarkConfig(
                name="integration_test",
                domain="fluid_dynamics",
                problem_type="pde_solving",
                input_shape=(64, 64),
                output_shape=(64, 64),
            )
            runner.registry.register_benchmark(benchmark)

            # Mock operators
            class TestFNO:
                __name__ = "TestFNO"

            class TestDeepONet:
                __name__ = "TestDeepONet"

            runner.registry.register_operator(TestFNO)
            runner.registry.register_operator(TestDeepONet)

            # Run comprehensive benchmark
            results = runner.run_comprehensive_benchmark(
                operators=["TestFNO", "TestDeepONet"],
                benchmarks=["integration_test"],
                validate_results=True,
                generate_analysis=True,
            )

            assert "integration_test" in results
            assert len(results["integration_test"]) <= 2  # Depends on compatibility

            # Generate publication report
            report = runner.generate_publication_report(results)

            assert report.title is not None
            assert len(report.abstract) > 0
            assert len(report.methodology) > 0

    def test_domain_specific_workflow(self):
        """Test domain-specific benchmarking workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(output_dir=tmpdir)

            # Add multiple fluid dynamics benchmarks
            for i in range(2):
                benchmark = BenchmarkConfig(
                    name=f"fluid_test_{i}",
                    domain="fluid_dynamics",
                    problem_type="pde_solving",
                    input_shape=(32, 32),
                    output_shape=(32, 32),
                )
                runner.registry.register_benchmark(benchmark)

            # Mock operator
            class FluidFNO:
                __name__ = "FluidFNO"

            runner.registry.register_operator(FluidFNO)

            # Run domain-specific suite
            domain_results = runner.execute_domain_specific_suite("fluid_dynamics")

            assert domain_results.domain == "fluid_dynamics"
            assert (
                len(domain_results.benchmark_results) <= 2
            )  # Depends on compatibility
            assert "total_benchmarks" in domain_results.summary_statistics


# Run tests if script is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
