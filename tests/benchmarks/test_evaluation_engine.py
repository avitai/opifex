"""Tests for the benchmarking evaluation engine.

This module contains comprehensive tests for the core benchmarking evaluation engine,
following test-driven development principles. Tests cover PDEBench compatibility,
statistical analysis, automated evaluation pipelines, and result management.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.benchmarking.evaluation_engine import (
    BenchmarkEvaluator,
    BenchmarkResult,
    EvaluationMetrics,
    StatisticalAnalyzer,
)
from opifex.neural.operators.foundations import (
    DeepONet,
    FourierNeuralOperator,
)


class TestBenchmarkResult:
    """Test BenchmarkResult data structure."""

    def test_benchmark_result_initialization(self):
        """Test BenchmarkResult initialization with all required fields."""
        metrics = {
            "mse": 0.001,
            "mae": 0.02,
            "relative_error": 0.05,
            "r2_score": 0.98,
        }

        result = BenchmarkResult(
            model_name="test_fno",
            dataset_name="burgers_1d",
            metrics=metrics,
            execution_time=120.5,
            memory_usage=1024,
            gpu_memory_usage=512,
            framework_version="1.0.0",
            timestamp="2024-12-17T10:00:00Z",
        )

        assert result.model_name == "test_fno"
        assert result.dataset_name == "burgers_1d"
        assert result.metrics["mse"] == 0.001
        assert result.execution_time == 120.5
        assert result.memory_usage == 1024
        assert result.gpu_memory_usage == 512
        assert result.framework_version == "1.0.0"

    def test_benchmark_result_validation(self):
        """Test BenchmarkResult validation for required fields."""
        # Test missing required fields
        with pytest.raises(ValueError, match="model_name is required"):
            BenchmarkResult(
                model_name="",
                dataset_name="test",
                metrics={},
                execution_time=0.0,
            )

        # Test negative execution time
        with pytest.raises(ValueError, match="execution_time must be positive"):
            BenchmarkResult(
                model_name="test",
                dataset_name="test",
                metrics={},
                execution_time=-1.0,
            )


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""

    def test_evaluation_metrics_initialization(self):
        """Test EvaluationMetrics initialization."""
        metrics = EvaluationMetrics()

        # Check that all standard metrics are available
        available_metrics = metrics.get_available_metrics()
        assert "mse" in available_metrics
        assert "mae" in available_metrics
        assert "relative_error" in available_metrics
        assert "r2_score" in available_metrics

    def test_mse_calculation(self):
        """Test MSE calculation."""
        metrics = EvaluationMetrics()

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        mse = metrics.calculate_mse(predictions, targets)
        expected_mse = jnp.mean((predictions - targets) ** 2)

        assert jnp.allclose(mse, expected_mse)
        assert mse > 0.0

    def test_mae_calculation(self):
        """Test MAE calculation."""
        metrics = EvaluationMetrics()

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        mae = metrics.calculate_mae(predictions, targets)
        expected_mae = jnp.mean(jnp.abs(predictions - targets))

        assert jnp.allclose(mae, expected_mae)
        assert mae > 0.0

    def test_relative_error_calculation(self):
        """Test relative error calculation."""
        metrics = EvaluationMetrics()

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        rel_error = metrics.calculate_relative_error(predictions, targets)
        expected_rel_error = jnp.mean(
            jnp.abs(predictions - targets) / (jnp.abs(targets) + 1e-8)
        )

        assert jnp.allclose(rel_error, expected_rel_error)
        assert rel_error > 0.0

    def test_r2_score_calculation(self):
        """Test R² score calculation."""
        metrics = EvaluationMetrics()

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.1, 1.9, 3.2, 3.8])

        r2 = metrics.calculate_r2_score(predictions, targets)

        # R² should be between 0 and 1 for reasonable predictions
        assert 0.0 <= r2 <= 1.0


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""

    def test_statistical_analyzer_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer(
            confidence_level=0.95,
            bootstrap_samples=1000,
        )

        assert analyzer.confidence_level == 0.95
        assert analyzer.bootstrap_samples == 1000

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()

        # Generate sample data
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (100,)) * 0.1 + 0.5

        ci_lower, ci_upper = analyzer.calculate_confidence_interval(data)

        # Confidence interval should bracket the mean
        mean_val = jnp.mean(data)
        assert ci_lower < mean_val < ci_upper
        assert ci_lower >= 0.0
        assert ci_upper >= ci_lower

    def test_significance_testing(self):
        """Test statistical significance testing."""
        analyzer = StatisticalAnalyzer()

        # Generate two datasets with known difference
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key, 2)

        data1 = jax.random.normal(key1, (50,)) * 0.1 + 0.5
        data2 = jax.random.normal(key2, (50,)) * 0.1 + 0.7  # Higher mean

        p_value, is_significant = analyzer.test_significance(data1, data2)

        assert 0.0 <= p_value <= 1.0
        assert isinstance(is_significant, bool)


class TestBenchmarkEvaluator:
    """Test the main benchmark evaluator."""

    def test_benchmark_evaluator_initialization(self):
        """Test BenchmarkEvaluator initialization."""
        evaluator = BenchmarkEvaluator(
            output_dir="./benchmark_results",
            save_detailed_results=True,
            enable_gpu_profiling=True,
        )

        assert str(evaluator.output_dir) == "benchmark_results"
        assert evaluator.save_detailed_results is True
        assert evaluator.enable_gpu_profiling is True

    def test_evaluate_model_with_fno(self):
        """Test model evaluation with FourierNeuralOperator."""
        evaluator = BenchmarkEvaluator()

        # Create a simple FNO model
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Generate synthetic test data
        key = jax.random.PRNGKey(0)
        batch_size = 4
        # FNO expects (batch, in_channels, *spatial_dims)
        input_data = jax.random.normal(key, (batch_size, 1, 64))
        target_data = jax.random.normal(key, (batch_size, 1, 64))

        # Evaluate model
        result = evaluator.evaluate_model(
            model=model,
            model_name="test_fno",
            input_data=input_data,
            target_data=target_data,
            dataset_name="synthetic_1d",
        )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test_fno"
        assert result.dataset_name == "synthetic_1d"
        assert "mse" in result.metrics
        assert "mae" in result.metrics
        assert result.execution_time > 0.0

    def test_evaluate_model_with_deeponet(self):
        """Test model evaluation with DeepONet."""
        evaluator = BenchmarkEvaluator()

        # Create a simple DeepONet model
        rngs = nnx.Rngs(42)
        model = DeepONet(
            branch_sizes=[64, 32, 16],
            trunk_sizes=[1, 32, 16],
            rngs=rngs,
        )

        # Generate synthetic test data for DeepONet
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)

        batch_size = 4
        num_locations = 20
        branch_data = jax.random.normal(key1, (batch_size, 64))
        trunk_data = jax.random.normal(key2, (batch_size, num_locations, 1))
        target_data = jax.random.normal(key, (batch_size, num_locations))

        # Evaluate model with custom forward function
        def forward_fn(model, inputs):
            branch_input, trunk_input = inputs
            return model(branch_input, trunk_input)

        result = evaluator.evaluate_model(
            model=model,
            model_name="test_deeponet",
            input_data=(branch_data, trunk_data),
            target_data=target_data,
            dataset_name="synthetic_deeponet",
            forward_fn=forward_fn,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test_deeponet"
        assert result.dataset_name == "synthetic_deeponet"
        assert "mse" in result.metrics
        assert result.execution_time > 0.0

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple models."""
        evaluator = BenchmarkEvaluator()

        # Create multiple models
        rngs = nnx.Rngs(42)

        fno_model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        deeponet_model = DeepONet(
            branch_sizes=[32, 16],
            trunk_sizes=[1, 16],
            rngs=rngs,
        )

        models = [
            ("small_fno", fno_model),
            ("small_deeponet", deeponet_model),
        ]

        # Generate test datasets
        key = jax.random.PRNGKey(0)

        # FNO dataset - format: (batch, in_channels, *spatial_dims)
        fno_input = jax.random.normal(key, (2, 1, 32))
        fno_target = jax.random.normal(key, (2, 1, 32))

        # DeepONet dataset
        branch_input = jax.random.normal(key, (2, 32))
        trunk_input = jax.random.normal(key, (2, 10, 1))
        deeponet_target = jax.random.normal(key, (2, 10))

        datasets = [
            ("fno_dataset", fno_input, fno_target, None),
            (
                "deeponet_dataset",
                (branch_input, trunk_input),
                deeponet_target,
                lambda model, inputs: model(inputs[0], inputs[1]),
            ),
        ]

        # Run batch evaluation
        results = evaluator.batch_evaluate(models, datasets)

        # Note: We expect only 2 results because FNO and DeepONet have
        # incompatible input formats
        # FNO works with single tensors, DeepONet works with tuple inputs
        assert len(results) == 2  # Only compatible model-dataset pairs
        assert all(isinstance(result, BenchmarkResult) for result in results)

        # Check that compatible combinations are covered
        model_names = {result.model_name for result in results}
        dataset_names = {result.dataset_name for result in results}

        # Each model should be evaluated on its compatible dataset
        assert "small_fno" in model_names or "small_deeponet" in model_names
        assert "fno_dataset" in dataset_names or "deeponet_dataset" in dataset_names

    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        evaluator = BenchmarkEvaluator(enable_gpu_profiling=True)

        # Create a model for profiling
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Generate test data
        key = jax.random.PRNGKey(0)
        # FNO expects (batch, in_channels, *spatial_dims)
        input_data = jax.random.normal(key, (8, 1, 64))

        # Profile model performance
        profile_result = evaluator.profile_model_performance(
            model=model,
            input_data=input_data,
            num_runs=10,
        )

        assert "mean_execution_time" in profile_result
        assert "std_execution_time" in profile_result
        assert "memory_usage" in profile_result
        assert profile_result["mean_execution_time"] > 0.0
        assert profile_result["std_execution_time"] >= 0.0

    def test_result_saving_and_loading(self):
        """Test saving and loading benchmark results."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = BenchmarkEvaluator(
                output_dir=temp_dir,
                save_detailed_results=True,
            )

            # Create a benchmark result
            metrics = {"mse": 0.001, "mae": 0.02}
            result = BenchmarkResult(
                model_name="test_model",
                dataset_name="test_dataset",
                metrics=metrics,
                execution_time=10.0,
            )

            # Save result
            evaluator.save_result(result)

            # Check that file was created in the raw_evaluations subdirectory
            raw_results_dir = Path(temp_dir) / "raw_evaluations"
            result_files = [f for f in raw_results_dir.iterdir() if f.suffix == ".json"]
            assert len(result_files) > 0

            # Load results
            loaded_results = evaluator.load_results()
            assert len(loaded_results) == 1
            assert loaded_results[0].model_name == "test_model"
            assert loaded_results[0].dataset_name == "test_dataset"
            assert loaded_results[0].metrics["mse"] == 0.001


class TestBenchmarkIntegration:
    """Test integration between different benchmarking components."""

    def test_end_to_end_benchmark_workflow(self):
        """Test complete benchmark workflow from model to analysis."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            evaluator = BenchmarkEvaluator(output_dir=temp_dir)
            analyzer = StatisticalAnalyzer()

            # Create test model
            rngs = nnx.Rngs(42)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                modes=4,
                num_layers=1,
                rngs=rngs,
            )

            # Generate multiple test runs
            results = []
            for run_id in range(5):
                key = jax.random.PRNGKey(run_id)
                # FNO expects (batch, in_channels, *spatial_dims)
                input_data = jax.random.normal(key, (4, 1, 32))
                target_data = jax.random.normal(key, (4, 1, 32))

                result = evaluator.evaluate_model(
                    model=model,
                    model_name=f"fno_run_{run_id}",
                    input_data=input_data,
                    target_data=target_data,
                    dataset_name="multiple_runs",
                )
                results.append(result)

            # Analyze results statistically
            mse_values = jnp.array([result.metrics["mse"] for result in results])
            ci_lower, ci_upper = analyzer.calculate_confidence_interval(mse_values)

            assert len(results) == 5
            assert all(result.metrics["mse"] > 0 for result in results)
            assert ci_lower < ci_upper

            # Test that all results were saved
            loaded_results = evaluator.load_results()
            assert len(loaded_results) == 5

    def test_cross_model_comparison(self):
        """Test statistical comparison between different models."""
        evaluator = BenchmarkEvaluator()
        analyzer = StatisticalAnalyzer()

        # Create two different models
        rngs = nnx.Rngs(42)

        model1 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        model2 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Evaluate both models on same dataset multiple times
        key = jax.random.PRNGKey(0)
        # FNO expects (batch, in_channels, *spatial_dims)
        input_data = jax.random.normal(key, (4, 1, 32))
        target_data = jax.random.normal(key, (4, 1, 32))

        model1_results = []
        model2_results = []

        for run_id in range(10):
            # Add small noise to create variation
            noise_key = jax.random.PRNGKey(run_id)
            noise = jax.random.normal(noise_key, input_data.shape) * 0.01
            noisy_input = input_data + noise

            result1 = evaluator.evaluate_model(
                model=model1,
                model_name="small_fno",
                input_data=noisy_input,
                target_data=target_data,
                dataset_name="comparison",
            )

            result2 = evaluator.evaluate_model(
                model=model2,
                model_name="large_fno",
                input_data=noisy_input,
                target_data=target_data,
                dataset_name="comparison",
            )

            model1_results.append(result1.metrics["mse"])
            model2_results.append(result2.metrics["mse"])

        # Statistical comparison
        model1_mse = jnp.array(model1_results)
        model2_mse = jnp.array(model2_results)

        p_value, is_significant = analyzer.test_significance(model1_mse, model2_mse)

        assert 0.0 <= p_value <= 1.0
        assert isinstance(is_significant, bool)
        assert len(model1_results) == 10
        assert len(model2_results) == 10


class TestPDEBenchIntegration:
    """Test PDEBench dataset integration and evaluation."""

    def test_pdebench_dataset_loader_initialization(self):
        """Test PDEBench dataset loader initialization."""
        # Import the module we're about to implement
        from opifex.benchmarking.pdebench_integration import PDEBenchLoader

        loader = PDEBenchLoader()
        assert loader.supported_datasets is not None
        assert isinstance(loader.supported_datasets, list)
        assert len(loader.supported_datasets) > 0

    def test_pdebench_advection_dataset_loading(self):
        """Test loading PDEBench Advection dataset."""
        from opifex.benchmarking.pdebench_integration import PDEBenchLoader

        loader = PDEBenchLoader()

        # Test loading synthetic advection data
        dataset = loader.load_dataset(
            dataset_name="advection",
            subset_size=10,  # Small subset for testing
            resolution="low",
        )

        assert "input_data" in dataset
        assert "target_data" in dataset
        assert "metadata" in dataset
        assert dataset["input_data"].shape[0] == 10  # batch size
        assert len(dataset["input_data"].shape) >= 3  # (batch, channels, spatial)

    def test_pdebench_burgers_dataset_loading(self):
        """Test loading PDEBench Burgers equation dataset."""
        from opifex.benchmarking.pdebench_integration import PDEBenchLoader

        loader = PDEBenchLoader()
        dataset = loader.load_dataset(
            dataset_name="burgers", subset_size=5, resolution="low"
        )

        assert "input_data" in dataset
        assert "target_data" in dataset
        assert "metadata" in dataset
        assert dataset["metadata"]["equation"] == "burgers"

    def test_pdebench_darcy_flow_dataset_loading(self):
        """Test loading PDEBench Darcy Flow dataset."""
        from opifex.benchmarking.pdebench_integration import PDEBenchLoader

        loader = PDEBenchLoader()
        dataset = loader.load_dataset(
            dataset_name="darcy_flow", subset_size=5, resolution="low"
        )

        assert "input_data" in dataset
        assert "target_data" in dataset
        assert dataset["metadata"]["equation"] == "darcy_flow"
        assert dataset["input_data"].ndim >= 3  # 2D spatial + batch

    def test_pdebench_baseline_comparison(self):
        """Test comparison with PDEBench baseline results."""
        from opifex.benchmarking.baseline_repository import BaselineRepository
        from opifex.benchmarking.pdebench_integration import PDEBenchLoader

        loader = PDEBenchLoader()
        baseline_repo = BaselineRepository()

        # Load a small dataset
        loader.load_dataset("advection", subset_size=3, resolution="low")

        # Get baseline performance metrics
        baseline_metrics = baseline_repo.get_baseline_metrics(
            dataset_name="advection", model_type="fno"
        )

        assert "mse" in baseline_metrics
        assert "mae" in baseline_metrics
        assert baseline_metrics["mse"] > 0

    def test_pdebench_automated_evaluation_pipeline(self):
        """Test automated evaluation pipeline with PDEBench datasets."""
        from opifex.benchmarking.pdebench_integration import (
            PDEBenchEvaluationPipeline,
        )
        from opifex.neural.operators.foundations import FourierNeuralOperator

        pipeline = PDEBenchEvaluationPipeline()

        # Create a simple FNO model for testing
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        # Run evaluation on small subset
        results = pipeline.evaluate_model_on_datasets(
            model=model,
            model_name="test_fno",
            datasets=["advection"],
            subset_size=3,
            resolution="low",
        )

        assert len(results) > 0
        assert all(isinstance(result, BenchmarkResult) for result in results)
        assert results[0].dataset_name == "advection"

    def test_pdebench_visualization_integration(self):
        """Test visualization tools integration with PDEBench results."""
        from opifex.benchmarking.visualization_tools import PDEBenchVisualizer

        visualizer = PDEBenchVisualizer()

        # Create mock benchmark results
        mock_results = [
            BenchmarkResult(
                model_name="FNO",
                dataset_name="advection",
                metrics={"mse": 0.001, "mae": 0.01},
                execution_time=5.0,
            ),
            BenchmarkResult(
                model_name="DeepONet",
                dataset_name="advection",
                metrics={"mse": 0.002, "mae": 0.015},
                execution_time=7.0,
            ),
        ]

        # Test figure generation (should return figure metadata, not actual plots)
        figure_metadata = visualizer.create_comparison_chart(
            results=mock_results,
            metric="mse",
            title="PDEBench Advection Comparison",
        )

        assert "figure_type" in figure_metadata
        assert "metrics_compared" in figure_metadata
        assert figure_metadata["figure_type"] == "comparison_chart"

    def test_pdebench_report_generation(self):
        """Test automated report generation for PDEBench results."""
        from opifex.benchmarking.report_generator import PDEBenchReportGenerator

        report_gen = PDEBenchReportGenerator()

        # Create mock results for multiple datasets
        mock_results = [
            BenchmarkResult(
                model_name="FNO",
                dataset_name="advection",
                metrics={"mse": 0.001, "mae": 0.01, "r2_score": 0.95},
                execution_time=5.0,
            ),
            BenchmarkResult(
                model_name="FNO",
                dataset_name="burgers",
                metrics={"mse": 0.0015, "mae": 0.012, "r2_score": 0.93},
                execution_time=6.0,
            ),
        ]

        # Generate report
        report = report_gen.generate_comprehensive_report(
            results=mock_results,
            include_baseline_comparison=True,
            include_statistical_analysis=True,
        )

        assert "evaluation_summary" in report
        assert "detailed_results" in report
        assert "statistical_analysis" in report
        assert len(report["detailed_results"]) == 2
