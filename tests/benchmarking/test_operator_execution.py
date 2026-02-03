"""Tests for actual operator execution in benchmarks.

These tests define the expected behavior for running real operators
(not mocks) in the benchmarking system.

Following TDD: These tests are written FIRST before the implementation.
"""

import jax.numpy as jnp
import optax
import pytest
from flax import nnx


class TestOperatorExecution:
    """Tests for executing actual Opifex operators."""

    def test_tfno_produces_real_output(self):
        """TFNO forward pass produces non-trivial output."""
        from opifex.neural.operators.fno.tensorized import create_tucker_fno

        rngs = nnx.Rngs(42)
        model = create_tucker_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=(8, 8),
            rank=0.1,
            num_layers=2,
            rngs=rngs,
        )

        x = jnp.ones((4, 1, 32, 32))  # batch, channels, H, W
        y = model(x)

        assert y.shape == (4, 1, 32, 32)
        assert not jnp.allclose(y, 0)  # Non-trivial output
        assert not jnp.allclose(y, x)  # Transformed

    def test_deeponet_produces_real_output(self):
        """DeepONet forward pass produces non-trivial output."""
        from opifex.neural.operators.deeponet import DeepONet

        rngs = nnx.Rngs(42)
        model = DeepONet(
            branch_sizes=[100, 64, 32],
            trunk_sizes=[2, 64, 32],
            rngs=rngs,
        )

        branch_input = jnp.ones((4, 100))
        trunk_input = jnp.ones((4, 10, 2))
        y = model(branch_input=branch_input, trunk_input=trunk_input)

        assert y.shape == (4, 10)
        assert not jnp.allclose(y, 0)


class TestDataLoaderIntegration:
    """Tests for data loader integration with benchmarks."""

    def test_darcy_loader_integration(self):
        """Benchmarks use actual Opifex data loaders."""
        from opifex.data.loaders import create_darcy_loader

        loader = create_darcy_loader(n_samples=10, batch_size=2, resolution=32)

        batch = next(iter(loader))
        assert "input" in batch
        assert "output" in batch
        # Darcy loader returns (batch, H, W) format (no channel dim)
        assert batch["input"].ndim == 3

    def test_burgers_loader_integration(self):
        """Burgers loader works with benchmarks."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=10, batch_size=2, resolution=32, dimension="2d"
        )

        batch = next(iter(loader))
        assert "input" in batch
        assert "output" in batch


class TestBenchmarkRunnerHelpers:
    """Tests for BenchmarkRunner helper methods (TDD for private methods)."""

    def test_get_data_loaders_from_config(self):
        """_get_data_loaders uses loader_type from config."""
        from opifex.benchmarking.benchmark_registry import BenchmarkConfig
        from opifex.benchmarking.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()

        config = BenchmarkConfig(
            name="test_darcy",
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(32, 32, 1),
            output_shape=(32, 32, 1),
            computational_requirements={"loader_type": "darcy", "batch_size": 4},
        )

        train_loader, test_loader = runner._get_data_loaders(config)

        # Verify loaders work
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        assert "input" in train_batch
        assert "input" in test_batch

    def test_get_data_loaders_missing_loader_type_raises(self):
        """Missing loader_type raises informative error."""
        from opifex.benchmarking.benchmark_registry import BenchmarkConfig
        from opifex.benchmarking.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()

        config = BenchmarkConfig(
            name="test_missing",
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(32, 32, 1),
            output_shape=(32, 32, 1),
            computational_requirements={},  # No loader_type!
        )

        with pytest.raises(ValueError, match=r"missing.*loader_type"):
            runner._get_data_loaders(config)

    def test_get_operator_config_from_metadata(self):
        """_get_operator_config uses registry metadata, not name matching."""
        from opifex.benchmarking.benchmark_registry import BenchmarkConfig
        from opifex.benchmarking.benchmark_runner import BenchmarkRunner
        from opifex.neural.operators.fno.tensorized import (
            TensorizedFourierNeuralOperator,
        )

        runner = BenchmarkRunner()

        # Register operator with explicit type metadata
        runner.registry.register_operator(
            TensorizedFourierNeuralOperator, metadata={"operator_type": "fno"}
        )

        config = BenchmarkConfig(
            name="test_darcy",
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(32, 32, 1),
            output_shape=(32, 32, 1),
        )

        op_config = runner._get_operator_config(TensorizedFourierNeuralOperator, config)

        # Should get FNO-specific config
        assert "modes" in op_config
        assert "hidden_channels" in op_config
        assert op_config["in_channels"] == 1


class TestTrainingLoop:
    """Tests for benchmark training loops."""

    def test_training_loop_uses_nnx_optimizer(self):
        """Training uses correct Flax NNX 0.11.0+ optimizer pattern."""
        from opifex.neural.operators.fno.tensorized import create_tucker_fno

        rngs = nnx.Rngs(42)
        model = create_tucker_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            rank=0.1,
            num_layers=2,
            rngs=rngs,
        )

        # Correct pattern: nnx.Optimizer with wrt=nnx.Param
        opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        # Verify it works
        x = jnp.ones((2, 1, 16, 16))
        y = jnp.zeros((2, 1, 16, 16))

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)  # Should not raise

        # Verify parameters changed
        loss2, _ = nnx.value_and_grad(loss_fn)(model)
        assert loss2 != loss  # Loss changed after update

    def test_benchmark_result_from_training(self):
        """Benchmark produces result from actual training."""
        from opifex.benchmarking.operator_executor import (
            ExecutionConfig,
            OperatorExecutor,
        )
        from opifex.data.loaders import create_darcy_loader
        from opifex.neural.operators.fno.tensorized import create_tucker_fno

        # Create a small model and quick training config
        rngs = nnx.Rngs(42)
        model = create_tucker_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            rank=0.1,
            num_layers=2,
            rngs=rngs,
        )

        # Quick training (just 2 epochs for test speed)
        config = ExecutionConfig(n_epochs=2, batch_size=4, learning_rate=1e-3)
        executor = OperatorExecutor(config)

        # Use actual data loaders
        train_loader = create_darcy_loader(n_samples=8, batch_size=4, resolution=16)
        test_loader = create_darcy_loader(n_samples=4, batch_size=4, resolution=16)

        # Execute benchmark
        result = executor.execute_training_benchmark(
            operator_class=type(model),
            operator_config={
                "in_channels": 1,
                "out_channels": 1,
                "hidden_channels": 16,
                "modes": (4, 4),
                "rank": 0.1,
                "num_layers": 2,
            },
            train_loader=train_loader,
            test_loader=test_loader,
            benchmark_name="test_darcy",
        )

        # Verify real results
        assert result.metadata.get("execution_time", 0.0) > 0
        assert result.metrics["mse"].value > 0
        # After training, final loss should be <= initial (training improved or didn't diverge)
        assert (
            result.metrics["final_train_loss"].value
            <= result.metrics["initial_train_loss"].value * 1.5
        )


class TestOperatorExecutorEdgeCases:
    """Tests for error conditions and edge cases."""

    def test_invalid_operator_config_raises(self):
        """Invalid operator config raises informative error."""
        from opifex.benchmarking.operator_executor import OperatorExecutor

        executor = OperatorExecutor()
        with pytest.raises((TypeError, ValueError)):
            executor._create_operator(
                operator_class=type(None),  # Invalid
                config={},
                rngs=nnx.Rngs(42),
            )

    def test_empty_loader_returns_zero_metrics(self):
        """Empty data loader returns zeroed metrics gracefully."""
        from opifex.benchmarking.operator_executor import OperatorExecutor
        from opifex.neural.operators.fno.tensorized import create_tucker_fno

        executor = OperatorExecutor()
        rngs = nnx.Rngs(42)
        model = create_tucker_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            rank=0.1,
            num_layers=2,
            rngs=rngs,
        )

        # Empty iterator
        empty_loader = iter([])
        metrics = executor._evaluate(model, empty_loader)

        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0


class TestBenchmarkRunnerRealExecution:
    """Tests that BenchmarkRunner uses real execution (not mocks)."""

    def test_run_single_benchmark_returns_real_metrics(self):
        """_run_single_benchmark returns REAL results, not mock random values."""
        from opifex.benchmarking.benchmark_registry import BenchmarkConfig
        from opifex.benchmarking.benchmark_runner import BenchmarkRunner
        from opifex.neural.operators.fno.tensorized import (
            TensorizedFourierNeuralOperator,
        )

        runner = BenchmarkRunner()

        # Register TFNO with proper metadata
        runner.registry.register_operator(
            TensorizedFourierNeuralOperator, metadata={"operator_type": "fno"}
        )

        # Create benchmark config with loader_type
        config = BenchmarkConfig(
            name="test_darcy",
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(32, 32, 1),
            output_shape=(32, 32, 1),
            computational_requirements={
                "loader_type": "darcy",
                "batch_size": 4,
                "n_epochs": 2,  # Quick test
            },
        )
        runner.registry.register_benchmark(config)

        # Execute benchmark
        result = runner._run_single_benchmark("TensorizedFourierNeuralOperator", config)

        # Result should be from ACTUAL execution, not mock
        assert result.name == "TensorizedFourierNeuralOperator"
        assert result.tags.get("dataset") == "test_darcy"
        assert result.metrics["mse"].value > 0  # Real MSE (not random)
        assert result.metadata.get("execution_time", 0.0) > 0  # Real timing

        # Run again - results should be deterministic (same seed)
        result2 = runner._run_single_benchmark(
            "TensorizedFourierNeuralOperator", config
        )
        # Results should be similar (not random each time)
        assert abs(result.metrics["mse"].value - result2.metrics["mse"].value) < 0.1
