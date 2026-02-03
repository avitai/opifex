"""Tests for benchmark configuration management.

Following TDD: These tests are written FIRST before the implementation.
"""

import pytest

from opifex.benchmarking.benchmark_registry import BenchmarkConfig, BenchmarkRegistry


class TestPDEBenchConfigs:
    """Tests for PDEBench-compatible benchmark configurations."""

    @pytest.fixture
    def registry_with_pdebench(self):
        """Create registry with PDEBench configs registered."""
        from opifex.benchmarking.pdebench_configs import register_pdebench_benchmarks

        registry = BenchmarkRegistry()
        register_pdebench_benchmarks(registry)
        return registry

    def test_darcy_config_exists(self, registry_with_pdebench):
        """Darcy flow benchmark config is properly defined."""
        config = registry_with_pdebench.get_benchmark_config("PDEBench_2D_DarcyFlow")

        assert config.domain == "fluid_dynamics"
        assert config.input_shape == (128, 128, 1)
        assert config.output_shape == (128, 128, 1)
        # DRY: Loader type should be in config, not inferred from name
        assert config.computational_requirements.get("loader_type") == "darcy"

    def test_burgers_config_exists(self, registry_with_pdebench):
        """Burgers equation benchmark config is properly defined."""
        config = registry_with_pdebench.get_benchmark_config("PDEBench_1D_Burgers")

        assert config.domain == "fluid_dynamics"
        assert "viscosity" in config.physics_constraints
        assert config.computational_requirements.get("loader_type") == "burgers"

    def test_navier_stokes_config_exists(self, registry_with_pdebench):
        """Navier-Stokes benchmark config is properly defined."""
        config = registry_with_pdebench.get_benchmark_config("PDEBench_2D_NavierStokes")

        assert config.domain == "fluid_dynamics"
        assert config.computational_requirements.get("loader_type") == "navier_stokes"

    def test_missing_config_raises_valueerror(self):
        """Requesting non-existent config raises ValueError."""
        registry = BenchmarkRegistry()
        with pytest.raises(ValueError, match="not found"):
            registry.get_benchmark_config("NonExistent_Benchmark")


class TestRealPDEBenchConfigs:
    """Tests for RealPDEBench configurations."""

    @pytest.fixture
    def registry_with_realpdebench(self):
        """Create registry with RealPDEBench configs registered."""
        from opifex.benchmarking.pdebench_configs import (
            register_realpdebench_benchmarks,
        )

        registry = BenchmarkRegistry()
        register_realpdebench_benchmarks(registry)
        return registry

    def test_cylinder_config_exists(self, registry_with_realpdebench):
        """Cylinder vortex shedding config is defined."""
        config = registry_with_realpdebench.get_benchmark_config(
            "RealPDEBench_Cylinder"
        )

        assert config.domain == "fluid_dynamics"
        assert "reynolds_number" in config.physics_constraints


class TestBenchmarkConfigValidation:
    """Tests for BenchmarkConfig validation."""

    def test_config_requires_name(self):
        """BenchmarkConfig requires a name."""
        with pytest.raises(ValueError, match="name"):
            BenchmarkConfig(
                name="",  # Empty name
                domain="fluid_dynamics",
                problem_type="operator_learning",
                input_shape=(32, 32, 1),
                output_shape=(32, 32, 1),
            )

    def test_config_requires_domain(self):
        """BenchmarkConfig requires a domain."""
        with pytest.raises(ValueError, match="Domain"):
            BenchmarkConfig(
                name="test",
                domain="",  # Empty domain
                problem_type="operator_learning",
                input_shape=(32, 32, 1),
                output_shape=(32, 32, 1),
            )

    def test_config_requires_shapes(self):
        """BenchmarkConfig requires input and output shapes."""
        with pytest.raises(ValueError, match="shapes"):
            BenchmarkConfig(
                name="test",
                domain="fluid_dynamics",
                problem_type="operator_learning",
                input_shape=(),  # Empty shape
                output_shape=(32, 32, 1),
            )


class TestBenchmarkRegistryOperatorMetadata:
    """Tests for operator registration with metadata (DRY principle)."""

    def test_register_operator_with_type_metadata(self):
        """Operators can be registered with explicit type metadata."""
        from opifex.neural.operators.fno.tensorized import (
            TensorizedFourierNeuralOperator,
        )

        registry = BenchmarkRegistry()
        registry.register_operator(
            TensorizedFourierNeuralOperator,
            metadata={"operator_type": "fno", "supports_mixed_precision": True},
        )

        # Metadata should be accessible
        assert (
            registry._operator_metadata["TensorizedFourierNeuralOperator"][
                "operator_type"
            ]
            == "fno"
        )
        assert (
            registry._operator_metadata["TensorizedFourierNeuralOperator"][
                "supports_mixed_precision"
            ]
            is True
        )

    def test_operator_metadata_used_for_config(self):
        """Operator metadata should be used instead of name-matching heuristics."""
        from opifex.neural.operators.fno.tensorized import (
            TensorizedFourierNeuralOperator,
        )

        registry = BenchmarkRegistry()

        # Register with explicit type
        registry.register_operator(
            TensorizedFourierNeuralOperator,
            metadata={"operator_type": "fno"},
        )

        # Verify metadata is stored
        metadata = registry._operator_metadata.get(
            "TensorizedFourierNeuralOperator", {}
        )
        assert metadata.get("operator_type") == "fno"

        # This metadata should be used by BenchmarkRunner._get_operator_config()
        # instead of checking if "FNO" in class.__name__


class TestLoaderTypeInConfig:
    """Tests for explicit loader_type in computational_requirements (DRY principle)."""

    def test_loader_type_in_computational_requirements(self):
        """BenchmarkConfig should have explicit loader_type, not name-based inference."""
        config = BenchmarkConfig(
            name="SomeArbitraryName",  # Name should NOT determine loader
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(64, 64, 1),
            output_shape=(64, 64, 1),
            computational_requirements={
                "loader_type": "darcy",  # Explicit loader type
                "batch_size": 32,
            },
        )

        assert config.computational_requirements["loader_type"] == "darcy"

    def test_loader_type_decoupled_from_name(self):
        """Loader type should be independent of benchmark name."""
        # A benchmark named "MyCustomBenchmark" using Darcy data
        config = BenchmarkConfig(
            name="MyCustomBenchmark",
            domain="fluid_dynamics",
            problem_type="operator_learning",
            input_shape=(64, 64, 1),
            output_shape=(64, 64, 1),
            computational_requirements={
                "loader_type": "darcy",
            },
        )

        # Name doesn't contain "darcy", but loader_type is explicit
        assert "darcy" not in config.name.lower()
        assert config.computational_requirements["loader_type"] == "darcy"
