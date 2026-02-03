"""Tests for production optimization components.

This module tests the Phase 7.4 Production Optimization implementation
including the Hybrid Performance Platform, adaptive JIT optimization,
and intelligent GPU memory management.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.production import (
    AdaptiveJAXOptimizer,
    HybridPerformancePlatform,
    IntelligentGPUMemoryManager,
    OptimizationStrategy,
    OptimizedModel,
    PerformanceMetrics,
    WorkloadProfile,
)


def get_model_input_features(model) -> int:
    """Extract the correct input feature dimension from a model.

    Args:
        model: The model to inspect

    Returns:
        int: The input feature dimension
    """
    if hasattr(model, "linear") and hasattr(model.linear, "in_features"):
        return model.linear.in_features
    return 64  # Default fallback for SimpleModel


class SimpleModel(nnx.Module):
    """Simple test model for production optimization tests."""

    def __init__(self, features: int = 64, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    rngs = nnx.Rngs(0)
    return SimpleModel(features=64, rngs=rngs)


@pytest.fixture
def sample_workload():
    """Create a sample workload profile for testing."""
    return WorkloadProfile(
        batch_size=32,
        sequence_length=128,
        memory_footprint=2.0,  # 2GB
        compute_intensity=8.0,  # High compute
        latency_requirement=10.0,  # 10ms
        throughput_requirement=100.0,  # 100 RPS
        model_complexity="medium",
    )


@pytest.fixture
def low_latency_workload():
    """Create a low-latency workload profile for testing."""
    return WorkloadProfile(
        batch_size=16,
        sequence_length=64,
        memory_footprint=1.0,  # 1GB
        compute_intensity=5.0,
        latency_requirement=2.0,  # 2ms - triggers latency optimization
        throughput_requirement=500.0,
        model_complexity="simple",
    )


@pytest.fixture
def memory_intensive_workload():
    """Create a memory-intensive workload profile for testing."""
    return WorkloadProfile(
        batch_size=8,
        sequence_length=512,
        memory_footprint=12.0,  # 12GB - triggers memory optimization
        compute_intensity=3.0,
        latency_requirement=50.0,
        throughput_requirement=20.0,
        model_complexity="complex",
    )


@pytest.fixture
def compute_intensive_workload():
    """Create a compute-intensive workload profile for testing."""
    return WorkloadProfile(
        batch_size=64,
        sequence_length=256,
        memory_footprint=4.0,
        compute_intensity=15.0,  # Very high compute - triggers aggressive fusion
        latency_requirement=20.0,
        throughput_requirement=50.0,
        model_complexity="complex",
    )


class TestWorkloadProfile:
    """Test WorkloadProfile data structure."""

    def test_workload_profile_creation(self, sample_workload):
        """Test basic workload profile creation."""
        assert sample_workload.batch_size == 32
        assert sample_workload.sequence_length == 128
        assert sample_workload.memory_footprint == 2.0
        assert sample_workload.compute_intensity == 8.0
        assert sample_workload.latency_requirement == 10.0
        assert sample_workload.throughput_requirement == 100.0
        assert sample_workload.model_complexity == "medium"


class TestPerformanceMetrics:
    """Test PerformanceMetrics data structure."""

    def test_performance_metrics_creation(self):
        """Test basic performance metrics creation."""
        metrics = PerformanceMetrics(
            latency_ms=5.2,
            throughput_rps=192.3,
            memory_usage_gb=1.8,
            gpu_utilization=0.87,
            energy_efficiency=0.92,
            improvement_factor=1.35,
        )

        assert metrics.latency_ms == 5.2
        assert metrics.throughput_rps == 192.3
        assert metrics.memory_usage_gb == 1.8
        assert metrics.gpu_utilization == 0.87
        assert metrics.energy_efficiency == 0.92
        assert metrics.improvement_factor == 1.35


class TestAdaptiveJAXOptimizer:
    """Test AdaptiveJAXOptimizer component."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization with default parameters."""
        optimizer = AdaptiveJAXOptimizer()

        assert optimizer.performance_threshold == 1.1
        assert optimizer.memory_efficiency_target == 0.85
        assert optimizer.cache_size == 100
        assert len(optimizer.optimization_cache) == 0

    def test_optimizer_custom_initialization(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = AdaptiveJAXOptimizer(
            performance_threshold=1.5, memory_efficiency_target=0.9, cache_size=50
        )

        assert optimizer.performance_threshold == 1.5
        assert optimizer.memory_efficiency_target == 0.9
        assert optimizer.cache_size == 50

    def test_workload_pattern_analysis_aggressive_fusion(
        self, compute_intensive_workload
    ):
        """Test workload analysis for compute-intensive workloads."""
        optimizer = AdaptiveJAXOptimizer()
        strategy = optimizer.analyze_workload_patterns(compute_intensive_workload)

        assert strategy == OptimizationStrategy.AGGRESSIVE_FUSION

    def test_workload_pattern_analysis_memory_efficient(
        self, memory_intensive_workload
    ):
        """Test workload analysis for memory-intensive workloads."""
        optimizer = AdaptiveJAXOptimizer()
        strategy = optimizer.analyze_workload_patterns(memory_intensive_workload)

        assert strategy == OptimizationStrategy.MEMORY_EFFICIENT

    def test_workload_pattern_analysis_latency_optimized(self, low_latency_workload):
        """Test workload analysis for latency-critical workloads."""
        optimizer = AdaptiveJAXOptimizer()
        strategy = optimizer.analyze_workload_patterns(low_latency_workload)

        assert strategy == OptimizationStrategy.LATENCY_OPTIMIZED

    def test_workload_pattern_analysis_balanced(self, sample_workload):
        """Test workload analysis for balanced workloads."""
        optimizer = AdaptiveJAXOptimizer()
        strategy = optimizer.analyze_workload_patterns(sample_workload)

        assert strategy == OptimizationStrategy.BALANCED

    def test_aggressive_kernel_fusion(self, sample_model):
        """Test aggressive kernel fusion optimization."""
        optimizer = AdaptiveJAXOptimizer()
        optimized_model = optimizer.apply_aggressive_kernel_fusion(sample_model)

        # Test that the optimized model can perform inference
        test_input = jnp.ones((16, 64))
        output = optimized_model(test_input)  # type: ignore[operator]

        assert output.shape == (16, 64)
        assert jnp.isfinite(output).all()

    def test_memory_optimization(self, sample_model):
        """Test memory optimization."""
        optimizer = AdaptiveJAXOptimizer()
        optimized_model = optimizer.apply_memory_optimization(sample_model)

        # Test that the optimized model can perform inference
        test_input = jnp.ones((16, 64))
        output = optimized_model(test_input)  # type: ignore[operator]

        assert output.shape == (16, 64)
        assert jnp.isfinite(output).all()

    def test_latency_optimization(self, sample_model):
        """Test latency optimization."""
        optimizer = AdaptiveJAXOptimizer()
        optimized_model = optimizer.apply_latency_optimization(sample_model)

        # Test that the optimized model can perform inference
        test_input = jnp.ones((16, 64))
        output = optimized_model(test_input)  # type: ignore[operator]

        assert output.shape == (16, 64)
        assert jnp.isfinite(output).all()

    def test_balanced_optimization(self, sample_model):
        """Test balanced optimization."""
        optimizer = AdaptiveJAXOptimizer()
        optimized_model = optimizer.apply_balanced_optimization(sample_model)

        # Test that the optimized model can perform inference
        test_input = jnp.ones((16, 64))
        output = optimized_model(test_input)  # type: ignore[operator]

        assert output.shape == (16, 64)
        assert jnp.isfinite(output).all()

    def test_benchmark_model_performance(self, sample_model, sample_workload):
        """Test model performance benchmarking."""
        optimizer = AdaptiveJAXOptimizer()

        # Use a simpler workload for testing
        test_workload = WorkloadProfile(
            batch_size=8,
            sequence_length=64,
            memory_footprint=1.0,
            compute_intensity=5.0,
            latency_requirement=10.0,
            throughput_requirement=100.0,
            model_complexity="simple",
        )

        metrics = optimizer.benchmark_model_performance(sample_model, test_workload)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.throughput_rps >= 0
        assert metrics.memory_usage_gb > 0
        assert 0 <= metrics.gpu_utilization <= 1
        assert 0 <= metrics.energy_efficiency <= 1

    def test_optimize_neural_operator(self, sample_model, sample_workload):
        """Test complete neural operator optimization."""
        optimizer = AdaptiveJAXOptimizer()

        optimized_container = optimizer.optimize_neural_operator(
            sample_model, sample_workload
        )

        assert isinstance(optimized_container, OptimizedModel)
        assert optimized_container.optimization_type == OptimizationStrategy.BALANCED
        assert isinstance(optimized_container.performance_metrics, PerformanceMetrics)
        assert "workload_profile" in optimized_container.optimization_metadata
        assert "optimization_timestamp" in optimized_container.optimization_metadata
        assert "jax_backend" in optimized_container.optimization_metadata

        # Test that the optimized model works - use model's actual input dimension
        input_features = get_model_input_features(sample_model)
        test_input = jnp.ones((sample_workload.batch_size, input_features))
        output = optimized_container.model(test_input)  # type: ignore[operator]
        # Output shape should match input shape for this linear layer
        assert output.shape == (sample_workload.batch_size, input_features)

    def test_optimization_caching(self, sample_model, sample_workload):
        """Test optimization result caching."""
        optimizer = AdaptiveJAXOptimizer(cache_size=5)

        # First optimization
        result1 = optimizer.optimize_neural_operator(sample_model, sample_workload)
        assert (
            len(optimizer.optimization_cache) <= 1
        )  # May not cache if improvement is too small

        # Second optimization with same inputs should use cache if available
        result2 = optimizer.optimize_neural_operator(sample_model, sample_workload)

        # Results should be consistent
        assert result1.optimization_type == result2.optimization_type


class TestIntelligentGPUMemoryManager:
    """Test IntelligentGPUMemoryManager component."""

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        manager = IntelligentGPUMemoryManager()

        assert manager.fragmentation_threshold == 0.15
        assert manager.gc_trigger_threshold == 0.85
        assert "small" in manager.pool_sizes
        assert "medium" in manager.pool_sizes
        assert "large" in manager.pool_sizes
        assert "xlarge" in manager.pool_sizes

    def test_memory_manager_custom_initialization(self):
        """Test memory manager initialization with custom parameters."""
        custom_pools = {"tiny": (1, 16), "big": (16, 1024)}

        manager = IntelligentGPUMemoryManager(
            fragmentation_threshold=0.2,
            gc_trigger_threshold=0.9,
            pool_sizes=custom_pools,
        )

        assert manager.fragmentation_threshold == 0.2
        assert manager.gc_trigger_threshold == 0.9
        assert manager.pool_sizes == custom_pools

    def test_select_memory_pool(self):
        """Test memory pool selection logic."""
        manager = IntelligentGPUMemoryManager()

        assert manager.select_memory_pool(16) == "small"  # 16MB
        assert manager.select_memory_pool(128) == "medium"  # 128MB
        assert manager.select_memory_pool(2048) == "large"  # 2GB
        assert manager.select_memory_pool(16384) == "xlarge"  # 16GB
        assert manager.select_memory_pool(100000) == "xlarge"  # Oversized

    def test_estimate_model_memory_usage(self, sample_model):
        """Test model memory usage estimation."""
        manager = IntelligentGPUMemoryManager()

        memory_usage = manager.estimate_model_memory_usage(sample_model, batch_size=32)

        assert memory_usage > 0
        assert isinstance(memory_usage, float)

    def test_optimize_multi_model_allocation(self, sample_model):
        """Test multi-model memory allocation optimization."""
        manager = IntelligentGPUMemoryManager()

        # Create multiple models with different batch sizes
        rngs = nnx.Rngs(1)
        model2 = SimpleModel(features=32, rngs=rngs)

        models = [(sample_model, 16), (model2, 8)]

        allocation_plan = manager.optimize_multi_model_allocation(models)

        assert "model_allocations" in allocation_plan
        assert "shared_regions" in allocation_plan
        assert "total_memory_mb" in allocation_plan
        assert "efficiency_score" in allocation_plan

        assert "model_0" in allocation_plan["model_allocations"]
        assert "model_1" in allocation_plan["model_allocations"]

        assert allocation_plan["total_memory_mb"] > 0
        assert 0 <= allocation_plan["efficiency_score"] <= 1


class TestHybridPerformancePlatform:
    """Test HybridPerformancePlatform integration."""

    def test_platform_initialization(self):
        """Test platform initialization with default components."""
        rngs = nnx.Rngs(0)
        platform = HybridPerformancePlatform(rngs=rngs)

        assert isinstance(platform.jit_optimizer, AdaptiveJAXOptimizer)
        assert isinstance(platform.memory_manager, IntelligentGPUMemoryManager)

    def test_platform_custom_initialization(self):
        """Test platform initialization with custom components."""
        custom_optimizer = AdaptiveJAXOptimizer(performance_threshold=1.5)
        custom_manager = IntelligentGPUMemoryManager(fragmentation_threshold=0.2)

        rngs = nnx.Rngs(0)
        platform = HybridPerformancePlatform(
            jit_optimizer=custom_optimizer, memory_manager=custom_manager, rngs=rngs
        )

        assert platform.jit_optimizer is custom_optimizer
        assert platform.memory_manager is custom_manager

    def test_optimize_for_production(self, sample_model, sample_workload):
        """Test comprehensive production optimization."""
        rngs = nnx.Rngs(0)
        platform = HybridPerformancePlatform(rngs=rngs)

        optimized_model = platform.optimize_for_production(
            sample_model, sample_workload
        )

        assert isinstance(optimized_model, OptimizedModel)
        assert "memory_plan" in optimized_model.optimization_metadata
        assert optimized_model.optimization_metadata["platform_optimization"] is True
        assert optimized_model.optimization_metadata["production_ready"] is True

        # Test that the optimized model works - use model's actual input dimension
        input_features = get_model_input_features(sample_model)
        test_input = jnp.ones((sample_workload.batch_size, input_features))
        output = optimized_model.model(test_input)  # type: ignore[operator]
        # Output shape should match input shape for this linear layer
        assert output.shape == (sample_workload.batch_size, input_features)


class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""

    def test_optimization_strategy_values(self):
        """Test optimization strategy enum values."""
        assert OptimizationStrategy.AGGRESSIVE_FUSION.value == "aggressive_fusion"
        assert OptimizationStrategy.MEMORY_EFFICIENT.value == "memory_efficient"
        assert OptimizationStrategy.LATENCY_OPTIMIZED.value == "latency_optimized"
        assert OptimizationStrategy.BALANCED.value == "balanced"


class TestProductionOptimizationIntegration:
    """Integration tests for production optimization components."""

    def test_end_to_end_optimization(self, sample_model):
        """Test end-to-end optimization workflow."""
        # Create different workload scenarios
        workloads = [
            WorkloadProfile(32, 128, 2.0, 8.0, 10.0, 100.0, "medium"),
            WorkloadProfile(
                16, 64, 1.0, 15.0, 5.0, 200.0, "simple"
            ),  # Compute intensive
            WorkloadProfile(
                8, 512, 12.0, 3.0, 50.0, 20.0, "complex"
            ),  # Memory intensive
            WorkloadProfile(64, 32, 0.5, 5.0, 2.0, 500.0, "simple"),  # Latency critical
        ]

        rngs = nnx.Rngs(0)
        platform = HybridPerformancePlatform(rngs=rngs)
        input_features = get_model_input_features(sample_model)

        for workload in workloads:
            optimized_model = platform.optimize_for_production(sample_model, workload)

            # Verify optimization was successful
            assert isinstance(optimized_model, OptimizedModel)
            assert optimized_model.optimization_metadata["production_ready"] is True

            # Verify model functionality - use model's actual input dimension
            test_input = jnp.ones((workload.batch_size, input_features))
            output = optimized_model.model(test_input)  # type: ignore[operator]
            # Output shape should match input shape for this linear layer
            assert output.shape == (workload.batch_size, input_features)
            assert jnp.isfinite(output).all()

    def test_optimization_strategy_selection(self):
        """Test that different workloads select appropriate optimization strategies."""
        optimizer = AdaptiveJAXOptimizer()

        # Test cases: (workload, expected_strategy)
        test_cases = [
            (
                WorkloadProfile(32, 128, 2.0, 15.0, 10.0, 100.0, "medium"),
                OptimizationStrategy.AGGRESSIVE_FUSION,
            ),
            (
                WorkloadProfile(8, 512, 12.0, 3.0, 50.0, 20.0, "complex"),
                OptimizationStrategy.MEMORY_EFFICIENT,
            ),
            (
                WorkloadProfile(64, 32, 0.5, 5.0, 2.0, 500.0, "simple"),
                OptimizationStrategy.LATENCY_OPTIMIZED,
            ),
            (
                WorkloadProfile(32, 128, 4.0, 8.0, 10.0, 100.0, "medium"),
                OptimizationStrategy.BALANCED,
            ),
        ]

        for workload, expected_strategy in test_cases:
            actual_strategy = optimizer.analyze_workload_patterns(workload)
            assert actual_strategy == expected_strategy, (
                f"Expected {expected_strategy}, got {actual_strategy} for workload {workload}"
            )
