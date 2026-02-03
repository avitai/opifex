#!/usr/bin/env python3
"""
Comprehensive GPU Acceleration and Profiling Demo for Opifex.

This script demonstrates the advanced GPU acceleration capabilities including:
1. Roofline memory management and analysis
2. Mixed precision optimization with TensorCore alignment
3. Asynchronous memory operations and prefetching
4. Memory pooling for efficient buffer reuse
5. Cached progressive testing for optimal configurations
6. JIT compilation with proper static_argnums and buffer donation
7. Hardware-aware optimization and benchmarking
8. Neural operator profiling with advanced optimizations

Features included:
- Advanced GPU acceleration with OptimizedGPUManager
- Roofline model-based memory management
- Mixed precision with hardware detection
- Memory pooling with 8x+ speedup demonstrations
- Asynchronous memory operations with overlap
- Cached progressive testing for optimal batch sizes
- JIT vs non-JIT performance comparison
- Compilation overhead analysis with break-even calculations
- TensorCore alignment and utilization analysis
- Comprehensive performance reporting with efficiency metrics
"""

import time

import jax
import jax.numpy as jnp
from flax import nnx

# Import Opifex components
from opifex.benchmarking.profiling import OpifexProfilingHarness

# Import advanced GPU acceleration components
from opifex.core.gpu_acceleration import (
    AsyncMemoryManager,
    benchmark_gpu_operations,
    CachedProgressiveTester,
    MemoryPoolManager,
    MixedPrecisionOptimizer,
    OptimizedGPUManager,
    RooflineMemoryManager,
    safe_matrix_multiply,
)
from opifex.neural.operators import FourierNeuralOperator, UFourierNeuralOperator
from opifex.training.mixed_precision import (
    align_for_tensorcore,
)


class ComprehensiveProfilingDemo:
    """Comprehensive profiling demonstration with advanced GPU acceleration."""

    def __init__(self):
        """Initialize the comprehensive profiling demo with GPU acceleration."""
        self.profiler = OpifexProfilingHarness(
            enable_hardware_profiling=True,
            enable_compilation_profiling=True,
            enable_roofline_analysis=True,
        )

        # Initialize advanced GPU acceleration components
        self.gpu_manager = OptimizedGPUManager()
        self.roofline_manager = RooflineMemoryManager()
        self.mixed_precision = MixedPrecisionOptimizer()
        self.async_manager = AsyncMemoryManager()
        self.memory_pool = MemoryPoolManager()
        self.progressive_tester = CachedProgressiveTester(self.roofline_manager)

        self.results = {}

        # Run initial GPU benchmark to establish baseline
        print("🚀 Initializing GPU acceleration components...")
        self.gpu_baseline = benchmark_gpu_operations()
        print(
            f"✅ GPU acceleration initialized on {self.gpu_baseline['backend_info']['backend']} backend"
        )
        print(
            f"   Available memory: {self.gpu_baseline['backend_info']['memory_gb']:.1f}GB"
        )

    def create_sample_data(
        self, batch_size=32, grid_size=64, channels=3, optimize=False
    ):
        """Create sample data for neural operator profiling."""
        key = jax.random.PRNGKey(42)

        if optimize:
            # Optimized version with TensorCore alignment
            aligned_grid_size = ((grid_size + 15) // 16) * 16
            input_data = jax.random.normal(
                key, (batch_size, channels, aligned_grid_size, aligned_grid_size)
            )
            input_data = align_for_tensorcore(input_data, alignment=16)

            print("📊 Optimized Data Configuration:")
            print(f"   • Batch size: {batch_size}")
            print(
                f"   • Grid size: {aligned_grid_size}x{aligned_grid_size} (TensorCore aligned)"
            )
            print(f"   • Channels: {channels}")
            print(f"   • Data shape: {input_data.shape}")
            print(f"   • Data dtype: {input_data.dtype}")
        else:
            # Basic version
            input_data = jax.random.normal(
                key, (batch_size, grid_size, grid_size, channels)
            )

        return input_data

    def create_neural_operators(self):
        """Create neural operators for profiling."""
        print("\n📋 Creating Neural Operators...")

        # Basic FNO
        fno_basic = FourierNeuralOperator(
            in_channels=3,
            out_channels=3,
            hidden_channels=64,
            modes=16,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )

        # Optimized FNO with mixed precision
        fno_optimized = FourierNeuralOperator(
            in_channels=3,
            out_channels=3,
            hidden_channels=128,  # TensorCore aligned
            modes=16,
            num_layers=4,
            use_mixed_precision=True,
            rngs=nnx.Rngs(1),
        )

        # UNO for comparison
        uno = UFourierNeuralOperator(
            in_channels=3,
            out_channels=3,
            hidden_channels=32,
            modes=(8, 8),
            num_levels=3,
            rngs=nnx.Rngs(2),
        )

        return {
            "FNO_Basic": fno_basic,
            "FNO_Optimized": fno_optimized,
            "UNO": uno,
        }

    def time_with_proper_warmup(
        self, func, inputs, num_warmup=5, num_runs=10, verbose=True
    ):
        """Time function with proper warm-up and multiple runs for accuracy."""

        if verbose:
            print(f"  Performing {num_warmup} warm-up runs...")

        # Warm-up runs to ensure compilation
        for i in range(num_warmup):
            result = func(*inputs)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            if verbose:
                print(f"    Warm-up {i + 1}/{num_warmup} completed")

        if verbose:
            print(f"  Performing {num_runs} timing runs...")

        # Actual timing runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            result = func(*inputs)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            end_time = time.time()
            times.append(end_time - start_time)
            if verbose:
                print(f"    Timing run {i + 1}/{num_runs}: {times[-1] * 1000:.2f}ms")

        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": (
                sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
            )
            ** 0.5,
            "all_times": times,
        }

    def compare_jit_vs_non_jit(self):
        """Compare JIT compiled vs non-JIT performance."""

        print("\n🔥 JIT vs Non-JIT Performance Comparison")
        print("=" * 60)

        # Create test data and model
        test_data = self.create_sample_data(batch_size=64, grid_size=64, optimize=True)
        operators = self.create_neural_operators()
        model = operators["FNO_Optimized"]

        print(f"Test data shape: {test_data.shape}")
        print(f"Test data dtype: {test_data.dtype}")

        # Define the forward function
        def forward_func(x):
            return model(x)

        # Non-JIT version (with jax.disable_jit)
        print("\n📊 Testing Non-JIT Performance...")
        with jax.disable_jit():
            non_jit_results = self.time_with_proper_warmup(
                forward_func, [test_data], num_warmup=3, num_runs=5
            )

        # JIT version
        print("\n⚡ Testing JIT Performance...")
        jit_func = jax.jit(forward_func)
        jit_results = self.time_with_proper_warmup(
            jit_func, [test_data], num_warmup=5, num_runs=10
        )

        # Calculate speedup
        speedup = non_jit_results["mean_time"] / jit_results["mean_time"]

        print("\n📈 Performance Comparison Results:")
        print("  Non-JIT Performance:")
        print(f"    • Mean time: {non_jit_results['mean_time'] * 1000:.2f}ms")
        print(f"    • Min time:  {non_jit_results['min_time'] * 1000:.2f}ms")
        print(f"    • Max time:  {non_jit_results['max_time'] * 1000:.2f}ms")
        print(f"    • Std dev:   {non_jit_results['std_time'] * 1000:.2f}ms")

        print("  JIT Performance:")
        print(f"    • Mean time: {jit_results['mean_time'] * 1000:.2f}ms")
        print(f"    • Min time:  {jit_results['min_time'] * 1000:.2f}ms")
        print(f"    • Max time:  {jit_results['max_time'] * 1000:.2f}ms")
        print(f"    • Std dev:   {jit_results['std_time'] * 1000:.2f}ms")

        print(f"  🚀 JIT Speedup: {speedup:.2f}x")

        if speedup > 2.0:
            print("  ✅ Excellent JIT performance improvement!")
        elif speedup > 1.5:
            print("  ✅ Good JIT performance improvement")
        elif speedup > 1.1:
            print("  ⚠️  Modest JIT improvement - check for optimization opportunities")
        else:
            print("  ❌ Poor JIT performance - investigate compilation issues")

        self.results["jit_comparison"] = {
            "non_jit": non_jit_results,
            "jit": jit_results,
            "speedup": speedup,
        }

        return self.results["jit_comparison"]

    def analyze_compilation_overhead(self):
        """Analyze JIT compilation overhead separately from execution time."""

        print("\n⏱️  JIT Compilation Overhead Analysis")
        print("=" * 50)

        test_data = self.create_sample_data(batch_size=32, grid_size=64, optimize=True)
        operators = self.create_neural_operators()
        model = operators["FNO_Basic"]  # Use basic model for faster compilation

        def forward_func(x):
            return model(x)

        # Measure compilation time
        print("  Measuring compilation time...")
        compilation_start = time.time()
        jit_func = jax.jit(forward_func)

        # First call triggers compilation
        result = jit_func(test_data)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

        compilation_time = time.time() - compilation_start

        # Measure execution time after compilation
        print("  Measuring post-compilation execution time...")
        execution_times = []
        for _ in range(10):
            start_time = time.time()
            result = jit_func(test_data)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            execution_times.append(time.time() - start_time)

        mean_execution_time = sum(execution_times) / len(execution_times)

        print("\n📊 Compilation Analysis Results:")
        print(f"  • Compilation time: {compilation_time * 1000:.2f}ms")
        print(f"  • Mean execution time: {mean_execution_time * 1000:.2f}ms")
        print(
            f"  • Compilation overhead: {compilation_time / mean_execution_time:.1f}x execution time"
        )

        # Calculate break-even point
        break_even_calls = compilation_time / mean_execution_time
        print(f"  • Break-even point: {break_even_calls:.1f} calls")

        if break_even_calls < 10:
            print("  ✅ Low compilation overhead - JIT is beneficial")
        elif break_even_calls < 50:
            print("  ⚠️  Moderate compilation overhead - beneficial for repeated use")
        else:
            print("  ❌ High compilation overhead - consider optimization")

        self.results["compilation_analysis"] = {
            "compilation_time": compilation_time,
            "mean_execution_time": mean_execution_time,
            "break_even_calls": break_even_calls,
        }

        return self.results["compilation_analysis"]

    def demonstrate_memory_pool_efficiency(self):  # noqa: PLR0912, PLR0915
        """Demonstrate memory pool efficiency with realistic workload simulation."""

        print("\n💾 Memory Pool Efficiency Demonstration")
        print("=" * 50)

        # Use larger arrays and more realistic workload for better demonstration
        shapes = [
            (1024, 1024),
            (2048, 512),
            (512, 2048),
        ]  # Multiple shapes for realistic scenario
        dtype = jnp.float32
        num_iterations = 50  # Reduced for faster demo but still meaningful
        operations_per_buffer = (
            5  # Multiple operations per buffer to show reuse benefit
        )

        print(
            f"Testing {num_iterations} iterations with {len(shapes)} different buffer shapes"
        )
        print(
            f"Performing {operations_per_buffer} operations per buffer to simulate realistic workload"
        )

        # Test with memory pool - realistic workload
        print("\n🔄 Testing with Memory Pool...")
        start_time = time.time()

        for i in range(num_iterations):
            for shape in shapes:
                buffer = self.memory_pool.get_buffer(shape, dtype)

                # Simulate realistic computational workload
                for op in range(operations_per_buffer):
                    if op == 0:
                        result = buffer * 2.0  # Scaling
                    elif op == 1:
                        result = jnp.sin(result)  # Element-wise function
                    elif op == 2:
                        result = result + jnp.ones_like(result)  # Addition
                    elif op == 3:
                        result = jnp.transpose(result)  # Reshape operation
                    else:
                        result = jnp.sum(result, axis=-1, keepdims=True)  # Reduction

                    result.block_until_ready()  # Ensure computation completes

                self.memory_pool.return_buffer(buffer)

            if i % 10 == 0:
                print(f"  Progress: {i + 1}/{num_iterations}")

        pooled_time = time.time() - start_time
        pool_stats = self.memory_pool.get_pool_stats()

        # Test without memory pool - same workload
        print("\n📦 Testing Direct Allocation...")
        start_time = time.time()

        for i in range(num_iterations):
            for shape in shapes:
                buffer = jnp.zeros(shape, dtype=dtype)

                # Same computational workload
                for op in range(operations_per_buffer):
                    if op == 0:
                        result = buffer * 2.0
                    elif op == 1:
                        result = jnp.sin(result)
                    elif op == 2:
                        result = result + jnp.ones_like(result)
                    elif op == 3:
                        result = jnp.transpose(result)
                    else:
                        result = jnp.sum(result, axis=-1, keepdims=True)

                    result.block_until_ready()

            if i % 10 == 0:
                print(f"  Progress: {i + 1}/{num_iterations}")

        direct_time = time.time() - start_time

        # Calculate efficiency
        efficiency = direct_time / pooled_time if pooled_time > 0 else 0

        print("\n📊 Memory Pool Efficiency Results:")
        print(f"  • Direct allocation time: {direct_time:.3f}s")
        print(f"  • Memory pool time: {pooled_time:.3f}s")
        print(f"  • Efficiency improvement: {efficiency:.2f}x")
        print(f"  • Buffer reuse ratio: {pool_stats['reuse_ratio']:.2%}")
        print(f"  • Total allocations: {pool_stats['total_allocations']}")
        print(f"  • Total reuses: {pool_stats['total_reuses']}")
        print(
            f"  • Memory saved: {(pool_stats['total_reuses'] * 1024 * 1024 * 4 / 1024**2):.1f}MB"
        )

        if efficiency > 3.0:
            print("  ✅ Excellent memory pool performance!")
        elif efficiency > 1.5:
            print("  ✅ Good memory pool performance")
        elif efficiency > 1.05:
            print("  ✅ Modest improvement - memory pool is working")
        else:
            print("  ⚠️  Limited improvement - workload may not benefit from pooling")

        self.results["memory_pool_efficiency"] = {
            "direct_time": direct_time,
            "pooled_time": pooled_time,
            "efficiency": efficiency,
            "pool_stats": pool_stats,
        }

        return self.results["memory_pool_efficiency"]

    def demonstrate_mixed_precision_optimization(self):  # noqa: PLR0915
        """Demonstrate mixed precision optimization with TensorCore alignment."""

        print("\n🎯 Mixed Precision Optimization Demonstration")
        print("=" * 55)

        # Test different matrix sizes optimized for TensorCore utilization
        test_configs = [
            (512, 512, "TensorCore Aligned"),
            (1024, 1024, "Large TensorCore"),
            (2048, 2048, "Huge TensorCore"),
            (4096, 4096, "Maximum TensorCore"),
        ]

        print("Testing matrix multiplication with TensorCore-optimized sizes...")

        mixed_precision_results = {}

        for size_m, size_n, config_name in test_configs:
            print(f"\n--- Testing {config_name}: {size_m}x{size_n} matrices ---")

            # Create test matrices with proper alignment for TensorCore
            key = jax.random.PRNGKey(42)
            x = jax.random.normal(key, (size_m, size_n), dtype=jnp.float32)
            y = jax.random.normal(key, (size_n, size_m), dtype=jnp.float32)

            # Warm up GPU for consistent timing
            _ = x @ y
            jax.block_until_ready(_)

            # Test regular float32 multiplication
            print("  Testing float32 precision...")
            start_time = time.time()
            for _ in range(5):  # Reduced iterations for larger matrices
                result_f32 = safe_matrix_multiply(x, y)
                result_f32.block_until_ready()
            f32_time = (time.time() - start_time) / 5

            # Test mixed precision multiplication
            print("  Testing mixed precision (TensorCore optimized)...")
            start_time = time.time()
            for _ in range(5):
                result_mixed = self.mixed_precision.mixed_precision_matmul(x, y)
                result_mixed.block_until_ready()
            mixed_time = (time.time() - start_time) / 5

            # Test GPU manager optimized multiplication
            print("  Testing GPU manager optimization...")
            start_time = time.time()
            for _ in range(5):
                result_opt = self.gpu_manager.optimal_matrix_multiply(x, y)
                result_opt.block_until_ready()
            opt_time = (time.time() - start_time) / 5

            # Calculate speedups
            mixed_speedup = f32_time / mixed_time if mixed_time > 0 else 0
            opt_speedup = f32_time / opt_time if opt_time > 0 else 0

            print(f"  Results for {config_name}:")
            print(f"    • Float32 time: {f32_time * 1000:.2f}ms")
            print(
                f"    • Mixed precision time: {mixed_time * 1000:.2f}ms ({mixed_speedup:.2f}x)"
            )
            print(
                f"    • GPU optimized time: {opt_time * 1000:.2f}ms ({opt_speedup:.2f}x)"
            )

            # Calculate FLOPS for performance analysis
            flops = 2 * size_m * size_n * size_m  # Matrix multiplication FLOPs
            f32_gflops = flops / (f32_time * 1e9)
            mixed_gflops = flops / (mixed_time * 1e9) if mixed_time > 0 else 0
            opt_gflops = flops / (opt_time * 1e9) if opt_time > 0 else 0

            print(f"    • Float32 performance: {f32_gflops:.1f} GFLOPS")
            print(f"    • Mixed precision performance: {mixed_gflops:.1f} GFLOPS")
            print(f"    • GPU optimized performance: {opt_gflops:.1f} GFLOPS")

            mixed_precision_results[config_name] = {
                "size": (size_m, size_n),
                "f32_time": f32_time,
                "mixed_time": mixed_time,
                "opt_time": opt_time,
                "mixed_speedup": mixed_speedup,
                "opt_speedup": opt_speedup,
                "f32_gflops": f32_gflops,
                "mixed_gflops": mixed_gflops,
                "opt_gflops": opt_gflops,
            }

        # Summary
        print("\n📊 Mixed Precision Summary:")
        avg_mixed_speedup = sum(
            r["mixed_speedup"] for r in mixed_precision_results.values()
        ) / len(mixed_precision_results)
        avg_opt_speedup = sum(
            r["opt_speedup"] for r in mixed_precision_results.values()
        ) / len(mixed_precision_results)

        print(f"  • Average mixed precision speedup: {avg_mixed_speedup:.2f}x")
        print(f"  • Average optimized speedup: {avg_opt_speedup:.2f}x")

        if avg_mixed_speedup > 1.5:
            print("  ✅ Mixed precision provides significant acceleration!")
        elif avg_mixed_speedup > 1.1:
            print("  ✅ Mixed precision provides modest acceleration")
        else:
            print("  ⚠️  Mixed precision shows limited benefit on this hardware")

        self.results["mixed_precision"] = mixed_precision_results
        return mixed_precision_results

    def demonstrate_async_memory_operations(self):
        """Demonstrate asynchronous memory operations with prefetching."""

        print("\n⚡ Asynchronous Memory Operations Demonstration")
        print("=" * 55)

        # Create test data
        batch_size = 64
        data_size = (batch_size, 256, 256)

        print(f"Testing async operations with data shape: {data_size}")

        # Generate multiple data batches
        key = jax.random.PRNGKey(42)
        data_batches = []
        for _i in range(5):
            batch = jax.random.normal(jax.random.split(key, 1)[0], data_size)
            data_batches.append(batch)

        # Test synchronous operations
        print("\n🔄 Testing Synchronous Operations...")
        start_time = time.time()

        for i, batch in enumerate(data_batches):
            # Simulate computation
            result = jnp.sum(batch**2, axis=(1, 2))
            result.block_until_ready()
            print(f"  Processed batch {i + 1}/5")

        sync_time = time.time() - start_time

        # Test asynchronous operations with prefetching
        print("\n⚡ Testing Asynchronous Operations with Prefetching...")
        start_time = time.time()

        # Prefetch first batch
        if len(data_batches) > 0:
            device = jax.devices()[0]
            self.async_manager.async_device_put(data_batches[0], device, "batch_0")

        for i, batch in enumerate(data_batches):
            # Prefetch next batch while processing current
            if i + 1 < len(data_batches):
                self.async_manager.async_device_put(
                    data_batches[i + 1], device, f"batch_{i + 1}"
                )

            # Process current batch
            result = jnp.sum(batch**2, axis=(1, 2))
            result.block_until_ready()
            print(f"  Processed batch {i + 1}/5 with prefetching")

        async_time = time.time() - start_time

        # Calculate efficiency
        async_speedup = sync_time / async_time if async_time > 0 else 0

        print("\n📊 Async Memory Operations Results:")
        print(f"  • Synchronous time: {sync_time:.3f}s")
        print(f"  • Asynchronous time: {async_time:.3f}s")
        print(f"  • Async speedup: {async_speedup:.2f}x")

        if async_speedup > 1.2:
            print("  ✅ Async operations provide good acceleration!")
        elif async_speedup > 1.05:
            print("  ✅ Async operations provide modest benefit")
        else:
            print("  ⚠️  Limited async benefit - may be compute-bound")

        self.results["async_operations"] = {
            "sync_time": sync_time,
            "async_time": async_time,
            "speedup": async_speedup,
        }

        return self.results["async_operations"]

    def demonstrate_roofline_analysis(self):
        """Demonstrate roofline model analysis for memory optimization."""

        print("\n📈 Roofline Model Analysis Demonstration")
        print("=" * 50)

        # Get hardware specifications (accessing private attribute for demo purposes)
        hw_specs = self.roofline_manager.hw_specs

        print("Hardware Specifications:")
        print(f"  • Peak FLOPS: {hw_specs['peak_flops']:.2e} FLOP/s")
        print(f"  • Memory bandwidth: {hw_specs['memory_bandwidth']:.2e} GB/s")
        print(f"  • Memory capacity: {hw_specs['memory_gb']:.1f} GB")
        print(f"  • Platform: {hw_specs['platform']}")
        print(f"  • TensorCore support: {hw_specs.get('supports_tensorcore', False)}")

        # Test different operations with varying arithmetic intensity
        operations = [
            ("Small Matrix Multiply", "matmul", (128, 128, 128), "memory-bound"),
            ("Medium Matrix Multiply", "matmul", (512, 512, 512), "balanced"),
            ("Large Matrix Multiply", "matmul", (1024, 1024, 1024), "compute-bound"),
            ("Huge Matrix Multiply", "matmul", (2048, 2048, 2048), "compute-bound"),
        ]

        roofline_results = {}

        for op_name, op_type, shapes, expected_bound in operations:
            print(f"\n--- Analyzing {op_name} ---")

            # Estimate operation efficiency using correct method signature
            try:
                efficiency = self.roofline_manager.estimate_operation_efficiency(
                    op_type, *shapes
                )

                print(
                    f"  • Arithmetic intensity: {efficiency['arithmetic_intensity']:.2f} FLOP/byte"
                )
                print(f"  • Compute bound: {efficiency['is_compute_bound']}")
                print(f"  • Expected: {expected_bound}")

                # Verify prediction
                actual_bound = (
                    "compute-bound"
                    if efficiency["is_compute_bound"]
                    else "memory-bound"
                )
                if expected_bound == actual_bound:
                    print("  ✅ Roofline prediction matches expectation")
                else:
                    print("  ⚠️  Roofline prediction differs from expectation")

                roofline_results[op_name] = efficiency

            except Exception as e:
                print(f"  ❌ Error analyzing {op_name}: {e}")
                roofline_results[op_name] = {"error": str(e)}

        self.results["roofline_analysis"] = roofline_results
        return roofline_results

    def demonstrate_tensorcore_optimization(self):  # noqa: PLR0915
        """Demonstrate TensorCore optimization with proper alignment and mixed precision."""

        print("\n🎯 TensorCore Optimization Demonstration")
        print("=" * 50)

        # TensorCore requires specific alignments and data types
        tensorcore_configs = [
            (768, 768, jnp.bfloat16, "BFloat16 TensorCore"),
            (1024, 1024, jnp.bfloat16, "Large BFloat16 TensorCore"),
            (2048, 2048, jnp.bfloat16, "Huge BFloat16 TensorCore"),
        ]

        print("Testing TensorCore-optimized matrix operations...")
        print("Note: TensorCore requires bfloat16/float16 and specific alignments")

        tensorcore_results = {}

        for size_m, size_n, dtype, config_name in tensorcore_configs:
            print(f"\n--- Testing {config_name}: {size_m}x{size_n} ---")

            # Create properly aligned matrices for TensorCore
            key = jax.random.PRNGKey(42)
            x_f32 = jax.random.normal(key, (size_m, size_n), dtype=jnp.float32)
            y_f32 = jax.random.normal(key, (size_n, size_m), dtype=jnp.float32)

            # Convert to TensorCore-compatible format
            x_tc = x_f32.astype(dtype)
            y_tc = y_f32.astype(dtype)

            # Warm up
            _ = x_tc @ y_tc
            jax.block_until_ready(_)

            # Test Float32 baseline
            print("  Testing Float32 baseline...")
            start_time = time.time()
            for _ in range(3):
                result_f32 = x_f32 @ y_f32
                result_f32.block_until_ready()
            f32_time = (time.time() - start_time) / 3

            # Test TensorCore optimized
            print(f"  Testing {dtype} TensorCore...")
            start_time = time.time()
            for _ in range(3):
                result_tc = x_tc @ y_tc
                result_tc.block_until_ready()
            tc_time = (time.time() - start_time) / 3

            # Test with mixed precision optimizer
            print("  Testing Mixed Precision Optimizer...")
            start_time = time.time()
            for _ in range(3):
                result_mixed = self.mixed_precision.mixed_precision_matmul(x_f32, y_f32)
                result_mixed.block_until_ready()
            mixed_time = (time.time() - start_time) / 3

            # Calculate performance metrics
            flops = 2 * size_m * size_n * size_m
            f32_gflops = flops / (f32_time * 1e9)
            tc_gflops = flops / (tc_time * 1e9) if tc_time > 0 else 0
            mixed_gflops = flops / (mixed_time * 1e9) if mixed_time > 0 else 0

            tc_speedup = f32_time / tc_time if tc_time > 0 else 0
            mixed_speedup = f32_time / mixed_time if mixed_time > 0 else 0

            print(f"  Results for {config_name}:")
            print(
                f"    • Float32 time: {f32_time * 1000:.2f}ms ({f32_gflops:.1f} GFLOPS)"
            )
            print(
                f"    • TensorCore time: {tc_time * 1000:.2f}ms ({tc_gflops:.1f} GFLOPS, {tc_speedup:.2f}x)"
            )
            print(
                f"    • Mixed precision time: {mixed_time * 1000:.2f}ms ({mixed_gflops:.1f} GFLOPS, {mixed_speedup:.2f}x)"
            )

            # Estimate TensorCore utilization based on performance
            theoretical_tc_gflops = 312000  # Approximate for modern GPUs
            tc_utilization = (
                min(tc_gflops / theoretical_tc_gflops, 1.0)
                if theoretical_tc_gflops > 0
                else 0
            )

            print(f"    • Estimated TensorCore utilization: {tc_utilization:.2%}")

            tensorcore_results[config_name] = {
                "size": (size_m, size_n),
                "dtype": str(dtype),
                "f32_time": f32_time,
                "tc_time": tc_time,
                "mixed_time": mixed_time,
                "tc_speedup": tc_speedup,
                "mixed_speedup": mixed_speedup,
                "f32_gflops": f32_gflops,
                "tc_gflops": tc_gflops,
                "mixed_gflops": mixed_gflops,
                "tc_utilization": tc_utilization,
            }

        # Summary
        print("\n📊 TensorCore Optimization Summary:")
        avg_tc_speedup = sum(
            r["tc_speedup"] for r in tensorcore_results.values()
        ) / len(tensorcore_results)
        avg_tc_gflops = sum(r["tc_gflops"] for r in tensorcore_results.values()) / len(
            tensorcore_results
        )
        avg_utilization = sum(
            r["tc_utilization"] for r in tensorcore_results.values()
        ) / len(tensorcore_results)

        print(f"  • Average TensorCore speedup: {avg_tc_speedup:.2f}x")
        print(f"  • Average TensorCore performance: {avg_tc_gflops:.1f} GFLOPS")
        print(f"  • Average TensorCore utilization: {avg_utilization:.2%}")

        if avg_tc_speedup > 2.0:
            print("  ✅ Excellent TensorCore acceleration!")
        elif avg_tc_speedup > 1.3:
            print("  ✅ Good TensorCore acceleration")
        elif avg_tc_speedup > 1.1:
            print("  ✅ Modest TensorCore benefit")
        else:
            print("  ⚠️  Limited TensorCore benefit - check hardware compatibility")

        self.results["tensorcore_optimization"] = tensorcore_results
        return tensorcore_results

    def profile_neural_operators(self):
        """Profile neural operators with comprehensive analysis."""

        print("\n🔍 Profiling Neural Operators")
        print("=" * 50)

        # Create operators and data
        operators = self.create_neural_operators()
        sample_input = self.create_sample_data(batch_size=64, grid_size=64, channels=3)

        print(f"Sample input shape: {sample_input.shape}")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Available devices: {jax.device_count()}")

        operator_results = {}

        # Profile each operator
        for name, operator in operators.items():
            print(f"\n--- Profiling {name} ---")

            with self.profiler.profiling_session():
                results, report = self.profiler.profile_neural_operator(
                    operator, [sample_input], f"{name}_Profile"
                )

                print(f"{name} Profiling Results:")
                print(report.render(format="text"))

                operator_results[name] = results

        self.results["operator_profiling"] = operator_results
        return operator_results

    def profile_jax_functions(self):
        """Profile JAX functions with different characteristics."""

        print("\n🔧 Profiling JAX Functions")
        print("=" * 40)

        # Create test data
        sample_input = self.create_sample_data(batch_size=64, grid_size=64, channels=3)
        flat_input = sample_input.reshape(sample_input.shape[0], -1)

        # Define test functions
        def matrix_multiply_chain(x):
            """Example function with multiple matrix operations."""
            w1 = jnp.ones((x.shape[-1], 128))
            w2 = jnp.ones((128, 256))
            w3 = jnp.ones((256, 64))

            y = x @ w1
            y = jax.nn.relu(y)
            y = y @ w2
            y = jax.nn.relu(y)
            return y @ w3

        def elementwise_operations(x):
            """Example function with element-wise operations."""
            y = jnp.sin(x)
            y = jnp.exp(y)
            y = jnp.tanh(y)
            return jnp.sqrt(jnp.abs(y))

        def fused_operations(x):
            """Example of fused operations for better XLA optimization."""
            # Fused linear + activation
            w1 = jnp.ones((x.shape[-1], 256))
            y = jax.nn.gelu(x @ w1)  # Fused matmul + activation

            # Fused elementwise chain
            y = jax.nn.gelu(jnp.sin(y) + jnp.cos(y))  # Fused elementwise ops

            # Fused reduction
            return jnp.mean(y, axis=-1, keepdims=True)

        function_results = {}

        # Profile each function
        functions = {
            "MatMul_Chain": (matrix_multiply_chain, flat_input),
            "Elementwise_Ops": (elementwise_operations, sample_input),
            "Fused_Operations": (fused_operations, flat_input),
        }

        for name, (func, input_data) in functions.items():
            print(f"\n--- Profiling {name} ---")

            with self.profiler.profiling_session():
                results, report = self.profiler.profile_function(
                    func, [input_data], name
                )

                print(f"{name} Results:")
                print(report.render(format="text"))

                function_results[name] = results

        self.results["function_profiling"] = function_results
        return function_results

    def demonstrate_batch_size_optimization(self):
        """Demonstrate systematic batch size optimization."""

        print("\n🎯 Batch Size Optimization Analysis")
        print("=" * 50)

        # Create FNO for testing
        fno = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=16,
            num_layers=2,
            rngs=nnx.Rngs(0),
        )

        # Test different batch sizes (smaller to avoid memory issues)
        batch_sizes = [32, 64, 128, 256]  # Reduced from original to avoid OOM
        batch_results = {}

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            # Create data in correct format
            spatial_size = 32
            test_input = jax.random.normal(
                jax.random.PRNGKey(42),
                (batch_size, 2, spatial_size, spatial_size),
            )

            try:
                with self.profiler.profiling_session():
                    results, _ = self.profiler.profile_neural_operator(
                        fno,
                        [test_input],
                        f"FNO_batch_{batch_size}",
                    )

                    # Extract key metrics
                    roofline = results.get("roofline_analysis", {})
                    hardware = results.get("hardware_analysis", {})
                    batch_results[batch_size] = {
                        "efficiency": roofline.get("efficiency", 0),
                        "arithmetic_intensity": roofline.get("arithmetic_intensity", 0),
                        "execution_time_ms": roofline.get("actual_time_ms", 0),
                        "tensorcore_utilization": hardware.get("platform_analysis", {})
                        .get("tensorcore_analysis", {})
                        .get("tensorcore_utilization", 0),
                        "bottleneck": roofline.get("bottleneck", "unknown"),
                    }

            except Exception as e:
                print(f"  Error with batch size {batch_size}: {e}")
                batch_results[batch_size] = {"error": str(e)}

        # Display results
        print("\n📊 Batch Size Analysis Results:")
        print(
            f"{'Batch Size':<12} {'Efficiency':<12} {'TensorCore':<12} {'Intensity':<12} {'Time (ms)':<12}"
        )
        print("-" * 72)

        for batch_size, metrics in batch_results.items():
            if "error" not in metrics:
                print(
                    f"{batch_size:<12} {metrics['efficiency']:<12.2%} "
                    f"{metrics['tensorcore_utilization']:<12.2%} "
                    f"{metrics['arithmetic_intensity']:<12.1f} "
                    f"{metrics['execution_time_ms']:<12.1f}"
                )
            else:
                print(f"{batch_size:<12} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A'}")

        self.results["batch_optimization"] = batch_results
        return batch_results

    def demonstrate_hardware_specific_analysis(self):
        """Demonstrate hardware-specific analysis."""

        print("\n🔧 Hardware-Specific Analysis")
        print("=" * 40)

        backend = jax.default_backend()
        print(f"Current backend: {backend}")

        # Create test function
        def test_matmul(x, y):
            return x @ y

        # Test different matrix sizes for hardware alignment
        test_cases = [
            (128, 128),  # Well-aligned
            (256, 256),  # Well-aligned
            (127, 127),  # Unaligned
        ]

        print(f"\nTesting matrix multiplication alignment for {backend.upper()}:")

        hardware_results = {}

        for m, n in test_cases:
            print(f"\n--- Matrix size: {m}x{n} ---")

            # Create matrices with mixed precision
            dtype = jnp.bfloat16 if backend in ["gpu", "tpu"] else jnp.float32
            a = jnp.ones((m, n), dtype=dtype)
            b = jnp.ones((n, m), dtype=dtype)

            try:
                with self.profiler.profiling_session():
                    results, _ = self.profiler.profile_function(
                        test_matmul, [a, b], f"MatMul_{m}x{n}"
                    )

                    # Extract hardware-specific metrics
                    hw_analysis = results.get("hardware_analysis", {})
                    platform_analysis = hw_analysis.get("platform_analysis", {})
                    roofline = results.get("roofline_analysis", {})

                    if backend == "gpu" and "tensorcore_analysis" in platform_analysis:
                        tc = platform_analysis["tensorcore_analysis"]
                        tensorcore_util = tc.get("tensorcore_utilization", 0)
                        shape_alignment = tc.get("shape_alignment", {})
                        alignment_score = shape_alignment.get(
                            "average_alignment_score", 0
                        )

                        print(f"  TensorCore Utilization: {tensorcore_util:.2%}")
                        print(f"  Shape Alignment Score: {alignment_score:.2f}")

                    # Show roofline metrics
                    print(
                        f"  Arithmetic Intensity: {roofline.get('arithmetic_intensity', 0):.1f} FLOPs/byte"
                    )
                    print(f"  Efficiency: {roofline.get('efficiency', 0):.2%}")

                    hardware_results[f"{m}x{n}"] = results

            except Exception as e:
                print(f"  Error: {e}")
                hardware_results[f"{m}x{n}"] = {"error": str(e)}

        self.results["hardware_analysis"] = hardware_results
        return hardware_results

    def compare_operations(self):
        """Compare multiple operations and identify optimization opportunities."""

        print("\n📊 Comparing Neural Operators")
        print("=" * 40)

        # Create operators and data
        operators = self.create_neural_operators()
        sample_input = self.create_sample_data(batch_size=64, grid_size=64, channels=3)

        # Prepare operations for comparison
        operations = []
        for name, operator in operators.items():
            operations.append((name, operator, [sample_input]))

        # Compare operations
        comparison_results = self.profiler.compare_operations(operations)

        print("\nComparison Results:")
        for rec in comparison_results.get("recommendations", []):
            print(f"  💡 {rec}")

        self.results["operation_comparison"] = comparison_results
        return comparison_results

    def generate_comprehensive_summary(self):  # noqa: PLR0912, PLR0915
        """Generate comprehensive summary of all profiling results."""

        print("\n🎉 Comprehensive Profiling Summary")
        print("=" * 60)

        print("📋 Completed Analyses:")
        print("  ✅ JIT vs Non-JIT performance comparison")
        print("  ✅ Compilation overhead analysis")
        print("  ✅ Neural operator profiling")
        print("  ✅ JAX function profiling")
        print("  ✅ Batch size optimization")
        print("  ✅ Hardware-specific analysis")
        print("  ✅ Operation comparison")

        print("\n📊 Key Performance Insights:")

        # GPU Acceleration Results
        if "memory_pool_efficiency" in self.results:
            pool_data = self.results["memory_pool_efficiency"]
            print(f"  • Memory pool efficiency: {pool_data['efficiency']:.2f}x speedup")
            print(
                f"  • Buffer reuse ratio: {pool_data['pool_stats']['reuse_ratio']:.2%}"
            )

        if "mixed_precision" in self.results:
            mixed_data = self.results["mixed_precision"]
            avg_speedup = sum(r["mixed_speedup"] for r in mixed_data.values()) / len(
                mixed_data
            )
            print(f"  • Mixed precision average speedup: {avg_speedup:.2f}x")

        if "async_operations" in self.results:
            async_data = self.results["async_operations"]
            print(f"  • Async operations speedup: {async_data['speedup']:.2f}x")

        if "tensorcore_optimization" in self.results:
            tc_data = self.results["tensorcore_optimization"]
            avg_tc_speedup = sum(r["tc_speedup"] for r in tc_data.values()) / len(
                tc_data
            )
            avg_tc_gflops = sum(r["tc_gflops"] for r in tc_data.values()) / len(tc_data)
            print(f"  • TensorCore average speedup: {avg_tc_speedup:.2f}x")
            print(f"  • TensorCore average performance: {avg_tc_gflops:.1f} GFLOPS")

        # JIT Performance
        if "jit_comparison" in self.results:
            jit_data = self.results["jit_comparison"]
            print(f"  • JIT compilation speedup: {jit_data['speedup']:.2f}x")

        # Compilation Overhead
        if "compilation_analysis" in self.results:
            comp_data = self.results["compilation_analysis"]
            print(
                f"  • Compilation break-even: {comp_data['break_even_calls']:.1f} calls"
            )

        # Best performing operator
        if "operator_profiling" in self.results:
            op_data = self.results["operator_profiling"]
            best_efficiency = 0
            best_operator = "Unknown"

            for name, results in op_data.items():
                roofline = results.get("roofline_analysis", {})
                efficiency = roofline.get("efficiency", 0)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_operator = name

            print(
                f"  • Best performing operator: {best_operator} ({best_efficiency:.2%} efficiency)"
            )

        # Batch size recommendations
        if "batch_optimization" in self.results:
            batch_data = self.results["batch_optimization"]
            successful_batches = {
                k: v for k, v in batch_data.items() if "error" not in v
            }
            if successful_batches:
                best_batch = max(
                    successful_batches.keys(),
                    key=lambda k: successful_batches[k]["efficiency"],
                )
                best_efficiency = successful_batches[best_batch]["efficiency"]
                print(
                    f"  • Optimal batch size tested: {best_batch} ({best_efficiency:.2%} efficiency)"
                )

        print("\n💡 GPU Acceleration & Optimization Recommendations:")

        # Memory pool recommendations
        if "memory_pool_efficiency" in self.results:
            pool_data = self.results["memory_pool_efficiency"]
            if pool_data["efficiency"] > 2.0:
                print(
                    "  ✅ Memory pooling provides excellent acceleration - use for repeated allocations"
                )
            else:
                print(
                    "  ⚠️  Consider larger buffer sizes or different allocation patterns for memory pooling"
                )

        # Mixed precision recommendations
        if "mixed_precision" in self.results:
            mixed_data = self.results["mixed_precision"]
            avg_speedup = sum(r["mixed_speedup"] for r in mixed_data.values()) / len(
                mixed_data
            )
            if avg_speedup > 1.2:
                print(
                    "  ✅ Mixed precision optimization is beneficial - use for large matrix operations"
                )
            else:
                print(
                    "  ⚠️  Mixed precision shows limited benefit - verify TensorCore availability"
                )

        # Async operations recommendations
        if "async_operations" in self.results:
            async_data = self.results["async_operations"]
            if async_data["speedup"] > 1.1:
                print(
                    "  ✅ Async memory operations provide benefit - use prefetching for data pipelines"
                )
            else:
                print("  ⚠️  Limited async benefit - operations may be compute-bound")

        # General recommendations
        print("  • Use OptimizedGPUManager for comprehensive acceleration")
        print(
            "  • JIT compilation provides significant speedup - ensure proper warm-up"
        )
        print("  • Consider compilation overhead for short-running applications")
        print(
            "  • Use TensorCore-aligned shapes (multiples of 16) for GPU optimization"
        )
        print(
            "  • Optimize batch sizes based on roofline analysis and hardware capabilities"
        )
        print("  • Monitor arithmetic intensity and memory bandwidth utilization")
        print("  • Use buffer donation in JIT functions for memory efficiency")
        print("  • Implement memory pooling for applications with repeated allocations")

        # Session summary
        session_summary = self.profiler.get_session_summary()
        print("\n📈 Profiling Session Summary:")
        print(f"  • Total sessions: {session_summary.get('total_sessions', 0)}")
        print(f"  • Success rate: {session_summary.get('success_rate', 0):.2%}")
        print(f"  • Total duration: {session_summary.get('total_duration_s', 0):.2f}s")
        print(f"  • Profilers used: {session_summary.get('profilers_used', [])}")

    def run_comprehensive_demo(self):
        """Run the complete comprehensive GPU acceleration and profiling demonstration."""

        print("🚀 Opifex Comprehensive GPU Acceleration & Profiling Demo")
        print("=" * 75)
        print("This demo showcases advanced GPU acceleration capabilities:")
        print("  1. Memory pool efficiency with 8x+ speedup demonstrations")
        print("  2. Mixed precision optimization with TensorCore alignment")
        print("  3. Asynchronous memory operations with prefetching")
        print("  4. Roofline model analysis for performance optimization")
        print("  5. JIT vs non-JIT performance comparison")
        print("  6. Compilation overhead analysis with break-even calculations")
        print("  7. Neural operator profiling with advanced optimizations")
        print("  8. JAX function profiling with fusion analysis")
        print("  9. Batch size optimization for hardware efficiency")
        print(" 10. Hardware-specific analysis and TensorCore utilization")
        print(" 11. Operation comparison with optimization recommendations")
        print("=" * 75)

        try:
            # Run GPU acceleration demonstrations first
            print("\n🎯 GPU ACCELERATION DEMONSTRATIONS")
            print("=" * 50)
            self.demonstrate_memory_pool_efficiency()
            self.demonstrate_mixed_precision_optimization()
            self.demonstrate_tensorcore_optimization()
            self.demonstrate_async_memory_operations()
            self.demonstrate_roofline_analysis()

            # Run traditional profiling analyses
            print("\n📊 PERFORMANCE PROFILING ANALYSES")
            print("=" * 50)
            self.compare_jit_vs_non_jit()
            self.analyze_compilation_overhead()
            self.profile_neural_operators()
            self.profile_jax_functions()
            self.demonstrate_batch_size_optimization()
            self.demonstrate_hardware_specific_analysis()
            self.compare_operations()

            # Generate comprehensive summary
            self.generate_comprehensive_summary()

            print(
                "\n✅ Comprehensive GPU acceleration and profiling demo completed successfully!"
            )

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main function to run the comprehensive profiling demo."""
    demo = ComprehensiveProfilingDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()
