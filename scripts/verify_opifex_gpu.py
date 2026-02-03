#!/usr/bin/env python
"""
Comprehensive GPU verification script for Opifex framework.

This script tests matrix multiplication with progressive sizes to identify
and resolve segmentation faults in JAX operations, based on workshop insights.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import our enhanced GPU utilities
from gpu_utils import (
    configure_jax_env_vars,
    get_gpu_memory_info,
    get_optimal_jax_env_vars,
    print_comprehensive_gpu_info,
)
from jax import random


def print_system_info() -> None:
    """Print system and JAX configuration information."""
    print("\n=== Opifex GPU Verification System Information ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Print important environment variables
    env_vars = [
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "JAX_ENABLE_X64",
        "TF_CPP_MIN_LOG_LEVEL",
        "JAX_PLATFORMS",
        "JAX_CUDA_PLUGIN_VERIFY",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK",
    ]

    print("\n=== Environment Variables ===")
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")


def test_small_matmul() -> bool:
    """Test small matrix multiplication for basic functionality."""
    print("\n=== Testing Small Matrix Multiplication ===")
    key = random.key(0)

    try:
        # Small matrices (should work fine)
        a = random.normal(key, (1000, 1000))
        b = random.normal(key, (1000, 1000))

        start = time.time()
        result = jnp.dot(a, b)
        # Force execution to complete
        result.block_until_ready()
        end = time.time()

        print(f"Small matrix multiplication completed in {end - start:.4f} seconds")
        print(f"Result shape: {result.shape}")
        print(f"Memory usage after operation: {get_gpu_memory_info()}")
        return True

    except Exception as e:
        print(f"❌ Small matrix multiplication failed: {e}")
        return False


def test_large_matmul(size: int = 10000) -> bool:
    """Test large matrix multiplication that might cause segfault."""
    print(f"\n=== Testing Large Matrix Multiplication ({size}x{size}) ===")
    print("This might cause segmentation fault if memory settings are incorrect")

    try:
        # Check available memory first
        memory_info = get_gpu_memory_info()
        if memory_info["free"] and memory_info["free"] < 4000:  # Less than 4GB free
            print(f"⚠️  Limited GPU memory ({memory_info['free']} MB free)")
            print("   Reducing matrix size for safety")
            size = min(size, 5000)

        print("Creating matrices...")
        a = np.random.normal(size=(size, size))
        b = np.random.normal(size=(size, size))

        print("Starting matrix multiplication...")
        start = time.time()

        # Use jit to optimize - workshop insight
        @jax.jit
        def matmul(x, y):
            return jnp.dot(x, y)

        result = matmul(a, b)
        # Force execution
        result.block_until_ready()
        end = time.time()

        print(f"Large matrix multiplication completed in {end - start:.4f} seconds")
        print(f"Result shape: {result.shape}")
        print(f"Final memory usage: {get_gpu_memory_info()}")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ GPU out of memory: {e}")
            print(
                "   Recommendation: Reduce XLA_PYTHON_CLIENT_MEM_FRACTION or use smaller matrices"
            )
        else:
            print(f"❌ Runtime error: {e}")
        return False

    except MemoryError as e:
        print(f"❌ Memory error: {e}")
        print("   Recommendation: Reduce matrix size or adjust memory settings")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   This might indicate a segmentation fault or CUDA configuration issue")
        return False


def test_neural_network_operations() -> bool:
    """Test basic neural network operations that Opifex uses."""
    print("\n=== Testing Neural Network Operations ===")

    try:
        # Test basic MLP operations similar to Opifex
        key = random.key(42)
        key, init_key = random.split(key)

        # Create a simple MLP
        class TestMLP(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.linear1 = nnx.Linear(784, 256, rngs=rngs)
                self.linear2 = nnx.Linear(256, 128, rngs=rngs)
                self.linear3 = nnx.Linear(128, 10, rngs=rngs)

            def __call__(self, x):
                x = nnx.relu(self.linear1(x))
                x = nnx.relu(self.linear2(x))
                return self.linear3(x)

        rngs = nnx.Rngs(init_key)
        model = TestMLP(rngs=rngs)

        # Test forward pass
        batch_size = 32
        x = random.normal(key, (batch_size, 784))

        start = time.time()
        output = model(x)
        output.block_until_ready()
        end = time.time()

        print(f"Neural network forward pass completed in {end - start:.4f} seconds")
        print(f"Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"Memory after neural network test: {get_gpu_memory_info()}")

        # Test gradient computation (important for Opifex)
        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        start = time.time()
        loss, _ = nnx.value_and_grad(loss_fn)(model, x)
        loss.block_until_ready()
        end = time.time()

        print(f"Gradient computation completed in {end - start:.4f} seconds")
        print(f"Loss value: {loss:.6f}")
        return True

    except Exception as e:
        print(f"❌ Neural network operations failed: {e}")
        return False


def test_opifex_specific_operations() -> bool:
    """Test operations specific to scientific machine learning."""
    print("\n=== Testing Opifex-Specific Operations ===")

    try:
        key = random.key(0)

        # Test PDE-like operations (finite differences)
        print("Testing PDE finite difference operations...")
        grid_size = 100
        x = jnp.linspace(0, 1, grid_size)
        y = jnp.linspace(0, 1, grid_size)
        X, Y = jnp.meshgrid(x, y)

        # Test function that might appear in PDE solving
        _ = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)

        # Compute gradients (common in PDE residuals)
        grad_u = jax.grad(lambda x, y: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y))

        # Test on a batch of points
        points = random.uniform(key, (1000, 2))
        grad_values = jax.vmap(lambda p: grad_u(p[0], p[1]))(points)
        grad_values.block_until_ready()

        print(f"PDE gradient computation shape: {grad_values.shape}")

        # Test physics-informed loss computation
        print("Testing physics-informed loss computation...")

        def physics_residual(x, y):
            """Simple Poisson equation residual."""
            u_xx = jax.grad(jax.grad(lambda x: jnp.sin(jnp.pi * x)), 0)(x)
            u_yy = jax.grad(jax.grad(lambda y: jnp.cos(jnp.pi * y)), 0)(y)
            return u_xx + u_yy + jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

        # Evaluate residual on batch of points
        residuals = jax.vmap(lambda p: physics_residual(p[0], p[1]))(points)
        residuals.block_until_ready()

        loss = jnp.mean(residuals**2)
        print(f"Physics residual loss: {loss:.6f}")

        print("✅ Opifex operations completed successfully")
        return True

    except Exception as e:
        print(f"❌ Opifex operations failed: {e}")
        return False


def run_comprehensive_tests() -> dict[str, bool]:
    """Run comprehensive GPU tests and return results."""
    print("🧪 Opifex GPU Comprehensive Test Suite")
    print("=" * 50)

    # Configure environment for optimal performance
    configure_jax_env_vars()

    # Print system information
    print_system_info()

    # Initialize results
    results = {
        "system_info": True,
        "small_matmul": False,
        "large_matmul": False,
        "neural_network": False,
        "opifex_operations": False,
    }

    # Run tests in order of increasing complexity
    print("\n" + "=" * 50)
    print("STARTING COMPREHENSIVE TESTS")
    print("=" * 50)

    # Test 1: Small matrix multiplication
    results["small_matmul"] = test_small_matmul()

    if not results["small_matmul"]:
        print("\n⚠️  Even small matrix multiplication failed.")
        print("   This indicates serious JAX/CUDA configuration issues.")
        return results

    # Test 2: Large matrix multiplication (progressive sizes)
    sizes = [5000, 10000, 15000]
    large_test_passed = False

    for size in sizes:
        try:
            if test_large_matmul(size):
                print(f"✅ Success: {size}x{size} matrix multiplication completed")
                large_test_passed = True
                break
            print(f"❌ Failed: {size}x{size} matrix multiplication")
            continue
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            break

    results["large_matmul"] = large_test_passed

    # Test 3: Neural network operations
    results["neural_network"] = test_neural_network_operations()

    # Test 4: Opifex-specific operations
    results["opifex_operations"] = test_opifex_specific_operations()

    return results


def print_recommendations(results: dict[str, bool]) -> None:
    """Print recommendations based on test results."""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR SCIML GPU SETUP")
    print("=" * 50)

    failed_tests = [test for test, passed in results.items() if not passed]

    if not failed_tests:
        print("🎉 All tests passed! Your GPU setup is optimal for Opifex.")
        print("\nOptimal configuration detected:")
        env_vars = get_optimal_jax_env_vars()
        for key, value in list(env_vars.items())[:5]:
            print(f"   {key}={value}")
        return

    print("⚠️  Some tests failed. Here are specific recommendations:")

    if "small_matmul" in failed_tests:
        print("\n🔴 Critical: Basic operations failing")
        print("1. Check JAX installation:")
        print("   uv pip install --upgrade 'jax[cuda12_pip]>=0.6.1' 'jaxlib>=0.6.1'")
        print("2. Verify CUDA installation:")
        print("   nvidia-smi")
        print("3. Check environment variables:")
        print("   export JAX_PLATFORMS=cuda,cpu")

    if "large_matmul" in failed_tests:
        print("\n🟡 Memory/Performance: Large matrix operations failing")
        print("1. Adjust memory fraction:")
        print("   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6  # Reduce from 0.75")
        print("2. Disable memory preallocation:")
        print("   export XLA_PYTHON_CLIENT_PREALLOCATE=false")
        print("3. Use smaller batch sizes in your Opifex models")

    if "neural_network" in failed_tests:
        print("\n🟠 Neural Networks: FLAX NNX operations failing")
        print("1. Update FLAX:")
        print("   uv pip install --upgrade 'flax>=0.10.6'")
        print("2. Check for cuDNN issues:")
        print("   export JAX_CUDA_PLUGIN_VERIFY=false")

    if "opifex_operations" in failed_tests:
        print("\n🟣 Opifex: Physics-informed operations failing")
        print("1. This may indicate gradient computation issues")
        print("2. Try enabling X64 precision for better numerical stability:")
        print("   export JAX_ENABLE_X64=1")
        print("3. Check for automatic differentiation conflicts")

    print("\n📋 Quick setup command:")
    print("python scripts/gpu_utils.py --comprehensive")


def main() -> None:
    """Main function to run GPU verification."""
    print("🔍 Opifex GPU Verification Tool")
    print("Based on workshop insights for optimal performance")
    print("=" * 60)

    # Check available devices using JAX's device detection
    backend = jax.default_backend()
    devices = jax.devices()

    print(f"JAX backend: {backend}")
    print(f"Available devices: {devices}")

    # Check if we have non-CPU devices available
    has_accelerator = any(device.device_kind != "cpu" for device in devices)

    if not has_accelerator:
        print(
            "❌ No accelerator devices detected. This tool is designed for GPU systems."
        )
        print("   For CPU-only setup, use: python scripts/gpu_utils.py")
        print("   JAX will use CPU backend by default.")
        return

    # Print comprehensive GPU info first
    print_comprehensive_gpu_info()

    # Run all tests
    results = run_comprehensive_tests()

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("🎉 SUCCESS: All GPU tests passed!")
        print("   Your system is ready for Opifex development.")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed.")
        print("   See recommendations below.")

    # Print specific recommendations
    print_recommendations(results)

    print("\n✅ GPU verification complete.")
    print("   Use 'uv run opifex-gpu-test' for ongoing test management.")


if __name__ == "__main__":
    main()
