#!/usr/bin/env python3
"""
Enhanced GPU Test Management Script for Opifex Framework.

Based on comprehensive insights from workshop project for handling GPU test failures,
with intelligent test routing and progressive testing strategies.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Import our enhanced utilities
try:
    from gpu_utils import (
        configure_jax_env_vars,
        print_comprehensive_gpu_info,
    )
except ImportError:
    # Fallback if gpu_utils is not available
    def configure_jax_env_vars(force_cpu=False):
        """Configure JAX environment variables for optimal performance."""

    def print_comprehensive_gpu_info():
        """Print comprehensive GPU information for debugging."""
        print("GPU utilities not available")


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available using nvidia-smi."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


class EnhancedGPUTestManager:
    """Enhanced GPU test manager with workshop insights and intelligent routing."""

    def __init__(self, verbose: bool = False):
        self.project_root = Path(__file__).parent.parent
        self.verbose = verbose
        self.test_results: dict[str, bool] = {}
        self.setup_environment()

    def setup_environment(self) -> None:
        """Set up environment variables for optimal GPU testing."""
        # Use our enhanced environment configuration
        configure_jax_env_vars()

        # Additional workshop-inspired environment variables
        workshop_env_vars = {
            "PYTEST_CUDA_ENABLED": "true",
            "JAX_PLATFORMS": "cuda,cpu",  # Prefer CUDA but fallback to CPU
            "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
            "JAX_CUDA_PLUGIN_VERIFY": "false",
            "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
            "CUDA_ROOT": "/usr/local/cuda",
            "CUDA_HOME": "/usr/local/cuda",
        }

        for key, value in workshop_env_vars.items():
            if key == "LD_LIBRARY_PATH":
                current = os.environ.get(key, "")
                if current and value not in current:
                    os.environ[key] = f"{value}:{current}"
                else:
                    os.environ[key] = value
            else:
                os.environ[key] = value

        if self.verbose:
            print("✅ Environment configured for optimal GPU testing")

    def detect_test_markers(self, test_path: str) -> dict[str, bool]:
        """Detect test markers in the given test path to determine strategy."""
        markers = {
            "has_gpu_tests": False,
            "has_cpu_tests": False,
            "has_cuda_specific": False,
            "requires_sequential": False,
        }

        try:
            # Search for pytest markers in test files
            if os.path.isfile(test_path):
                test_files = [test_path]
            elif os.path.isdir(test_path):
                test_files = list(Path(test_path).rglob("test_*.py"))
            else:
                test_files = []

            for test_file in test_files:
                try:
                    with open(test_file) as f:
                        content = f.read()

                    if "@pytest.mark.gpu" in content or "mark.gpu" in content:
                        markers["has_gpu_tests"] = True
                        markers["requires_sequential"] = (
                            True  # GPU tests should run sequentially
                        )

                    if "@pytest.mark.cuda" in content or "mark.cuda" in content:
                        markers["has_cuda_specific"] = True
                        markers["requires_sequential"] = True

                    if "@pytest.mark.cpu" in content or "mark.cpu" in content:
                        markers["has_cpu_tests"] = True

                    # If no specific markers, assume it's CPU-safe for parallel execution
                    if not any(
                        [markers["has_gpu_tests"], markers["has_cuda_specific"]]
                    ):
                        markers["has_cpu_tests"] = True

                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Could not analyze {test_file}: {e}")
                    # Default to safe assumptions
                    markers["has_cpu_tests"] = True

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Test marker detection failed: {e}")
            # Default to safe assumptions
            markers["has_cpu_tests"] = True

        return markers

    def check_gpu_status(self) -> tuple[bool, str, list | None]:
        """Comprehensive GPU status check with detailed information."""
        try:
            import jax

            # Basic availability check using system-level detection
            system_gpu = has_nvidia_gpu()
            if not system_gpu:
                return False, "No NVIDIA GPU detected via nvidia-smi", None

            # JAX GPU check
            devices = jax.devices()
            backend = jax.default_backend()
            gpu_available = backend == "gpu"

            if not gpu_available:
                return (
                    False,
                    f"JAX cannot access GPU. Backend: {backend}, Available devices: {devices}",
                    None,
                )

            # Test basic GPU operation
            try:
                import jax.numpy as jnp

                test_array = jnp.ones((10, 10))
                result = jnp.sum(test_array)
                result.block_until_ready()

                return True, f"GPU fully functional: {backend}", devices

            except Exception as e:
                return False, f"GPU detected but operations failing: {e}", devices

        except ImportError as e:
            return False, f"JAX not available: {e}", None
        except Exception as e:
            return False, f"GPU check failed: {e}", None

    def run_cpu_tests(
        self, test_path: str = "tests/", extra_args: list[str] = None
    ) -> subprocess.CompletedProcess:
        """Run tests with CPU-only configuration and parallel execution."""
        if self.verbose:
            print("🔄 Running CPU-only tests with parallel execution...")

        # Force CPU-only environment
        original_platforms = os.environ.get("JAX_PLATFORMS", "")
        os.environ["JAX_PLATFORMS"] = "cpu"

        cmd = [
            "uv",
            "run",
            "pytest",
            test_path,
            "-m",
            "not gpu and not cuda",  # Exclude GPU/CUDA marked tests
            "-n",
            "auto",  # Parallel execution for CPU tests
            "--tb=short",
            "--verbose" if self.verbose else "--quiet",
        ]

        if extra_args:
            cmd.extend(extra_args)

        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        end_time = time.time()

        # Restore environment
        if original_platforms:
            os.environ["JAX_PLATFORMS"] = original_platforms
        else:
            os.environ.pop("JAX_PLATFORMS", None)

        test_type = "CPU"
        success = result.returncode == 0
        self.test_results[f"{test_type}_{test_path}"] = success

        if self.verbose:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} CPU tests completed in {end_time - start_time:.2f}s")

        return result

    def run_gpu_tests(
        self, test_path: str = "tests/", extra_args: list[str] = None
    ) -> subprocess.CompletedProcess:
        """Run GPU tests with sequential execution to prevent conflicts."""
        gpu_available, gpu_status, _ = self.check_gpu_status()

        if not gpu_available:
            if self.verbose:
                print(f"❌ GPU not available: {gpu_status}")
                print("🔄 Falling back to CPU-only tests...")
            return self.run_cpu_tests(test_path, extra_args)

        if self.verbose:
            print(f"✅ GPU available: {gpu_status}")
            print("🔄 Running GPU tests sequentially to prevent conflicts...")

        cmd = [
            "uv",
            "run",
            "pytest",
            test_path,
            "-m",
            "gpu or cuda",  # Only GPU/CUDA marked tests
            "--tb=short",
            "--maxfail=1",  # Fail fast on GPU issues per workshop insight
            "--verbose" if self.verbose else "--quiet",
        ]

        # Sequential execution for GPU tests (no -n auto)
        if extra_args:
            cmd.extend(extra_args)

        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        end_time = time.time()

        test_type = "GPU"
        success = result.returncode == 0
        self.test_results[f"{test_type}_{test_path}"] = success

        if self.verbose:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} GPU tests completed in {end_time - start_time:.2f}s")

        if not success and self.verbose:
            print("❌ GPU tests failed. Consider running CPU fallback.")

        return result

    def run_smart_test_execution(
        self, test_path: str = "tests/", extra_args: list[str] = None
    ) -> subprocess.CompletedProcess:
        """Smart test execution with workshop-inspired intelligent routing."""
        if self.verbose:
            print("🚀 Starting smart test execution with intelligent routing...")
            print("=" * 60)

        # Analyze test markers to determine strategy
        markers = self.detect_test_markers(test_path)

        if self.verbose:
            print(f"📋 Test Analysis for {test_path}:")
            for marker, detected in markers.items():
                icon = "✅" if detected else "❌"
                print(f"   {icon} {marker}: {detected}")

        total_start_time = time.time()

        # Phase 1: Always run CPU tests first (fast parallel execution)
        if markers["has_cpu_tests"]:
            if self.verbose:
                print("\n📋 Phase 1: CPU Tests (Parallel Execution)")
            cpu_result = self.run_cpu_tests(test_path, extra_args)

            if cpu_result.returncode != 0:
                if self.verbose:
                    print("❌ CPU tests failed - stopping execution")
                return cpu_result
            if self.verbose:
                print("✅ CPU tests passed")
        else:
            cpu_result = None
            if self.verbose:
                print("INFO: No CPU tests detected, skipping Phase 1")

        # Phase 2: Run GPU tests if available and requested
        if markers["has_gpu_tests"] or markers["has_cuda_specific"]:
            if self.verbose:
                print("\n📋 Phase 2: GPU Tests (Sequential Execution)")
            gpu_result = self.run_gpu_tests(test_path, extra_args)
        else:
            gpu_result = cpu_result
            if self.verbose:
                print("INFO: No GPU-specific tests detected, skipping Phase 2")

        total_end_time = time.time()

        # Summary report
        if self.verbose:
            print("\n📋 Smart Test Execution Summary:")
            print(f"   Total time: {total_end_time - total_start_time:.2f}s")
            print("   Test results:")
            for test_name, passed in self.test_results.items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"     {status} {test_name}")

        # Return the most relevant result
        return gpu_result if gpu_result else cpu_result

    def run_progressive_gpu_test(self, sizes: list[int] = None) -> bool:
        """Run progressive GPU tests with increasing complexity."""
        if sizes is None:
            sizes = [1000, 5000, 10000]

        if self.verbose:
            print("🔬 Running progressive GPU tests...")

        gpu_available, gpu_status, _ = self.check_gpu_status()
        if not gpu_available:
            if self.verbose:
                print(f"❌ GPU not available for progressive testing: {gpu_status}")
            return False

        try:
            import jax.numpy as jnp
            from jax import random

            key = random.key(0)

            for size in sizes:
                try:
                    if self.verbose:
                        print(f"   Testing {size}x{size} matrix operations...")

                    a = random.normal(key, (size, size))
                    b = random.normal(key, (size, size))

                    start = time.time()
                    result = jnp.dot(a, b)
                    result.block_until_ready()
                    end = time.time()

                    if self.verbose:
                        print(f"   ✅ {size}x{size} completed in {end - start:.3f}s")

                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ {size}x{size} failed: {e}")
                    return False

            if self.verbose:
                print("✅ All progressive GPU tests passed")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ Progressive GPU testing failed: {e}")
            return False

    def print_comprehensive_status(self) -> None:
        """Print comprehensive GPU and test status."""
        print("🔍 Opifex Enhanced GPU Test Manager Status")
        print("=" * 60)

        # System information
        print_comprehensive_gpu_info()

        # Test environment status
        print("\n🧪 Test Environment Status:")
        important_vars = [
            "JAX_PLATFORMS",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "JAX_SKIP_CUDA_CONSTRAINTS_CHECK",
            "PYTEST_CUDA_ENABLED",
        ]

        for var in important_vars:
            value = os.environ.get(var, "Not set")
            print(f"   {var}: {value}")


def main() -> None:
    """Enhanced main entry point with comprehensive command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced GPU Test Manager for Opifex Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/gpu_test_manager.py check              # Check GPU status
  python scripts/gpu_test_manager.py cpu                # Run CPU-only tests
  python scripts/gpu_test_manager.py gpu                # Run GPU tests
  python scripts/gpu_test_manager.py smart              # Smart test execution
  python scripts/gpu_test_manager.py status             # Comprehensive status
  python scripts/gpu_test_manager.py progressive        # Progressive GPU testing
  python scripts/gpu_test_manager.py tests/core/ -v     # Test specific path
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="smart",
        help="Command to execute (default: smart)",
    )
    parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/",
        help="Test path to execute (default: tests/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--maxfail",
        type=int,
        default=1,
        help="Maximum number of failures before stopping",
    )
    parser.add_argument(
        "--tb", choices=["short", "long", "no"], default="short", help="Traceback style"
    )

    args = parser.parse_args()

    # Create manager with verbosity setting
    manager = EnhancedGPUTestManager(verbose=args.verbose)

    # Prepare extra arguments for pytest
    extra_args = []
    if args.maxfail != 1:
        extra_args.extend(["--maxfail", str(args.maxfail)])
    if args.tb != "short":
        extra_args.extend(["--tb", args.tb])

    # Execute requested command
    if args.command == "check":
        gpu_available, gpu_status, _ = manager.check_gpu_status()
        if gpu_available:
            print(f"✅ GPU Available: {gpu_status}")
            sys.exit(0)
        else:
            print(f"❌ GPU Not Available: {gpu_status}")
            sys.exit(1)

    elif args.command == "cpu":
        result = manager.run_cpu_tests(args.test_path, extra_args)
        sys.exit(result.returncode)

    elif args.command == "gpu":
        result = manager.run_gpu_tests(args.test_path, extra_args)
        sys.exit(result.returncode)

    elif args.command == "smart":
        result = manager.run_smart_test_execution(args.test_path, extra_args)
        sys.exit(result.returncode)

    elif args.command == "status":
        manager.print_comprehensive_status()
        sys.exit(0)

    elif args.command == "progressive":
        success = manager.run_progressive_gpu_test()
        sys.exit(0 if success else 1)

    elif args.command.startswith("tests/") or os.path.exists(args.command):
        # Treat as test path
        test_path = args.command
        result = manager.run_smart_test_execution(test_path, extra_args)
        sys.exit(result.returncode)

    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
