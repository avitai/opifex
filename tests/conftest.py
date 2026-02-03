"""
Opifex Testing Configuration - Enhanced GPU Safety and Dependency Management

This module provides comprehensive pytest configuration with:
1. Proactive GPU environment detection and configuration
2. Local .venv CUDA library management
3. Automatic JAX backend configuration before any operations
4. Optional dependency management with mocking
5. Hardware-aware test execution with intelligent fallback
6. Comprehensive error handling and diagnostics
"""

import os
import warnings
from pathlib import Path

import pytest


def setup_cuda_environment():
    """Set up CUDA environment variables for JAX."""
    # Set CUDA library path
    cuda_lib_path = "/usr/local/cuda/lib64"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if Path(cuda_lib_path).exists() and cuda_lib_path not in current_ld_path:
        if current_ld_path:
            new_ld_path = f"{cuda_lib_path}:{current_ld_path}"
        else:
            new_ld_path = cuda_lib_path
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

    # Set additional CUDA environment variables
    os.environ["CUDA_ROOT"] = "/usr/local/cuda"
    os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # JAX CUDA configuration - respect existing JAX_PLATFORMS setting
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    # Disable CUDA plugin validation to bypass cuSPARSE check
    os.environ["JAX_CUDA_PLUGIN_VERIFY"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


# Removed duplicate pytest_configure function - functionality moved to main pytest_configure below


# Add the project root to the Python path
project_root = Path(__file__).parent.parent
import sys


sys.path.insert(0, str(project_root))

# Import Opifex testing infrastructure after environment setup
try:
    from opifex.core.testing_infrastructure import (
        DependencyManager,
        ensure_safe_jax_environment,
        MockMetricsImplementation,
    )

    # Initialize the environment proactively to prevent segmentation faults
    _test_environment = ensure_safe_jax_environment()
except ImportError:
    # Fallback if Opifex testing infrastructure is not available
    _test_environment = None
    DependencyManager = None
    MockMetricsImplementation = None


# Now safe to import JAX and other components
import jax
import jax.numpy as jnp


# Configure JAX for testing - X64 precision for numerical accuracy
os.environ["JAX_ENABLE_X64"] = "True"

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="jax")

# Global managers
_dependency_manager = None


def get_dependency_manager():
    """Get the global dependency manager."""
    global _dependency_manager  # noqa: PLW0603
    if _dependency_manager is None and DependencyManager is not None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


@pytest.fixture
def device():
    """Provide a device fixture that works with both CPU and GPU."""
    # Try to get GPU device first, fall back to CPU
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            return gpu_devices[0]
        return jax.devices("cpu")[0]
    except Exception:
        return jax.devices("cpu")[0]


@pytest.fixture
def rngs():
    """Provide RNG fixture for tests."""
    try:
        from flax import nnx

        return nnx.Rngs(0)
    except ImportError:
        # Fallback if flax is not available
        import jax

        return jax.random.PRNGKey(0)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment for safe execution."""
    if _test_environment is not None:
        print("\n🔧 Opifex Test Environment Configuration:")
        print(f"📱 Backend: {_test_environment.backend.value}")
        print(f"🌍 Environment: {_test_environment.environment_type.value}")
        print(f"🚀 GPU Available: {_test_environment.gpu_available}")
        print(f"✅ GPU Safe: {_test_environment.gpu_safe}")
        cuda_paths = _test_environment.cuda_env.cuda_library_paths or []
        print(f"📦 CUDA Libraries: {len(cuda_paths)} paths")
        print(f"🔗 JAX CUDA: {_test_environment.cuda_env.jax_cuda_available}")
        print(
            f"📊 Dependencies: {len([d for d in _test_environment.dependencies.values() if d.value == 'available'])} available"
        )

        # Log CUDA environment details
        if _test_environment.cuda_env.venv_cuda_available:
            print(
                f"💾 CUDA Version: {_test_environment.cuda_env.cuda_version or 'Unknown'}"
            )
            print(f"🎯 GPU Devices: {_test_environment.cuda_env.gpu_devices_detected}")

        # Confirm JAX backend configuration
        jax_platforms = os.environ.get("JAX_PLATFORMS", "cpu")
        print(f"⚙️ JAX Platforms: {jax_platforms}")

        # Display safety status
        if _test_environment.gpu_safe:
            print("🟢 GPU backend enabled with full safety validation")
        elif _test_environment.gpu_available:
            print("🟡 GPU available but unsafe - using CPU fallback")
        else:
            print("🔵 CPU-only environment - optimal for development")

    return _test_environment


@pytest.fixture(scope="session")
def dependency_manager():
    """Provide the dependency manager for tests."""
    return get_dependency_manager()


@pytest.fixture(scope="session")
def test_environment():
    """Provide the test environment configuration."""
    return _test_environment


@pytest.fixture
def mock_prometheus_metrics():
    """Provide a mock Prometheus metrics implementation."""
    if MockMetricsImplementation is not None:
        return MockMetricsImplementation()
    return None


@pytest.fixture
def safe_context():
    """Provide a safe JAX context for tests."""
    if _test_environment is not None:
        from opifex.core.testing_infrastructure import ensure_safe_jax_environment

        ensure_safe_jax_environment()

    # Return a simple context manager that does nothing
    from contextlib import nullcontext

    return nullcontext()


def pytest_runtest_setup(item):
    """Set up for each test run - ensure clean JAX state."""
    # Clear any JAX compilation cache to avoid issues
    try:
        jax.clear_caches()

        # If this is a GPU test, ensure GPU memory is available
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                # Force garbage collection before GPU tests
                import gc

                gc.collect()

                # Verify GPU is available for GPU-marked tests
                try:
                    gpu_devices = jax.devices("gpu")
                    if not gpu_devices:
                        pytest.skip("GPU test skipped: No GPU devices available")
                except Exception:
                    pytest.skip("GPU test skipped: GPU not accessible")

    except Exception:
        pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test run - clean up GPU memory."""
    try:
        # Clear JAX caches after each test
        jax.clear_caches()

        # Force garbage collection to free GPU memory
        import gc

        gc.collect()

        # If this was a GPU test, ensure memory cleanup
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                try:
                    # Additional GPU memory cleanup
                    # Force any pending operations to complete
                    dummy = jnp.array([1.0])
                    dummy.block_until_ready()
                    del dummy
                except Exception:
                    pass

    except Exception:
        pass


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Configure logging for individual tests."""
    import logging

    # Set appropriate log levels for testing
    logging.getLogger("opifex").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


@pytest.fixture
def sample_data():
    """Provide sample data for testing using JAX's default device selection."""
    return {
        "x_1d": jnp.linspace(0, 1, 100),
        "x_2d": jnp.meshgrid(jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10)),
        "y_simple": jnp.sin(jnp.linspace(0, 2 * jnp.pi, 100)),
        "matrix_small": jnp.ones((10, 10)),
        "matrix_medium": jnp.ones((100, 100)),
    }


@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for visualization tests."""
    from unittest.mock import MagicMock, patch

    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        yield {"fig": mock_fig, "ax": mock_ax, "subplots": mock_subplots}


@pytest.fixture
def benchmark_result():
    """Sample BenchmarkResult for testing."""
    from opifex.benchmarking.evaluation_engine import BenchmarkResult

    return BenchmarkResult(
        model_name="TestModel",
        dataset_name="TestDataset",
        metrics={"mse": 0.001, "mae": 0.01, "relative_error": 0.05},
        execution_time=1.5,
        framework_version="flax_nnx",
    )


@pytest.fixture
def sample_field_data():
    """Sample field data for visualization tests."""
    return {
        "field_2d": jnp.sin(jnp.linspace(0, 2 * jnp.pi, 64).reshape(8, 8)),
        "field_3d": jnp.ones((8, 8, 3)),
        "field_sequence": jnp.ones((10, 16, 16)),
        "ground_truth": jnp.ones((16, 16)),
        "prediction": jnp.ones((16, 16)) * 1.01,
    }


@pytest.fixture
def mock_registry_service():
    """Mock registry for scalability tests."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.retrieve_functional = AsyncMock(return_value=None)
    mock.search_functionals = AsyncMock(return_value=[])
    return mock


# Enhanced pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest environment with proper JAX/CUDA handling and custom markers."""
    # Setup CUDA environment first, before any JAX imports
    setup_cuda_environment()

    # Suppress CUDA warnings and errors in test output
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="jax._src.xla_bridge"
    )
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")

    # Import JAX and configure after environment setup
    try:
        import jax

        # Force JAX to initialize with current environment
        # Respect the JAX_PLATFORMS environment variable
        current_platforms = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
        jax.config.update("jax_platforms", current_platforms)

        # Check if CUDA is available
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            cpu_devices = [d for d in devices if d.platform == "cpu"]

            config.addinivalue_line(
                "markers",
                f"gpu_available: GPU devices available: {len(gpu_devices) > 0}",
            )
            config.addinivalue_line("markers", f"devices: Available devices: {devices}")

            print(f"\nGPU available for testing: {len(gpu_devices) > 0}")
            if gpu_devices:
                print(f"GPU devices: {gpu_devices}")
            print(f"CPU devices: {cpu_devices}")

        except Exception as e:
            print(f"\nDevice detection failed: {e}")
            config.addinivalue_line(
                "markers", "gpu_available: GPU devices available: False"
            )

    except ImportError as e:
        print(f"\nJAX import failed: {e}")

    # Configure custom pytest markers
    config.addinivalue_line(
        "markers", "gpu_required: mark test as requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "gpu_preferred: mark test as preferring GPU but can run on CPU"
    )
    config.addinivalue_line(
        "markers", "cpu_safe: mark test as safe to run on CPU-only systems"
    )
    config.addinivalue_line(
        "markers", "requires_prometheus: mark test as requiring Prometheus client"
    )
    config.addinivalue_line("markers", "requires_psutil: mark test as requiring psutil")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "cuda_local: mark test as requiring local .venv CUDA"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment capabilities."""
    dep_manager = get_dependency_manager()

    # Handle slow tests
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)

    if dep_manager is None:
        return

    for item in items:
        # Handle dependency-required tests
        if item.get_closest_marker(
            "requires_prometheus"
        ) and not dep_manager.is_available("prometheus_client"):
            item.add_marker(pytest.mark.skip(reason="Prometheus client not available"))

        if item.get_closest_marker("requires_psutil") and not dep_manager.is_available(
            "psutil"
        ):
            item.add_marker(pytest.mark.skip(reason="psutil not available"))


@pytest.fixture(scope="function")
def reset_jax_config():
    """Reset JAX configuration after each test."""
    yield
    # Clear JAX compilation cache to prevent memory issues
    jax.clear_caches()


def pytest_runtest_makereport(item, call):
    """Handle test execution with comprehensive error handling."""
    if call.when == "call" and call.excinfo and "CUDA" in str(call.excinfo.value):
        # Check for GPU-related failures and provide helpful diagnostics
        print("\n⚠️ CUDA-related test failure detected:")
        if _test_environment is not None:
            print(f"Environment Type: {_test_environment.environment_type.value}")
            print(f"GPU Safe: {_test_environment.gpu_safe}")
        print(f"JAX Platforms: {os.environ.get('JAX_PLATFORMS', 'cpu')}")

        if _test_environment is not None and not _test_environment.gpu_safe:
            print("💡 Consider running with CPU-only backend: JAX_PLATFORMS=cpu")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_session():
    """Clean up test session resources."""
    yield

    # Final cleanup
    jax.clear_caches()

    # Log final environment status
    if _test_environment is not None:
        print(
            f"\n🏁 Test session completed with {_test_environment.environment_type.value} environment"
        )
    else:
        print("\n🏁 Test session completed")
