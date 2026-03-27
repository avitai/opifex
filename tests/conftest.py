"""Opifex Testing Configuration.

Lightweight pytest configuration that defers JAX initialization and device
probing to test execution time, keeping test collection fast and side-effect
free.  Environment variables are managed by activate.sh / .opifex.env / .env
and pytest-env in pyproject.toml — this module does NOT mutate them.
"""

import os
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root on sys.path (needed for editable installs to resolve)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
import sys


sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Lazy imports — resolved at first use, not at collection time
# ---------------------------------------------------------------------------
_test_environment = None
_test_env_initialized = False
DependencyManager = None
MockMetricsImplementation = None


def _init_test_environment():
    """Initialize test environment lazily on first use."""
    global _test_environment, _test_env_initialized, DependencyManager, MockMetricsImplementation  # noqa: PLW0603
    if _test_env_initialized:
        return _test_environment
    _test_env_initialized = True
    try:
        from opifex.core.testing_infrastructure import (
            DependencyManager as _DepMgr,
            ensure_safe_jax_environment,
            MockMetricsImplementation as _MockMetrics,
        )

        DependencyManager = _DepMgr
        MockMetricsImplementation = _MockMetrics
        _test_environment = ensure_safe_jax_environment()
    except ImportError:
        pass
    return _test_environment


import jax
import jax.numpy as jnp


# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="jax")

_dependency_manager = None


def get_dependency_manager():
    """Get the global dependency manager, initializing lazily."""
    global _dependency_manager  # noqa: PLW0603
    _init_test_environment()
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
    except Exception:  # JAX backend errors are unpredictable (RuntimeError, XlaError)
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
    """Initialize test environment lazily at session start (not collection time)."""
    env = _init_test_environment()
    return env


@pytest.fixture(scope="session")
def dependency_manager():
    """Provide the dependency manager for tests."""
    return get_dependency_manager()


@pytest.fixture(scope="session")
def test_environment():
    """Provide the test environment configuration."""
    return _init_test_environment()


@pytest.fixture
def mock_prometheus_metrics():
    """Provide a mock Prometheus metrics implementation."""
    _init_test_environment()
    if MockMetricsImplementation is not None:
        return MockMetricsImplementation()
    return None


@pytest.fixture
def safe_context():
    """Provide a safe JAX context for tests."""
    env = _init_test_environment()
    if env is not None:
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
    from calibrax.core.models import Metric

    from opifex.benchmarking.evaluation_engine import BenchmarkResult

    return BenchmarkResult(
        name="TestModel",
        tags={"dataset": "TestDataset"},
        metrics={
            "mse": Metric(value=0.001),
            "mae": Metric(value=0.01),
            "relative_error": Metric(value=0.05),
        },
        metadata={
            "execution_time": 1.5,
            "framework_version": "flax_nnx",
        },
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


def pytest_configure(config):
    """Register custom markers.  No JAX init or device probing here."""
    warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.xla_bridge")
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")

    config.addinivalue_line("markers", "gpu_required: mark test as requiring GPU hardware")
    config.addinivalue_line(
        "markers", "gpu_preferred: mark test as preferring GPU but can run on CPU"
    )
    config.addinivalue_line("markers", "cpu_safe: mark test as safe to run on CPU-only systems")
    config.addinivalue_line(
        "markers", "requires_prometheus: mark test as requiring Prometheus client"
    )
    config.addinivalue_line("markers", "requires_psutil: mark test as requiring psutil")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "cuda_local: mark test as requiring local .venv CUDA")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


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
        if item.get_closest_marker("requires_prometheus") and not dep_manager.is_available(
            "prometheus_client"
        ):
            item.add_marker(pytest.mark.skip(reason="Prometheus client not available"))

        if item.get_closest_marker("requires_psutil") and not dep_manager.is_available("psutil"):
            item.add_marker(pytest.mark.skip(reason="psutil not available"))


@pytest.fixture(scope="function")
def reset_jax_config():
    """Reset JAX configuration after each test."""
    yield
    # Clear JAX compilation cache to prevent memory issues
    jax.clear_caches()


def pytest_runtest_makereport(item, call):
    """Log CUDA-related test failures via logging instead of print."""
    if call.when == "call" and call.excinfo and "CUDA" in str(call.excinfo.value):
        import logging

        logger = logging.getLogger("opifex.tests")
        env = _init_test_environment()
        logger.warning(
            "CUDA-related test failure: %s (JAX_PLATFORMS=%s, gpu_safe=%s)",
            item.nodeid,
            os.environ.get("JAX_PLATFORMS", "cpu"),
            getattr(env, "gpu_safe", None),
        )


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_session():
    """Clean up test session resources."""
    yield
    jax.clear_caches()
