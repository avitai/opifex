"""Opifex Testing Configuration.

Lightweight pytest configuration that defers JAX initialization and device
probing to test execution time, keeping test collection fast and side-effect
free.  Environment variables are managed by activate.sh / .opifex.env / .env
and pytest-env in pyproject.toml — this module does NOT mutate them.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root on sys.path (needed for editable installs to resolve)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp


# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="jax")

# ---------------------------------------------------------------------------
# Lazy imports — resolved at first use, not at collection time
# ---------------------------------------------------------------------------
_dependency_manager = None


def get_dependency_manager():
    """Get the global dependency manager, initializing lazily."""
    global _dependency_manager  # noqa: PLW0603
    if _dependency_manager is None:
        try:
            from opifex.core.testing_infrastructure import DependencyManager

            _dependency_manager = DependencyManager()
        except ImportError:
            pass
    return _dependency_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Provide a device fixture that works with both CPU and GPU."""
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
        return jax.random.PRNGKey(0)


@pytest.fixture(scope="session")
def dependency_manager():
    """Provide the dependency manager for tests."""
    return get_dependency_manager()


@pytest.fixture(scope="session")
def test_environment():
    """Provide a minimal test environment description."""
    from opifex.core.testing_infrastructure import (
        BackendType,
        CUDAEnvironment,
        DependencyStatus,
        EnvironmentType,
        TestEnvironment,
    )

    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    return TestEnvironment(
        backend=BackendType.GPU if has_gpu else BackendType.CPU,
        environment_type=EnvironmentType.GPU_SAFE if has_gpu else EnvironmentType.CPU_ONLY,
        gpu_available=has_gpu,
        gpu_safe=has_gpu,
        cuda_env=CUDAEnvironment(),
        dependencies={
            name: DependencyStatus.AVAILABLE
            for name in ("prometheus_client", "psutil")
            if (mgr := get_dependency_manager()) and mgr.is_available(name)
        },
    )


@pytest.fixture
def mock_prometheus_metrics():
    """Provide a mock Prometheus metrics implementation."""
    from opifex.core.testing_infrastructure import MockMetricsImplementation

    return MockMetricsImplementation()


@pytest.fixture
def safe_context():
    """Provide a safe JAX context for tests (no-op)."""
    from contextlib import nullcontext

    return nullcontext()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):
    """Set up for each test run - ensure clean JAX state."""
    try:
        jax.clear_caches()

        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                import gc

                gc.collect()

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
        jax.clear_caches()

        import gc

        gc.collect()

        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                try:
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

    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)

    if dep_manager is None:
        return

    for item in items:
        if item.get_closest_marker("requires_prometheus") and not dep_manager.is_available(
            "prometheus_client"
        ):
            item.add_marker(pytest.mark.skip(reason="Prometheus client not available"))

        if item.get_closest_marker("requires_psutil") and not dep_manager.is_available("psutil"):
            item.add_marker(pytest.mark.skip(reason="psutil not available"))


def pytest_runtest_makereport(item, call):
    """Log CUDA-related test failures via logging instead of print."""
    if call.when == "call" and call.excinfo and "CUDA" in str(call.excinfo.value):
        import logging

        test_logger = logging.getLogger("opifex.tests")
        test_logger.warning(
            "CUDA-related test failure: %s (JAX_PLATFORMS=%s)",
            item.nodeid,
            os.environ.get("JAX_PLATFORMS", "cpu"),
        )


@pytest.fixture(scope="function")
def reset_jax_config():
    """Reset JAX configuration after each test."""
    yield
    jax.clear_caches()


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_session():
    """Clean up test session resources."""
    yield
    jax.clear_caches()
