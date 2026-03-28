"""Opifex Testing Infrastructure - Dependency Management and Test Utilities.

Environment configuration (JAX backend, x64, memory) is handled entirely by
setup.sh / activate.sh / setup_env.py / pytest-env.  This module provides:

1. Optional dependency management with graceful degradation
2. Mock implementations for unavailable optional dependencies
3. Dataclasses for test environment metadata
"""

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BackendType(Enum):
    """Available backend types for testing."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class DependencyStatus(Enum):
    """Status of optional dependencies."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MOCK = "mock"
    MISSING = "missing"


class EnvironmentType(Enum):
    """GPU environment classification."""

    GPU_SAFE = "gpu_safe"
    GPU_AVAILABLE_UNSAFE = "gpu_unsafe"
    CPU_ONLY = "cpu_only"
    UNKNOWN = "unknown"


class CompilationStrategy(Enum):
    """Available compilation strategies for neural operator models."""

    SAFE_JIT = "safe_jit"
    NO_JIT = "no_jit"
    EAGER = "eager"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CUDAEnvironment:
    """Local CUDA environment configuration."""

    venv_cuda_available: bool = False
    cuda_library_paths: list[str] | None = None
    jax_cuda_available: bool = False
    gpu_devices_detected: int = 0
    cuda_version: str | None = None
    environment_variables: dict[str, str] | None = None

    def __post_init__(self):
        if self.cuda_library_paths is None:
            self.cuda_library_paths = []
        if self.environment_variables is None:
            self.environment_variables = {}


@dataclass
class TestEnvironment:
    """Test environment configuration."""

    backend: BackendType
    environment_type: EnvironmentType
    gpu_available: bool
    gpu_safe: bool
    cuda_env: CUDAEnvironment
    dependencies: dict[str, DependencyStatus]
    memory_limit_mb: int | None = None
    process_isolation: bool = False


@dataclass
class GPUTestResult:
    """Result of GPU stability testing."""

    is_stable: bool
    error_message: str | None = None
    max_safe_memory_mb: int | None = None
    test_duration_ms: float = 0.0


@dataclass
class CompilationResult:
    """Result of model compilation attempt."""

    success: bool
    strategy_used: CompilationStrategy
    error_message: str | None = None
    fallback_applied: bool = False
    compilation_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------


def _check_dependency(name: str) -> DependencyStatus:
    """Check whether a single optional dependency is importable."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return DependencyStatus.AVAILABLE
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    return DependencyStatus.MISSING


class DependencyManager:
    """Manages optional dependencies with graceful degradation.

    Provides mock implementations when dependencies are unavailable.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dependencies: dict[str, DependencyStatus] = {}
        self.mocks: dict[str, Any] = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check availability of optional dependencies."""
        for name in ("prometheus_client", "psutil"):
            self.dependencies[name] = _check_dependency(name)

    def get_dependency_status(self, dependency: str) -> DependencyStatus:
        """Get the status of a dependency."""
        return self.dependencies.get(dependency, DependencyStatus.UNAVAILABLE)

    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        return self.get_dependency_status(dependency) == DependencyStatus.AVAILABLE

    def register_mock(self, dependency: str, mock_implementation: Any) -> None:
        """Register a mock implementation for a dependency."""
        self.mocks[dependency] = mock_implementation
        self.dependencies[dependency] = DependencyStatus.MOCK
        self.logger.debug(f"Registered mock for {dependency}")

    def get_implementation(self, dependency: str) -> Any:
        """Get the implementation (real or mock) for a dependency."""
        status = self.get_dependency_status(dependency)

        if status == DependencyStatus.AVAILABLE:
            if dependency == "prometheus_client":
                import prometheus_client

                return prometheus_client
            if dependency == "psutil":
                import psutil  # type: ignore[import-untyped]

                return psutil
        elif status == DependencyStatus.MOCK:
            return self.mocks[dependency]

        raise ImportError(f"Dependency {dependency} not available and no mock registered")


def requires_dependency(dependency: str, mock_implementation: Any | None = None):
    """Decorator to handle optional dependencies with mocking support.

    Args:
        dependency: Name of the required dependency
        mock_implementation: Optional mock to use if dependency unavailable
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dep_manager = DependencyManager()

            if not dep_manager.is_available(dependency):
                if mock_implementation:
                    dep_manager.register_mock(dependency, mock_implementation)
                else:
                    pytest = __import__("pytest")
                    pytest.skip(f"Dependency {dependency} not available")

            return func(*args, **kwargs)

        return wrapper

    return decorator


class MockMetricsImplementation:
    """Mock implementation for Prometheus metrics when unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def labels(self, **kwargs):
        """Set labels for metrics (mock implementation)."""
        return self

    def set(self, value):
        """Set metric value (mock implementation)."""

    def observe(self, value):
        """Observe metric value (mock implementation)."""

    def inc(self, value=1):
        """Increment metric by value (mock implementation)."""

    def get_metrics_data(self) -> str:
        """Get metrics data in string format.

        Returns:
            str: Formatted metrics data
        """
        return "Prometheus metrics not available"

    def health_check(self) -> dict[str, Any]:
        """Perform health check (mock implementation)."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics_available": False,
            "message": "Mock metrics implementation active",
        }


def get_dependency_manager() -> DependencyManager:
    """Get a dependency manager instance."""
    return DependencyManager()
