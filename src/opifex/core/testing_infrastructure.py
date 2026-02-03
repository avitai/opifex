"""
Opifex Testing Infrastructure - Comprehensive GPU Safety and Dependency Management

This module provides robust testing infrastructure that handles:
1. GPU segmentation fault prevention and recovery
2. Optional dependency management with graceful degradation
3. Hardware-aware test execution
4. Comprehensive error handling and diagnostics
5. Local .venv CUDA environment detection and management

Key Features:
- Proactive GPU environment validation before JAX initialization
- Local .venv CUDA library detection and configuration
- Automatic JAX backend configuration based on environment safety
- Process isolation for GPU-intensive operations
- Comprehensive test environment management
"""

import functools
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import jax


# Import JAX after environment configuration
_jax_configured = False

# Type definitions
T = TypeVar("T")
TestFunction = Callable[..., Any]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    GPU_SAFE = "gpu_safe"  # GPU available and stable
    GPU_AVAILABLE_UNSAFE = "gpu_unsafe"  # GPU present but causes segfaults
    CPU_ONLY = "cpu_only"  # No GPU or GPU libraries unavailable
    UNKNOWN = "unknown"  # Environment not yet assessed


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
    """Enhanced test environment configuration."""

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


class EnvironmentDetector:
    """
    Comprehensive environment detection for local .venv CUDA setup.

    This class detects and validates the local CUDA environment before
    any JAX operations to prevent segmentation faults.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.project_root = Path(__file__).parent.parent.parent
        self.venv_path = self.project_root / ".venv"

    def detect_cuda_environment(self) -> CUDAEnvironment:
        """
        Detect local .venv CUDA environment configuration.

        Returns:
            CUDAEnvironment with detected configuration
        """
        cuda_env = CUDAEnvironment()

        # Check if .venv exists
        if not self.venv_path.exists():
            self.logger.warning("No .venv directory found")
            return cuda_env

        # Detect CUDA library paths in .venv
        cuda_env.cuda_library_paths = self._find_cuda_libraries()
        cuda_env.venv_cuda_available = len(cuda_env.cuda_library_paths) > 0

        # Check for JAX CUDA support
        cuda_env.jax_cuda_available = self._check_jax_cuda_support()

        # Detect GPU hardware
        cuda_env.gpu_devices_detected = self._detect_gpu_hardware()

        # Get CUDA version if available
        cuda_env.cuda_version = self._get_cuda_version()

        # Prepare environment variables
        cuda_env.environment_variables = self._prepare_environment_variables(cuda_env)

        return cuda_env

    def _find_cuda_libraries(self) -> list[str]:
        """Find CUDA libraries in .venv."""
        cuda_paths: list[str] = []
        nvidia_path = self.venv_path / "lib" / "python3.12" / "site-packages" / "nvidia"

        if not nvidia_path.exists():
            return cuda_paths

        # Check for key CUDA libraries
        cuda_libs = [
            "cublas",
            "cusolver",
            "cusparse",
            "cudnn",
            "cufft",
            "curand",
            "nccl",
            "nvjitlink",
        ]

        for lib in cuda_libs:
            lib_path = nvidia_path / lib / "lib"
            if lib_path.exists():
                cuda_paths.append(str(lib_path))

        return cuda_paths

    def _check_jax_cuda_support(self) -> bool:
        """Check if JAX with CUDA support is installed."""
        try:
            # Check if jax-cuda is in the environment
            # Fix: Check for 'gpu' platform instead of 'cuda' since that's what JAX uses
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import jax; print('gpu' in [d.platform for d in jax.devices()])",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and "True" in result.stdout
        except Exception as e:
            self.logger.debug(f"JAX CUDA check failed: {e}")
            return False

    def _detect_gpu_hardware(self) -> int:
        """Detect available GPU hardware."""
        try:
            # Use nvidia-smi to detect GPUs
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return len([line for line in result.stdout.split("\n") if line.strip()])
            return 0
        except Exception:
            return 0

    def _get_cuda_version(self) -> str | None:
        """Get CUDA version if available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return None

    def _prepare_environment_variables(
        self, cuda_env: CUDAEnvironment
    ) -> dict[str, str]:
        """Prepare environment variables for CUDA setup."""
        env_vars = {}

        if cuda_env.venv_cuda_available and cuda_env.cuda_library_paths:
            # Set LD_LIBRARY_PATH for local CUDA libraries
            ld_path = ":".join(cuda_env.cuda_library_paths)
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if existing_ld:
                ld_path = f"{ld_path}:{existing_ld}"
            env_vars["LD_LIBRARY_PATH"] = ld_path

            # JAX CUDA configuration
            if cuda_env.jax_cuda_available and cuda_env.gpu_devices_detected > 0:
                env_vars.update(
                    {
                        "JAX_PLATFORMS": "cuda,cpu",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
                        "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",
                        "JAX_CUDA_PLUGIN_VERIFY": "false",
                        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
                        "TF_CPP_MIN_LOG_LEVEL": "1",
                    }
                )
            else:
                # Force CPU if GPU not properly available
                env_vars["JAX_PLATFORMS"] = "cpu"
        else:
            # CPU-only configuration
            env_vars["JAX_PLATFORMS"] = "cpu"

        return env_vars


class JAXConfigurationManager:
    """
    Manages JAX configuration for safe testing across environments.

    This class ensures JAX is configured appropriately before any
    neural operator code is imported or executed.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._configured = False

    def configure_jax_for_environment(
        self, cuda_env: CUDAEnvironment
    ) -> EnvironmentType:
        """
        Configure JAX based on detected CUDA environment.

        Args:
            cuda_env: Detected CUDA environment configuration

        Returns:
            EnvironmentType classification of the environment
        """
        if self._configured:
            return self._get_current_environment_type()

        # Apply environment variables
        if cuda_env.environment_variables:
            for key, value in cuda_env.environment_variables.items():
                os.environ[key] = value
                self.logger.debug(f"Set {key}={value}")

        # Determine environment type and configure accordingly
        env_type = self._classify_environment(cuda_env)

        if env_type == EnvironmentType.GPU_SAFE:
            self._configure_gpu_safe()
        elif env_type == EnvironmentType.GPU_AVAILABLE_UNSAFE:
            self._configure_gpu_unsafe_fallback()
        else:
            self._configure_cpu_only()

        self._configured = True
        self.logger.info(f"JAX configured for {env_type.value} environment")

        return env_type

    def _classify_environment(self, cuda_env: CUDAEnvironment) -> EnvironmentType:
        """Classify the environment type based on CUDA configuration."""
        if not cuda_env.venv_cuda_available or not cuda_env.jax_cuda_available:
            return EnvironmentType.CPU_ONLY

        if cuda_env.gpu_devices_detected == 0:
            return EnvironmentType.CPU_ONLY

        # Test GPU stability
        if self._test_gpu_basic_stability():
            return EnvironmentType.GPU_SAFE
        return EnvironmentType.GPU_AVAILABLE_UNSAFE

    def _test_gpu_basic_stability(self) -> bool:
        """Test basic GPU stability without triggering JAX compilation."""
        try:
            # Import JAX after environment configuration
            import jax
            import jax.numpy as jnp

            # Simple device check
            devices = jax.devices()

            # Fix: Check for GPU platform instead of 'cuda' device_kind
            # device_kind contains the actual GPU model name, not 'cuda'
            gpu_devices = [d for d in devices if d.platform == "gpu"]

            if not gpu_devices:
                return False

            # Very basic operation without JIT compilation
            with jax.default_device(gpu_devices[0]):
                x = jnp.array([1.0, 2.0, 3.0])
                y = jnp.array([4.0, 5.0, 6.0])
                _ = x + y  # Simple operation, no compilation

            return True

        except Exception as e:
            self.logger.warning(f"GPU stability test failed: {e}")
            return False

    def _configure_gpu_safe(self):
        """Configure JAX for safe GPU usage."""
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        os.environ["JAX_ENABLE_X64"] = (
            "True"  # Enable x64 precision for numerical accuracy
        )

    def _configure_gpu_unsafe_fallback(self):
        """Configure JAX to fallback to CPU for unsafe GPU."""
        os.environ["JAX_PLATFORMS"] = "cpu"
        self.logger.warning("GPU detected but unstable, falling back to CPU")

    def _configure_cpu_only(self):
        """Configure JAX for CPU-only operation."""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_ENABLE_X64"] = "True"

    def _get_current_environment_type(self) -> EnvironmentType:
        """Get current environment type based on JAX configuration."""
        platforms = os.environ.get("JAX_PLATFORMS", "cpu")
        if "cuda" in platforms:
            return EnvironmentType.GPU_SAFE
        return EnvironmentType.CPU_ONLY


# Global environment management
class _SingletonManager:
    """Singleton manager to avoid global statement warnings."""

    _environment_detector: EnvironmentDetector | None = None
    _jax_config_manager: JAXConfigurationManager | None = None
    _current_environment: TestEnvironment | None = None

    @classmethod
    def get_environment_detector(cls) -> EnvironmentDetector:
        """Get the singleton environment detector."""
        if cls._environment_detector is None:
            cls._environment_detector = EnvironmentDetector()
        return cls._environment_detector

    @classmethod
    def get_jax_config_manager(cls) -> JAXConfigurationManager:
        """Get the singleton JAX configuration manager."""
        if cls._jax_config_manager is None:
            cls._jax_config_manager = JAXConfigurationManager()
        return cls._jax_config_manager

    @classmethod
    def get_current_environment(cls) -> TestEnvironment | None:
        """Get the current test environment."""
        return cls._current_environment

    @classmethod
    def set_current_environment(cls, environment: TestEnvironment) -> None:
        """Set the current test environment."""
        cls._current_environment = environment


def get_environment_detector() -> EnvironmentDetector:
    """Get the global environment detector."""
    return _SingletonManager.get_environment_detector()


def get_jax_config_manager() -> JAXConfigurationManager:
    """Get the global JAX configuration manager."""
    return _SingletonManager.get_jax_config_manager()


def ensure_safe_jax_environment() -> TestEnvironment:
    """
    Ensure JAX is configured for safe testing.

    This function should be called before any JAX operations to prevent
    segmentation faults in unstable GPU environments.

    Returns:
        TestEnvironment with safe configuration
    """
    current_env = _SingletonManager.get_current_environment()
    if current_env is not None:
        return current_env

    # Detect CUDA environment
    detector = get_environment_detector()
    cuda_env = detector.detect_cuda_environment()

    # Configure JAX based on environment
    config_manager = get_jax_config_manager()
    env_type = config_manager.configure_jax_for_environment(cuda_env)

    # Import JAX after configuration

    # Create test environment
    test_environment = TestEnvironment(
        backend=BackendType.GPU
        if env_type == EnvironmentType.GPU_SAFE
        else BackendType.CPU,
        environment_type=env_type,
        gpu_available=cuda_env.gpu_devices_detected > 0,
        gpu_safe=env_type == EnvironmentType.GPU_SAFE,
        cuda_env=cuda_env,
        dependencies=_check_dependencies(),
    )

    _SingletonManager.set_current_environment(test_environment)
    return test_environment


def _check_dependencies() -> dict[str, DependencyStatus]:
    """Check status of optional dependencies."""
    dependencies = {}

    # Check Prometheus client
    try:
        import importlib.util

        spec = importlib.util.find_spec("prometheus_client")
        if spec is not None:
            dependencies["prometheus_client"] = DependencyStatus.AVAILABLE
        else:
            dependencies["prometheus_client"] = DependencyStatus.MISSING
    except ImportError:
        dependencies["prometheus_client"] = DependencyStatus.MISSING

    # Check psutil
    try:
        import importlib.util

        spec = importlib.util.find_spec("psutil")
        if spec is not None:
            dependencies["psutil"] = DependencyStatus.AVAILABLE
        else:
            dependencies["psutil"] = DependencyStatus.MISSING
    except ImportError:
        dependencies["psutil"] = DependencyStatus.MISSING

    return dependencies


# Simplified GPU testing interface
class GPUStabilityTester:
    """Simplified GPU stability tester with clean API."""

    def __init__(self, max_test_time_seconds: float = 10.0):
        self.max_test_time_seconds = max_test_time_seconds
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def test_gpu_stability(self) -> GPUTestResult:
        """Test GPU stability using new environment management."""
        start_time = time.time()

        try:
            env = ensure_safe_jax_environment()
            duration_ms = (time.time() - start_time) * 1000

            return GPUTestResult(
                is_stable=env.gpu_safe,
                error_message=None if env.gpu_safe else "GPU environment unsafe",
                max_safe_memory_mb=8192 if env.gpu_safe else None,
                test_duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return GPUTestResult(
                is_stable=False,
                error_message=str(e),
                test_duration_ms=duration_ms,
            )


class DependencyManager:
    """
    Manages optional dependencies with graceful degradation.

    Provides mock implementations when dependencies are unavailable.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.dependencies = {}
        self.mocks = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check availability of optional dependencies."""
        # Check Prometheus client
        try:
            import importlib.util

            spec = importlib.util.find_spec("prometheus_client")
            if spec is not None:
                self.dependencies["prometheus_client"] = DependencyStatus.AVAILABLE
            else:
                self.dependencies["prometheus_client"] = DependencyStatus.MISSING
        except ImportError:
            self.dependencies["prometheus_client"] = DependencyStatus.MISSING

        # Check psutil
        try:
            import importlib.util

            spec = importlib.util.find_spec("psutil")
            if spec is not None:
                self.dependencies["psutil"] = DependencyStatus.AVAILABLE
            else:
                self.dependencies["psutil"] = DependencyStatus.MISSING
        except ImportError:
            self.dependencies["psutil"] = DependencyStatus.MISSING

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

        raise ImportError(
            f"Dependency {dependency} not available and no mock registered"
        )


class TestEnvironmentManager:
    """
    Enhanced test environment manager with proactive GPU safety.

    This class manages the overall test environment, ensuring safe
    execution across different hardware configurations.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._environment = None

    def get_test_environment(self, force_refresh: bool = False) -> TestEnvironment:
        """
        Get the current test environment with proactive safety measures.

        Args:
            force_refresh: Force re-assessment of the environment

        Returns:
            TestEnvironment with safe configuration
        """
        if self._environment is None or force_refresh:
            self._environment = ensure_safe_jax_environment()

        return self._environment


def requires_dependency(dependency: str, mock_implementation: Any | None = None):
    """
    Decorator to handle optional dependencies with mocking support.

    Args:
        dependency: Name of the required dependency
        mock_implementation: Optional mock to use if dependency unavailable
    """

    def decorator(func: TestFunction) -> TestFunction:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get environment and dependency manager
            ensure_safe_jax_environment()
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
        """
        Get metrics data in string format.

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


def get_test_environment_manager() -> TestEnvironmentManager:
    """Get the global test environment manager."""
    return TestEnvironmentManager()


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager."""
    return DependencyManager()


# =============================================================================
# MODEL SAFETY FRAMEWORK
# =============================================================================


class CompilationStrategy(Enum):
    """Available compilation strategies for neural operator models."""

    SAFE_JIT = "safe_jit"  # Safe JIT with fallback
    NO_JIT = "no_jit"  # No JIT compilation
    EAGER = "eager"  # Eager execution only
    AUTO = "auto"  # Automatic strategy selection


@dataclass
class CompilationResult:
    """Result of model compilation attempt."""

    success: bool
    strategy_used: CompilationStrategy
    error_message: str | None = None
    fallback_applied: bool = False
    compilation_time_ms: float = 0.0


class SafeJITCompiler:
    """
    Safe JIT compiler with automatic fallback mechanisms.

    This class provides safe JIT compilation for neural operator models,
    automatically falling back to safer strategies when compilation fails.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._compilation_cache = {}

    def safe_jit(
        self, func: Callable, strategy: CompilationStrategy = CompilationStrategy.AUTO
    ) -> Callable:
        """
        Apply safe JIT compilation with fallback mechanisms.

        Args:
            func: Function to compile
            strategy: Compilation strategy to use

        Returns:
            Safely compiled function
        """
        # Get current environment
        env = ensure_safe_jax_environment()

        # Determine compilation strategy
        if strategy == CompilationStrategy.AUTO:
            strategy = self._select_strategy(env)

        return self._apply_strategy(func, strategy, env)

    def _select_strategy(self, env: TestEnvironment) -> CompilationStrategy:
        """Select optimal compilation strategy based on environment."""
        # For GPU unsafe environments, use NO_JIT for safety
        if (
            hasattr(env, "environment_type")
            and env.environment_type == EnvironmentType.GPU_AVAILABLE_UNSAFE
        ):
            return CompilationStrategy.NO_JIT

        # For unknown environments, use NO_JIT for safety
        if (
            hasattr(env, "environment_type")
            and env.environment_type == EnvironmentType.UNKNOWN
        ):
            return CompilationStrategy.NO_JIT

        # For CPU only environments, use SAFE_JIT (compatible with CPU)
        if (
            hasattr(env, "environment_type")
            and env.environment_type == EnvironmentType.CPU_ONLY
        ):
            return CompilationStrategy.SAFE_JIT

        # Default to SAFE_JIT for stable GPU environments
        return CompilationStrategy.SAFE_JIT

    def _apply_strategy(
        self, func: Callable, strategy: CompilationStrategy, env: TestEnvironment
    ) -> Callable:
        """Apply the selected compilation strategy."""
        try:
            if strategy == CompilationStrategy.SAFE_JIT:
                # Apply JAX JIT directly using device-agnostic approach
                try:
                    from flax import nnx

                    return nnx.jit(func)
                except ImportError:
                    import jax

                    return jax.jit(func)
            if strategy == CompilationStrategy.NO_JIT:
                return self._no_jit_wrapper(func)
            if strategy == CompilationStrategy.EAGER:
                return self._eager_wrapper(func)
            # Default to no JIT for safety
            return self._no_jit_wrapper(func)

        except Exception as e:
            self.logger.warning(
                f"Compilation failed with {strategy.value}, falling back to no-JIT: {e}"
            )
            return self._no_jit_wrapper(func)

    def _no_jit_wrapper(self, func: Callable) -> Callable:
        """Return function without JIT compilation."""
        # No wrapping needed - just return the function as-is
        return func

    def _eager_wrapper(self, func: Callable) -> Callable:
        """Create an eager execution wrapper."""

        @functools.wraps(func)
        def eager_wrapper(*args, **kwargs):
            # Force eager execution using JAX's device-agnostic approach
            with jax.disable_jit():
                return func(*args, **kwargs)

        return eager_wrapper


# Model safety functionality moved to opifex.core.model_safety module
