"""Tests for opifex.core.testing_infrastructure module.

Tests dependency management, mock implementations, dataclasses, and enums.
Environment detection / JAX configuration is handled by setup.sh + activate.sh.
"""

from unittest.mock import Mock, patch

import pytest

from opifex.core.testing_infrastructure import (
    BackendType,
    CompilationResult,
    CompilationStrategy,
    CUDAEnvironment,
    DependencyManager,
    DependencyStatus,
    EnvironmentType,
    get_dependency_manager,
    GPUTestResult,
    MockMetricsImplementation,
    requires_dependency,
    TestEnvironment,
)


class TestEnums:
    """Test enum definitions."""

    def test_backend_type_enum(self):
        """Test BackendType enum values."""
        assert BackendType.CPU.value == "cpu"
        assert BackendType.GPU.value == "gpu"
        assert BackendType.AUTO.value == "auto"
        assert len(BackendType) == 3

    def test_dependency_status_enum(self):
        """Test DependencyStatus enum values."""
        assert DependencyStatus.AVAILABLE.value == "available"
        assert DependencyStatus.UNAVAILABLE.value == "unavailable"
        assert DependencyStatus.MOCK.value == "mock"
        assert DependencyStatus.MISSING.value == "missing"
        assert len(DependencyStatus) == 4

    def test_environment_type_enum(self):
        """Test EnvironmentType enum values."""
        assert EnvironmentType.GPU_SAFE.value == "gpu_safe"
        assert EnvironmentType.GPU_AVAILABLE_UNSAFE.value == "gpu_unsafe"
        assert EnvironmentType.CPU_ONLY.value == "cpu_only"
        assert EnvironmentType.UNKNOWN.value == "unknown"
        assert len(EnvironmentType) == 4

    def test_compilation_strategy_enum(self):
        """Test CompilationStrategy enum values."""
        assert CompilationStrategy.SAFE_JIT.value == "safe_jit"
        assert CompilationStrategy.NO_JIT.value == "no_jit"
        assert CompilationStrategy.EAGER.value == "eager"
        assert CompilationStrategy.AUTO.value == "auto"
        assert len(CompilationStrategy) == 4


class TestDataclasses:
    """Test dataclass definitions."""

    def test_cuda_environment_default_initialization(self):
        """Test CUDAEnvironment default initialization."""
        cuda_env = CUDAEnvironment()

        assert cuda_env.venv_cuda_available is False
        assert cuda_env.cuda_library_paths == []
        assert cuda_env.jax_cuda_available is False
        assert cuda_env.gpu_devices_detected == 0
        assert cuda_env.cuda_version is None
        assert cuda_env.environment_variables == {}

    def test_cuda_environment_custom_initialization(self):
        """Test CUDAEnvironment custom initialization."""
        cuda_paths = ["/path/to/cuda1", "/path/to/cuda2"]
        env_vars = {"CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORM_NAME": "gpu"}

        cuda_env = CUDAEnvironment(
            venv_cuda_available=True,
            cuda_library_paths=cuda_paths,
            jax_cuda_available=True,
            gpu_devices_detected=2,
            cuda_version="12.0",
            environment_variables=env_vars,
        )

        assert cuda_env.venv_cuda_available is True
        assert cuda_env.cuda_library_paths == cuda_paths
        assert cuda_env.jax_cuda_available is True
        assert cuda_env.gpu_devices_detected == 2
        assert cuda_env.cuda_version == "12.0"
        assert cuda_env.environment_variables == env_vars

    def test_cuda_environment_post_init(self):
        """Test CUDAEnvironment post-init handling of None values."""
        cuda_env = CUDAEnvironment(cuda_library_paths=None, environment_variables=None)

        assert cuda_env.cuda_library_paths == []
        assert cuda_env.environment_variables == {}

    def test_test_environment_initialization(self):
        """Test TestEnvironment initialization."""
        cuda_env = CUDAEnvironment()
        dependencies = {"dep1": DependencyStatus.AVAILABLE}

        test_env = TestEnvironment(
            backend=BackendType.GPU,
            environment_type=EnvironmentType.GPU_SAFE,
            gpu_available=True,
            gpu_safe=True,
            cuda_env=cuda_env,
            dependencies=dependencies,
            memory_limit_mb=8192,
            process_isolation=True,
        )

        assert test_env.backend == BackendType.GPU
        assert test_env.environment_type == EnvironmentType.GPU_SAFE
        assert test_env.gpu_available is True
        assert test_env.gpu_safe is True
        assert test_env.cuda_env is cuda_env
        assert test_env.dependencies == dependencies
        assert test_env.memory_limit_mb == 8192
        assert test_env.process_isolation is True

    def test_gpu_test_result_initialization(self):
        """Test GPUTestResult initialization."""
        result = GPUTestResult(
            is_stable=True,
            error_message="Test error",
            max_safe_memory_mb=4096,
            test_duration_ms=150.5,
        )

        assert result.is_stable is True
        assert result.error_message == "Test error"
        assert result.max_safe_memory_mb == 4096
        assert result.test_duration_ms == 150.5

    def test_compilation_result_initialization(self):
        """Test CompilationResult initialization."""
        result = CompilationResult(
            success=True,
            strategy_used=CompilationStrategy.SAFE_JIT,
            error_message=None,
            fallback_applied=False,
            compilation_time_ms=75.2,
        )

        assert result.success is True
        assert result.strategy_used == CompilationStrategy.SAFE_JIT
        assert result.error_message is None
        assert result.fallback_applied is False
        assert result.compilation_time_ms == 75.2


class TestDependencyManager:
    """Test DependencyManager class."""

    def test_dependency_manager_initialization(self):
        """Test DependencyManager initialization."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()

            assert hasattr(manager, "logger")
            assert hasattr(manager, "dependencies")
            assert hasattr(manager, "mocks")

    @patch("importlib.util.find_spec")
    def test_check_dependencies_prometheus_available(self, mock_find_spec):
        """Test dependency check when prometheus_client is available."""
        mock_spec = Mock()
        mock_find_spec.return_value = mock_spec

        manager = DependencyManager()

        assert manager.dependencies["prometheus_client"] == DependencyStatus.AVAILABLE

    @patch("importlib.util.find_spec")
    def test_check_dependencies_prometheus_missing(self, mock_find_spec):
        """Test dependency check when prometheus_client is missing."""
        mock_find_spec.return_value = None

        manager = DependencyManager()

        assert manager.dependencies["prometheus_client"] == DependencyStatus.MISSING

    @patch("importlib.util.find_spec")
    def test_check_dependencies_import_error(self, mock_find_spec):
        """Test dependency check when ImportError occurs."""
        mock_find_spec.side_effect = ImportError("Module not found")

        manager = DependencyManager()

        assert manager.dependencies["prometheus_client"] == DependencyStatus.MISSING

    def test_get_dependency_status_existing(self):
        """Test getting status of existing dependency."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()
            manager.dependencies = {"test_dep": DependencyStatus.AVAILABLE}

            status = manager.get_dependency_status("test_dep")
            assert status == DependencyStatus.AVAILABLE

    def test_get_dependency_status_non_existing(self):
        """Test getting status of non-existing dependency."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()

            status = manager.get_dependency_status("non_existent")
            assert status == DependencyStatus.UNAVAILABLE

    def test_is_available_true(self):
        """Test is_available when dependency is available."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()
            manager.dependencies = {"test_dep": DependencyStatus.AVAILABLE}

            assert manager.is_available("test_dep") is True

    def test_is_available_false(self):
        """Test is_available when dependency is not available."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()
            manager.dependencies = {"test_dep": DependencyStatus.MISSING}

            assert manager.is_available("test_dep") is False

    def test_register_mock(self):
        """Test registering a mock implementation."""
        with patch.object(DependencyManager, "_check_dependencies"):
            manager = DependencyManager()
            mock_impl = Mock()

            manager.register_mock("test_dep", mock_impl)

            assert manager.mocks["test_dep"] is mock_impl
            assert manager.dependencies["test_dep"] == DependencyStatus.MOCK

    @patch("importlib.util.find_spec")
    def test_get_implementation_available_prometheus(self, mock_find_spec):
        """Test getting prometheus implementation when available."""
        mock_find_spec.return_value = Mock()

        manager = DependencyManager()
        manager.dependencies["prometheus_client"] = DependencyStatus.AVAILABLE

        implementation = manager.get_implementation("prometheus_client")
        assert implementation is not None

    def test_get_implementation_mock(self):
        """Test getting mock implementation."""
        manager = DependencyManager()
        mock_impl = MockMetricsImplementation()
        manager.register_mock("test_dep", mock_impl)

        implementation = manager.get_implementation("test_dep")
        assert implementation is mock_impl

    def test_get_implementation_unavailable(self):
        """Test getting implementation when unavailable."""
        manager = DependencyManager()
        manager.dependencies["missing_dep"] = DependencyStatus.UNAVAILABLE

        with pytest.raises(ImportError, match="Dependency missing_dep not available"):
            manager.get_implementation("missing_dep")


class TestMockMetricsImplementation:
    """Test MockMetricsImplementation class."""

    def test_mock_metrics_initialization(self):
        """Test MockMetricsImplementation initialization."""
        mock_metrics = MockMetricsImplementation("test", "description")
        assert isinstance(mock_metrics, MockMetricsImplementation)

    def test_mock_metrics_labels(self):
        """Test labels method."""
        mock_metrics = MockMetricsImplementation()
        result = mock_metrics.labels(test="value")
        assert result is mock_metrics

    def test_mock_metrics_set(self):
        """Test set method."""
        mock_metrics = MockMetricsImplementation()
        mock_metrics.set(42)

    def test_mock_metrics_observe(self):
        """Test observe method."""
        mock_metrics = MockMetricsImplementation()
        mock_metrics.observe(1.5)

    def test_mock_metrics_inc(self):
        """Test inc method."""
        mock_metrics = MockMetricsImplementation()
        mock_metrics.inc()
        mock_metrics.inc(5)

    def test_mock_metrics_get_metrics_data(self):
        """Test get_metrics_data method."""
        mock_metrics = MockMetricsImplementation()
        data = mock_metrics.get_metrics_data()
        assert data == "Prometheus metrics not available"

    def test_mock_metrics_health_check(self):
        """Test health_check method."""
        mock_metrics = MockMetricsImplementation()
        health = mock_metrics.health_check()

        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert health["metrics_available"] is False
        assert "message" in health


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_dependency_manager(self):
        """Test get_dependency_manager function."""
        manager = get_dependency_manager()
        assert isinstance(manager, DependencyManager)


class TestDecorators:
    """Test decorator functions."""

    @patch("opifex.core.testing_infrastructure.DependencyManager")
    def test_requires_dependency_decorator_available(self, mock_dep_manager_class):
        """Test requires_dependency decorator with available dependency."""
        mock_dep_manager = Mock()
        mock_dep_manager.is_available.return_value = True
        mock_dep_manager_class.return_value = mock_dep_manager

        @requires_dependency("test_dep")
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"
        mock_dep_manager.is_available.assert_called_once_with("test_dep")

    @patch("opifex.core.testing_infrastructure.DependencyManager")
    def test_requires_dependency_decorator_unavailable_with_mock(self, mock_dep_manager_class):
        """Test requires_dependency decorator with unavailable dependency but mock provided."""
        mock_dep_manager = Mock()
        mock_dep_manager.is_available.return_value = False
        mock_dep_manager_class.return_value = mock_dep_manager

        mock_implementation = Mock()

        @requires_dependency("test_dep", mock_implementation)
        def test_function():
            return "success with mock"

        result = test_function()
        assert result == "success with mock"
        mock_dep_manager.is_available.assert_called_once_with("test_dep")
        mock_dep_manager.register_mock.assert_called_once_with("test_dep", mock_implementation)

    @patch("opifex.core.testing_infrastructure.DependencyManager")
    def test_requires_dependency_decorator_unavailable_skip(self, mock_dep_manager_class):
        """Test requires_dependency decorator with unavailable dependency - should skip."""
        mock_dep_manager = Mock()
        mock_dep_manager.is_available.return_value = False
        mock_dep_manager_class.return_value = mock_dep_manager

        @requires_dependency("test_dep")
        def test_function():
            return "should not execute"

        with pytest.raises(pytest.skip.Exception):
            test_function()
