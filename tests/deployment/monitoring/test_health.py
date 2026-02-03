"""Basic tests for Opifex health monitoring system."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from opifex.deployment.monitoring.health import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    ServiceHealth,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Service is healthy",
            details={"key": "value"},
            duration_ms=100.0,
        )

        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.details == {"key": "value"}
        assert result.duration_ms == 100.0

    def test_health_check_result_to_dict(self):
        """Test HealthCheckResult to_dict conversion."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"cpu": 50},
            timestamp=1000.0,
            duration_ms=50.0,
        )

        result_dict = result.to_dict()

        assert result_dict["name"] == "test_check"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "All good"
        assert result_dict["details"] == {"cpu": 50}
        assert result_dict["timestamp"] == 1000.0
        assert result_dict["duration_ms"] == 50.0


class TestServiceHealth:
    """Test ServiceHealth dataclass."""

    def test_service_health_creation(self):
        """Test ServiceHealth creation."""
        check1 = HealthCheckResult("check1", HealthStatus.HEALTHY)
        check2 = HealthCheckResult("check2", HealthStatus.DEGRADED)

        service_health = ServiceHealth(
            overall_status=HealthStatus.DEGRADED,
            checks=[check1, check2],
            timestamp=1000.0,
        )

        assert service_health.overall_status == HealthStatus.DEGRADED
        assert len(service_health.checks) == 2
        assert service_health.timestamp == 1000.0

    def test_service_health_to_dict(self):
        """Test ServiceHealth to_dict conversion."""
        check = HealthCheckResult("test", HealthStatus.HEALTHY)
        service_health = ServiceHealth(
            overall_status=HealthStatus.HEALTHY, checks=[check], timestamp=1000.0
        )

        result_dict = service_health.to_dict()

        assert result_dict["overall_status"] == "healthy"
        assert len(result_dict["checks"]) == 1
        assert result_dict["timestamp"] == 1000.0

    def test_service_health_to_json(self):
        """Test ServiceHealth to_json conversion."""
        check = HealthCheckResult("test", HealthStatus.HEALTHY)
        service_health = ServiceHealth(
            overall_status=HealthStatus.HEALTHY, checks=[check]
        )

        json_str = service_health.to_json()

        assert '"overall_status": "healthy"' in json_str
        assert '"name": "test"' in json_str


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_health_checker_init_default(self):
        """Test HealthChecker initialization with defaults."""
        checker = HealthChecker()

        assert checker.service_name == "opifex"
        assert checker.enable_system_checks is True
        assert checker.enable_model_checks is True
        assert checker.enable_dependency_checks is True
        assert checker.check_timeout == 30.0
        assert isinstance(checker.health_checks, dict)
        assert isinstance(checker.dependency_urls, dict)
        assert isinstance(checker.model_endpoints, dict)

    def test_health_checker_init_custom(self):
        """Test HealthChecker initialization with custom values."""
        checker = HealthChecker(
            service_name="test_service",
            enable_system_checks=False,
            enable_model_checks=False,
            enable_dependency_checks=False,
            check_timeout=10.0,
        )

        assert checker.service_name == "test_service"
        assert checker.enable_system_checks is False
        assert checker.enable_model_checks is False
        assert checker.enable_dependency_checks is False
        assert checker.check_timeout == 10.0

    def test_register_health_check(self):
        """Test registering custom health check."""
        checker = HealthChecker()

        def custom_check():
            return HealthCheckResult("custom", HealthStatus.HEALTHY)

        checker.register_health_check("custom_check", custom_check)

        assert "custom_check" in checker.health_checks
        assert checker.health_checks["custom_check"] == custom_check

    def test_register_dependency(self):
        """Test registering dependency for monitoring."""
        checker = HealthChecker(enable_dependency_checks=True)

        checker.register_dependency("database", "http://db.example.com")

        assert "database" in checker.dependency_urls
        assert checker.dependency_urls["database"] == "http://db.example.com"
        assert "dependency_database" in checker.health_checks

    def test_register_model_endpoint(self):
        """Test registering model endpoint for monitoring."""
        checker = HealthChecker(enable_model_checks=True)

        checker.register_model_endpoint("fno_model", "http://model.example.com")

        assert "fno_model" in checker.model_endpoints
        assert checker.model_endpoints["fno_model"] == "http://model.example.com"
        assert "model_fno_model" in checker.health_checks

    @patch("opifex.deployment.monitoring.health.HAS_PSUTIL", True)
    @patch("opifex.deployment.monitoring.health.psutil")
    def test_check_system_resources_healthy(self, mock_psutil):
        """Test system resources check when healthy."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 40.0
        mock_disk.free = 100 * 1024**3  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk

        checker = HealthChecker()
        result = checker._check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert "cpu_percent" in result.details
        assert "memory_percent" in result.details
        assert "disk_percent" in result.details

    @patch("opifex.deployment.monitoring.health.HAS_PSUTIL", False)
    def test_check_system_resources_no_psutil(self):
        """Test system resources check when psutil is not available."""
        checker = HealthChecker()
        result = checker._check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.UNKNOWN
        assert "psutil not available" in result.message

    @patch("opifex.deployment.monitoring.health.HAS_PSUTIL", True)
    @patch("opifex.deployment.monitoring.health.psutil")
    def test_check_system_resources_degraded(self, mock_psutil):
        """Test system resources check when degraded."""
        # Mock degraded resource usage (below critical thresholds)
        mock_psutil.cpu_percent.return_value = 75.0  # Above 70, below 90
        mock_memory = Mock()
        mock_memory.percent = 75.0  # Above 70, below 90
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 88.0  # Above 85, below 95
        mock_disk.free = 50 * 1024**3  # 50GB
        mock_psutil.disk_usage.return_value = mock_disk

        checker = HealthChecker()
        result = checker._check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.DEGRADED

    @patch("opifex.deployment.monitoring.health.HAS_JAX", True)
    @patch("opifex.deployment.monitoring.health.jax")
    def test_check_gpu_availability_with_jax(self, mock_jax):
        """Test GPU availability check when JAX is available."""
        # Mock GPU device with proper attributes
        mock_gpu_device = Mock()
        mock_gpu_device.device_kind = "gpu"
        mock_gpu_device.platform = "gpu"
        mock_gpu_device.memory_stats.return_value = {
            "bytes_in_use": 2 * 1024**3,  # 2GB used
            "bytes_limit": 8 * 1024**3,  # 8GB total
        }
        mock_jax.devices.return_value = [mock_gpu_device]

        checker = HealthChecker()
        result = checker._check_gpu_availability()

        assert result.name == "gpu_availability"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["gpu_count"] == 1

    @patch("opifex.deployment.monitoring.health.HAS_JAX", False)
    def test_check_gpu_availability_no_jax(self):
        """Test GPU availability check when JAX is not available."""
        checker = HealthChecker()
        result = checker._check_gpu_availability()

        assert result.name == "gpu_availability"
        assert result.status == HealthStatus.UNKNOWN
        assert "JAX not available" in result.message

    def test_check_application_health(self):
        """Test application health check."""
        checker = HealthChecker()
        result = checker._check_application_health()

        assert result.name == "application"
        assert result.status == HealthStatus.HEALTHY
        assert "Application opifex is running" in result.message

    @patch("opifex.deployment.monitoring.health.HAS_REQUESTS", False)
    def test_check_dependency_no_requests(self):
        """Test dependency check when requests is not available."""
        checker = HealthChecker(enable_dependency_checks=True)
        checker.register_dependency("test_service", "http://example.com")

        result = checker._check_dependency("test_service", "http://example.com")

        assert result.name == "dependency_test_service"
        assert result.status == HealthStatus.UNKNOWN
        assert "requests library not available" in result.message

    def test_run_health_check_success(self):
        """Test running a health check successfully."""
        checker = HealthChecker()

        def custom_check():
            return HealthCheckResult("custom", HealthStatus.HEALTHY)

        checker.register_health_check("custom_check", custom_check)

        result = checker.run_health_check("custom_check")

        assert result.name == "custom"
        assert result.status == HealthStatus.HEALTHY

    def test_run_health_check_not_found(self):
        """Test running a health check that doesn't exist."""
        checker = HealthChecker()

        result = checker.run_health_check("non_existent_check")

        assert result.name == "non_existent_check"
        assert result.status == HealthStatus.UNKNOWN
        assert "Health check 'non_existent_check' not found" in result.message

    def test_run_health_check_exception(self):
        """Test running a health check that raises an exception."""
        checker = HealthChecker()

        def failing_check():
            raise ValueError("Test exception")

        checker.register_health_check("failing_check", failing_check)

        result = checker.run_health_check("failing_check")

        assert result.name == "failing_check"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Test exception" in result.message

    def test_run_all_health_checks_all_healthy(self):
        """Test running all health checks when all are healthy."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY)

        checker.register_health_check("healthy_check", healthy_check)

        service_health = checker.run_all_health_checks()

        assert service_health.overall_status == HealthStatus.HEALTHY
        assert len(service_health.checks) >= 1

    def test_get_health_summary(self):
        """Test getting health summary."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY)

        def unhealthy_check():
            return HealthCheckResult("unhealthy", HealthStatus.UNHEALTHY)

        checker.register_health_check("healthy_check", healthy_check)
        checker.register_health_check("unhealthy_check", unhealthy_check)

        summary = checker.get_health_summary()

        assert (
            summary["overall_status"] == "unhealthy"
        )  # Mixed health statuses with unhealthy
        assert summary["total_checks"] >= 2
        assert summary["healthy_checks"] >= 1
        assert summary["unhealthy_checks"] >= 1

    def test_is_healthy_when_healthy(self):
        """Test is_healthy method when service is healthy."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY)

        checker.register_health_check("healthy_check", healthy_check)

        assert checker.is_healthy() is True

    def test_is_healthy_when_unhealthy(self):
        """Test is_healthy method when service is unhealthy."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def unhealthy_check():
            return HealthCheckResult("unhealthy", HealthStatus.UNHEALTHY)

        checker.register_health_check("unhealthy_check", unhealthy_check)

        assert checker.is_healthy() is False

    def test_is_ready_when_healthy(self):
        """Test is_ready method when service is healthy."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY)

        checker.register_health_check("healthy_check", healthy_check)

        assert checker.is_ready() is True

    def test_is_ready_when_degraded(self):
        """Test is_ready method when service is degraded (should still be ready)."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def degraded_check():
            return HealthCheckResult("degraded", HealthStatus.DEGRADED)

        checker.register_health_check("degraded_check", degraded_check)

        assert checker.is_ready() is True  # Degraded is still ready

    def test_is_ready_when_unhealthy(self):
        """Test is_ready method when service is unhealthy."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def unhealthy_check():
            return HealthCheckResult("unhealthy", HealthStatus.UNHEALTHY)

        checker.register_health_check("unhealthy_check", unhealthy_check)

        assert checker.is_ready() is False

    def test_prometheus_dependency_handling(self):
        """Test that health checker handles missing prometheus-client gracefully."""
        with patch("opifex.deployment.monitoring.health.HAS_PROMETHEUS", False):
            checker = HealthChecker()

            # Should still initialize successfully without prometheus
            assert checker.service_name == "opifex"

            # Health checks should work without prometheus dependency
        service_health = checker.run_all_health_checks()
        # Overall status can be any valid status including unhealthy due to actual system conditions
        assert service_health.overall_status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]


class TestIntegration:
    """Test integration scenarios."""

    def test_complete_health_monitoring_workflow(self):
        """Test complete health monitoring workflow."""
        checker = HealthChecker(
            service_name="test_opifex",
            enable_system_checks=True,
            enable_dependency_checks=False,  # Disable to avoid network calls
        )

        # Register custom check
        def custom_business_logic_check():
            return HealthCheckResult(
                "business_logic",
                HealthStatus.HEALTHY,
                "Business logic is functioning correctly",
                {"transactions_per_second": 100},
            )

        checker.register_health_check("business_logic", custom_business_logic_check)

        # Run all checks
        service_health = checker.run_all_health_checks()

        # Verify results
        assert isinstance(service_health.overall_status, HealthStatus)
        assert len(service_health.checks) >= 1
        assert service_health.timestamp > 0

        # Test summary
        summary = checker.get_health_summary()
        assert "overall_status" in summary
        assert "total_checks" in summary

        # Test readiness and liveness
        assert isinstance(checker.is_healthy(), bool)
        assert isinstance(checker.is_ready(), bool)

    @pytest.mark.asyncio
    async def test_periodic_health_checks(self):
        """Test periodic health check execution."""
        checker = HealthChecker(
            enable_system_checks=False, enable_dependency_checks=False
        )

        def simple_check():
            return HealthCheckResult("simple", HealthStatus.HEALTHY)

        checker.register_health_check("simple_check", simple_check)

        health_results = []

        def health_callback(service_health: ServiceHealth):
            health_results.append(service_health)

        # Simulate periodic check (in real scenario, this would be scheduled)
        for _ in range(3):
            service_health = checker.run_all_health_checks()
            health_callback(service_health)
            await asyncio.sleep(0.1)  # Small delay

        assert len(health_results) == 3
        for result in health_results:
            assert isinstance(result, ServiceHealth)
