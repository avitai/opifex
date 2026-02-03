"""
Health Check Management for Opifex Framework.

Comprehensive health monitoring for production deployment.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Optional dependencies with proper type checking
try:
    import psutil  # type: ignore[import-untyped]

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False  # type: ignore[misc]

try:
    import jax  # type: ignore[import-untyped]

    HAS_JAX = True
except ImportError:
    HAS_JAX = False  # type: ignore[misc]

try:
    import requests  # type: ignore[import-untyped]

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False  # type: ignore[misc]

try:
    import prometheus_client  # type: ignore[import-untyped] # noqa: F401

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False  # type: ignore[misc]


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert health check result to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ServiceHealth:
    """Overall service health status."""

    overall_status: HealthStatus
    checks: list[HealthCheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert service health to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks],
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert service health to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HealthChecker:
    """
    Comprehensive health checker for Opifex production services.

    This class provides enterprise-grade health monitoring including
    system resources, service dependencies, model availability, and
    custom health checks for scientific computing workloads.
    """

    def __init__(
        self,
        service_name: str = "opifex",
        enable_system_checks: bool = True,
        enable_model_checks: bool = True,
        enable_dependency_checks: bool = True,
        check_timeout: float = 30.0,
    ):
        """
        Initialize health checker.

        Args:
            service_name: Name of the service being monitored
            enable_system_checks: Enable system resource checks
            enable_model_checks: Enable model health checks
            enable_dependency_checks: Enable external dependency checks
            check_timeout: Timeout for individual health checks (seconds)
        """
        self.service_name = service_name
        self.enable_system_checks = enable_system_checks
        self.enable_model_checks = enable_model_checks
        self.enable_dependency_checks = enable_dependency_checks
        self.check_timeout = check_timeout
        self.logger = logging.getLogger(__name__)

        # Register health check functions
        self.health_checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self.dependency_urls: dict[str, str] = {}
        self.model_endpoints: dict[str, str] = {}

        # Initialize default health checks
        self._register_default_checks()

        self.logger.info(f"Health checker initialized for service: {service_name}")

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        if self.enable_system_checks:
            self.register_health_check("system_resources", self._check_system_resources)
            if HAS_JAX:
                self.register_health_check(
                    "gpu_availability", self._check_gpu_availability
                )

        # Always register basic application health
        self.register_health_check("application", self._check_application_health)

    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a custom health check function."""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")

    def register_dependency(self, name: str, url: str) -> None:
        """Register an external dependency for health monitoring."""
        self.dependency_urls[name] = url
        if self.enable_dependency_checks:
            self.register_health_check(
                f"dependency_{name}", lambda: self._check_dependency(name, url)
            )
        self.logger.info(f"Registered dependency: {name} -> {url}")

    def register_model_endpoint(self, name: str, endpoint: str) -> None:
        """Register a model endpoint for health monitoring."""
        self.model_endpoints[name] = endpoint
        if self.enable_model_checks:
            self.register_health_check(
                f"model_{name}", lambda: self._check_model_endpoint(name, endpoint)
            )
        self.logger.info(f"Registered model endpoint: {name} -> {endpoint}")

    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        start_time = time.time()

        if not HAS_PSUTIL:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system monitoring",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }

            # Determine health status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage detected"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 85:
                status = HealthStatus.DEGRADED
                message = "High resource usage detected"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details=details,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU availability and health."""
        start_time = time.time()

        if not HAS_JAX:
            return HealthCheckResult(
                name="gpu_availability",
                status=HealthStatus.UNKNOWN,
                message="JAX not available for GPU monitoring",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == "gpu"]

            if not gpu_devices:
                return HealthCheckResult(
                    name="gpu_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="No GPU devices found",
                    details={"total_devices": len(devices)},
                    duration_ms=(time.time() - start_time) * 1000,
                )

            gpu_details = []
            total_memory_used = 0
            total_memory_limit = 0

            for i, device in enumerate(gpu_devices):
                try:
                    memory_stats = device.memory_stats()
                    if memory_stats:
                        memory_used = memory_stats.get("bytes_in_use", 0)
                        memory_limit = memory_stats.get("bytes_limit", 0)
                        total_memory_used += memory_used
                        total_memory_limit += memory_limit

                        gpu_details.append(
                            {
                                "device_id": i,
                                "platform": device.platform,
                                "memory_used_gb": memory_used / (1024**3),
                                "memory_limit_gb": memory_limit / (1024**3),
                                "memory_percent": (memory_used / memory_limit * 100)
                                if memory_limit > 0
                                else 0,
                            }
                        )
                except Exception as e:
                    gpu_details.append(
                        {
                            "device_id": i,
                            "error": str(e),
                        }
                    )

            # Calculate overall GPU memory usage
            memory_percent = (
                (total_memory_used / total_memory_limit * 100)
                if total_memory_limit > 0
                else 0
            )

            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical GPU memory usage"
            elif memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High GPU memory usage"
            else:
                status = HealthStatus.HEALTHY
                message = f"{len(gpu_devices)} GPU(s) available"

            return HealthCheckResult(
                name="gpu_availability",
                status=status,
                message=message,
                details={
                    "gpu_count": len(gpu_devices),
                    "total_memory_percent": memory_percent,
                    "gpus": gpu_details,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="gpu_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check GPU availability: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _check_application_health(self) -> HealthCheckResult:
        """Check basic application health."""
        start_time = time.time()

        try:
            # Basic application health checks
            details = {
                "service_name": self.service_name,
                "jax_available": HAS_JAX,
                "psutil_available": HAS_PSUTIL,
                "requests_available": HAS_REQUESTS,
                "uptime_seconds": time.time() - start_time,
            }

            return HealthCheckResult(
                name="application",
                status=HealthStatus.HEALTHY,
                message=f"Application {self.service_name} is running",
                details=details,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="application",
                status=HealthStatus.UNHEALTHY,
                message=f"Application health check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _check_dependency(self, name: str, url: str) -> HealthCheckResult:
        """Check external dependency health."""
        start_time = time.time()

        if not HAS_REQUESTS:
            return HealthCheckResult(
                name=f"dependency_{name}",
                status=HealthStatus.UNKNOWN,
                message="requests library not available for dependency checks",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            response = requests.get(url, timeout=self.check_timeout)

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = f"Dependency {name} is reachable"
            elif 400 <= response.status_code < 500:
                status = HealthStatus.DEGRADED
                message = (
                    f"Dependency {name} returned client error: {response.status_code}"
                )
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Dependency {name} returned error: {response.status_code}"

            return HealthCheckResult(
                name=f"dependency_{name}",
                status=status,
                message=message,
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                },
                duration_ms=(time.time() - start_time) * 1000,
            )

        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name=f"dependency_{name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency {name} timed out",
                details={"url": url, "timeout_seconds": self.check_timeout},
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"dependency_{name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to reach dependency {name}: {e}",
                details={"url": url},
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _check_model_endpoint(self, name: str, endpoint: str) -> HealthCheckResult:
        """Check model endpoint health."""
        start_time = time.time()

        if not HAS_REQUESTS:
            return HealthCheckResult(
                name=f"model_{name}",
                status=HealthStatus.UNKNOWN,
                message="requests library not available for model checks",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            # Try health endpoint first
            health_url = (
                f"{endpoint}/health" if not endpoint.endswith("/health") else endpoint
            )
            response = requests.get(health_url, timeout=self.check_timeout)

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = f"Model {name} is healthy"
                details = {
                    "endpoint": endpoint,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                }

                # Try to parse response for additional details
                try:
                    response_data = response.json()
                    details.update(response_data)
                except Exception:
                    pass  # Ignore JSON parsing errors

            else:
                status = HealthStatus.UNHEALTHY
                message = f"Model {name} health check failed: {response.status_code}"
                details = {"endpoint": endpoint, "status_code": response.status_code}

            return HealthCheckResult(
                name=f"model_{name}",
                status=status,
                message=message,
                details=details,
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name=f"model_{name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Model {name} health check failed: {e}",
                details={"endpoint": endpoint},
                duration_ms=(time.time() - start_time) * 1000,
            )

    def run_health_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if check_name not in self.health_checks:
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{check_name}' not found",
            )

        try:
            return self.health_checks[check_name]()
        except Exception as e:
            self.logger.exception(f"Health check '{check_name}' failed")
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
            )

    def run_all_health_checks(self) -> ServiceHealth:
        """Run all registered health checks."""
        start_time = time.time()
        results = []

        for check_name in self.health_checks:
            try:
                result = self.run_health_check(check_name)
                results.append(result)
            except Exception as e:
                self.logger.exception(f"Failed to run health check '{check_name}'")
                results.append(
                    HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check execution failed: {e}",
                    )
                )

        # Determine overall health status
        if not results:
            overall_status = HealthStatus.UNKNOWN
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            overall_status = HealthStatus.DEGRADED
        elif all(r.status == HealthStatus.HEALTHY for r in results):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        total_duration = (time.time() - start_time) * 1000
        self.logger.info(
            f"Health check completed in {total_duration:.2f}ms - "
            f"Status: {overall_status.value}"
        )

        return ServiceHealth(
            overall_status=overall_status,
            checks=results,
        )

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of current health status."""
        service_health = self.run_all_health_checks()

        return {
            "service_name": self.service_name,
            "overall_status": service_health.overall_status.value,
            "total_checks": len(service_health.checks),
            "healthy_checks": len(
                [c for c in service_health.checks if c.status == HealthStatus.HEALTHY]
            ),
            "degraded_checks": len(
                [c for c in service_health.checks if c.status == HealthStatus.DEGRADED]
            ),
            "unhealthy_checks": len(
                [c for c in service_health.checks if c.status == HealthStatus.UNHEALTHY]
            ),
            "unknown_checks": len(
                [c for c in service_health.checks if c.status == HealthStatus.UNKNOWN]
            ),
            "timestamp": service_health.timestamp,
        }

    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        service_health = self.run_all_health_checks()
        return service_health.overall_status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """Check if the service is ready (healthy or degraded)."""
        service_health = self.run_all_health_checks()
        return service_health.overall_status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        ]

    async def run_periodic_health_checks(
        self,
        interval_seconds: float = 60.0,
        callback: Callable[[ServiceHealth], None] | None = None,
    ) -> None:
        """Run health checks periodically."""
        self.logger.info(
            f"Starting periodic health checks every {interval_seconds} seconds"
        )

        while True:
            try:
                service_health = self.run_all_health_checks()

                if callback:
                    callback(service_health)

                self.logger.debug(
                    f"Periodic health check completed - "
                    f"Status: {service_health.overall_status.value}"
                )

            except Exception:
                self.logger.exception("Periodic health check failed")

            await asyncio.sleep(interval_seconds)
