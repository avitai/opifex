# FILE PLACEMENT: opifex/deployment/monitoring/metrics.py
#
# FIXED PrometheusMetrics Implementation
# Fixes missing custom_metrics attribute initialization
#
# This file should REPLACE: opifex/deployment/monitoring/metrics.py

"""
Comprehensive metrics collection system for Opifex framework.

Provides Prometheus-compatible metrics collection with optional dependencies,
custom scientific computing metrics, and robust error handling.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any


# Optional Prometheus imports with graceful fallback
try:
    import prometheus_client

    CollectorRegistry = prometheus_client.CollectorRegistry
    Counter = prometheus_client.Counter
    Gauge = prometheus_client.Gauge
    generate_latest = prometheus_client.generate_latest
    Histogram = prometheus_client.Histogram
    start_http_server = prometheus_client.start_http_server

    has_prometheus = True
except ImportError:
    has_prometheus = False
    # Initialize fallback variables for optional dependencies
    prometheus_client = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    start_http_server = None  # type: ignore[assignment]

# Optional JAX imports
try:
    import jax

    has_jax = True
except ImportError:
    has_jax = False

# Optional psutil for system metrics
try:
    import psutil  # type: ignore[import-untyped]

    has_psutil = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    has_psutil = False


# Export module-level flags for test compatibility
HAS_PROMETHEUS = has_prometheus
HAS_PSUTIL = has_psutil
HAS_JAX = has_jax


@dataclass
class MetricConfig:
    """Configuration for custom metrics."""

    name: str
    description: str
    labels: list[str] = field(default_factory=list)
    metric_type: str = "counter"  # counter, gauge, histogram
    buckets: list[float] | None = None
    namespace: str = "opifex"
    subsystem: str = ""


class PrometheusMetrics:
    """
    Comprehensive Prometheus metrics collection for Opifex framework.

    Handles neural operator training, inference, and system metrics
    with graceful fallback when Prometheus is not available.
    """

    def __init__(
        self,
        namespace: str = "opifex",
        port: int = 8080,
        enable_gpu_metrics: bool = True,
        enable_training_metrics: bool = True,
        enable_inference_metrics: bool = True,
        registry: Any | None = None,
    ):
        """
        Initialize Prometheus metrics system.

        Args:
            namespace: Metrics namespace prefix
            port: HTTP server port for metrics endpoint
            enable_gpu_metrics: Whether to collect GPU metrics
            enable_training_metrics: Whether to collect training metrics
            enable_inference_metrics: Whether to collect inference metrics
            registry: Custom Prometheus registry (optional)
        """
        self.namespace = namespace
        self.port = port
        self.enable_gpu_metrics = enable_gpu_metrics
        self.enable_training_metrics = enable_training_metrics
        self.enable_inference_metrics = enable_inference_metrics

        # CRITICAL FIX: Initialize custom_metrics attribute
        self.custom_metrics: dict[str, Any] = {}

        # Initialize metrics availability flags
        self._metrics_enabled = HAS_PROMETHEUS
        self.logger = logging.getLogger(__name__)

        # Initialize Prometheus registry and metrics
        if self._metrics_enabled:
            try:
                # FIXED: Check CollectorRegistry is not None before calling
                if CollectorRegistry is not None:
                    self.registry = registry or CollectorRegistry()
                else:
                    self.registry = None
                    self._metrics_enabled = False

                if self._metrics_enabled:
                    self._initialize_metrics()
                    self.logger.info("Prometheus metrics initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
                self._metrics_enabled = False
                self.registry = None
        else:
            self.registry = None
            self.logger.warning(
                "Prometheus client not available. Metrics collection disabled."
            )

    def _initialize_metrics(self):
        """Initialize standard metrics for neural operators and training."""
        if not self._metrics_enabled or not HAS_PROMETHEUS:
            return

        # Training metrics
        if self.enable_training_metrics and Gauge is not None:
            self.training_loss = Gauge(
                f"{self.namespace}_training_loss",
                "Current training loss value",
                ["model_type", "job_id"],
                registry=self.registry,
            )

            self.validation_loss = Gauge(
                f"{self.namespace}_validation_loss",
                "Current validation loss value",
                ["model_type", "job_id"],
                registry=self.registry,
            )

            self.gradient_norm = Gauge(
                f"{self.namespace}_gradient_norm",
                "Gradient norm during training",
                ["model_type", "job_id"],
                registry=self.registry,
            )

        # Inference metrics
        if (
            self.enable_inference_metrics
            and Histogram is not None
            and Gauge is not None
        ):
            self.inference_duration = Histogram(
                f"{self.namespace}_inference_duration_seconds",
                "Time spent on inference",
                ["model_type", "model_version"],
                buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
                registry=self.registry,
            )

            self.model_accuracy = Gauge(
                f"{self.namespace}_model_accuracy",
                "Model accuracy metrics",
                ["model_type", "model_version"],
                registry=self.registry,
            )

        # System metrics
        if Gauge is not None:
            self.memory_usage = Gauge(
                f"{self.namespace}_memory_usage_bytes",
                "Current memory usage",
                ["type"],
                registry=self.registry,
            )

        # GPU metrics (if enabled and available)
        if self.enable_gpu_metrics and has_jax and Gauge is not None:
            self.gpu_memory = Gauge(
                f"{self.namespace}_gpu_memory_bytes",
                "GPU memory usage",
                ["device_id"],
                registry=self.registry,
            )

    def create_custom_metric(self, config: MetricConfig) -> Any | None:
        """
        Create a custom metric based on configuration.

        Args:
            config: Metric configuration

        Returns:
            Created metric object or None if Prometheus unavailable
        """
        if not self._metrics_enabled or not HAS_PROMETHEUS:
            return None

        metric_name = f"{config.namespace}_{config.subsystem}_{config.name}".strip("_")

        try:
            if config.metric_type == "counter" and Counter is not None:
                metric = Counter(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry,
                )
            elif config.metric_type == "gauge" and Gauge is not None:
                metric = Gauge(
                    metric_name,
                    config.description,
                    config.labels,
                    registry=self.registry,
                )
            elif config.metric_type == "histogram" and Histogram is not None:
                buckets = config.buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
                metric = Histogram(
                    metric_name,
                    config.description,
                    config.labels,
                    buckets=buckets,
                    registry=self.registry,
                )
            else:
                _raise_unsupported_metric_error(config.metric_type)

            # Store in custom metrics registry
            self.custom_metrics[config.name] = metric
            return metric

        except Exception:
            self.logger.exception(f"Failed to create custom metric {config.name}")
            return None

    def get_custom_metric(self, name: str) -> Any | None:
        """
        Get a custom metric by name.

        Args:
            name: Metric name

        Returns:
            Metric object or None if not found
        """
        return self.custom_metrics.get(name)

    def record_training_metrics(self, model_type: str, job_id: str, loss: float):
        """Record training metrics."""
        if not self._metrics_enabled or not self.enable_training_metrics:
            return

        try:
            self.training_loss.labels(model_type=model_type, job_id=job_id).set(loss)
        except Exception:
            self.logger.exception("Failed to record training metrics")

    def record_inference_metrics(
        self, model_type: str, model_version: str, duration: float
    ):
        """Record inference metrics."""
        if not self._metrics_enabled or not self.enable_inference_metrics:
            return

        try:
            self.inference_duration.labels(
                model_type=model_type, model_version=model_version
            ).observe(duration)
        except Exception:
            self.logger.exception("Failed to record inference metrics")

    def record_inference_accuracy(
        self, model_type: str, model_version: str, accuracy: float
    ):
        """Record model accuracy."""
        if not self._metrics_enabled or not self.enable_inference_metrics:
            return

        try:
            self.model_accuracy.labels(
                model_type=model_type, model_version=model_version
            ).set(accuracy)
        except Exception:
            self.logger.exception("Failed to record accuracy")

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not self._metrics_enabled or not HAS_PROMETHEUS or not HAS_PSUTIL:
            return

        try:
            # Only use psutil if it's available
            if psutil is not None:
                # Get system metrics (keep for future use)
                _ = psutil.cpu_percent()  # Trigger CPU measurement
                memory = psutil.virtual_memory()

                # Update CPU and memory metrics
                if hasattr(self, "memory_usage"):
                    self.memory_usage.labels(type="used").set(memory.used)
                    self.memory_usage.labels(type="total").set(memory.total)
        except Exception as e:
            self.logger.warning(f"Failed to update system metrics: {e}")

    def update_gpu_metrics(self):
        """Update GPU metrics if available."""
        if (
            not self._metrics_enabled
            or not has_prometheus
            or not self.enable_gpu_metrics
        ):
            return

        try:
            if has_jax:
                devices = jax.devices()
                for device in devices:
                    if hasattr(self, "gpu_memory") and hasattr(device, "id"):
                        # Mock GPU memory usage - in practice would use nvidia-ml-py
                        self.gpu_memory.labels(device_id=str(device.id)).set(
                            1024 * 1024 * 1024  # Mock GPU memory value
                        )
        except Exception as e:
            self.logger.warning(f"Failed to update GPU metrics: {e}")

    def start_metrics_server(self, port: int | None = None) -> None:
        """Start HTTP metrics server for Prometheus scraping."""
        if not self._metrics_enabled:
            self.logger.warning(
                "Metrics server cannot start - Prometheus not available"
            )
            return

        server_port = port or self.port
        try:
            if self.registry is not None and start_http_server is not None:
                start_http_server(server_port, registry=self.registry)
                self.logger.info(
                    f"Prometheus metrics server started on port {server_port}"
                )
            else:
                self.logger.error(
                    "Cannot start server: registry or start_http_server is None"
                )
        except Exception:
            self.logger.exception("Failed to start metrics server")

    def get_metrics_data(self) -> str:
        """Get current metrics data in Prometheus format."""
        if not self._metrics_enabled:
            return "Prometheus metrics not available"

        try:
            if self.registry is not None and generate_latest is not None:
                return generate_latest(self.registry).decode("utf-8")
            return "Prometheus metrics not available"
        except Exception:
            self.logger.exception("Failed to generate metrics data")
            return "Prometheus metrics not available"

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check for metrics collection.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self._metrics_enabled else "disabled",
            "metrics_enabled": self._metrics_enabled,
            "prometheus_available": has_prometheus,
            "jax_available": has_jax,
            "psutil_available": has_psutil,
            "gpu_metrics_enabled": self.enable_gpu_metrics,
            "training_metrics_enabled": self.enable_training_metrics,
            "inference_metrics_enabled": self.enable_inference_metrics,
            "custom_metrics_count": len(
                self.custom_metrics
            ),  # This was causing the AttributeError
            "timestamp": time.time(),
        }


class CustomMetrics:
    """
    Custom metrics for scientific computing-specific monitoring.

    Provides specialized metrics for neural operators, scientific
    simulations, and domain-specific performance monitoring.
    """

    def __init__(self, prometheus_metrics: PrometheusMetrics):
        """
        Initialize custom metrics.

        Args:
            prometheus_metrics: PrometheusMetrics instance for metric creation
        """
        self.prometheus_metrics = prometheus_metrics

        # Initialize scientific computing metrics
        self._initialize_neural_operator_metrics()
        self._initialize_simulation_metrics()

    def _initialize_neural_operator_metrics(self):
        """Initialize neural operator specific metrics."""
        if not self.prometheus_metrics._metrics_enabled:
            return

        # FNO specific metrics
        fno_config = MetricConfig(
            name="fno_forward_time",
            description="Time taken for FNO forward pass",
            labels=["model_id", "resolution"],
            metric_type="histogram",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
        )
        self.fno_forward_time = self.prometheus_metrics.create_custom_metric(fno_config)

        # DeepONet metrics
        deeponet_branch_config = MetricConfig(
            name="deeponet_branch_size",
            description="DeepONet branch network size",
            labels=["model_id"],
            metric_type="gauge",
        )
        self.deeponet_branch_size = self.prometheus_metrics.create_custom_metric(
            deeponet_branch_config
        )

        deeponet_trunk_config = MetricConfig(
            name="deeponet_trunk_size",
            description="DeepONet trunk network size",
            labels=["model_id"],
            metric_type="gauge",
        )
        self.deeponet_trunk_size = self.prometheus_metrics.create_custom_metric(
            deeponet_trunk_config
        )

    def _initialize_simulation_metrics(self):
        """Initialize simulation-specific metrics."""
        if not self.prometheus_metrics._metrics_enabled:
            return

        # PDE accuracy metrics
        pde_accuracy_config = MetricConfig(
            name="pde_accuracy",
            description="PDE solution accuracy",
            labels=["equation_type", "method"],
            metric_type="gauge",
        )
        self.pde_accuracy = self.prometheus_metrics.create_custom_metric(
            pde_accuracy_config
        )

        # Computational efficiency
        efficiency_config = MetricConfig(
            name="computational_efficiency",
            description="Computational efficiency score",
            labels=["algorithm", "problem_size"],
            metric_type="gauge",
        )
        self.computational_efficiency = self.prometheus_metrics.create_custom_metric(
            efficiency_config
        )

        # Simulation step time
        step_time_config = MetricConfig(
            name="simulation_step_time",
            description="Time per simulation step",
            labels=["simulation_type", "grid_size"],
            metric_type="histogram",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
        )
        self.simulation_step_time = self.prometheus_metrics.create_custom_metric(
            step_time_config
        )

        # Convergence metrics
        convergence_config = MetricConfig(
            name="simulation_convergence",
            description="Simulation convergence iterations",
            labels=["simulation_type", "tolerance"],
            metric_type="histogram",
            buckets=[1, 10, 100, 1000, 10000],
        )
        self.simulation_convergence = self.prometheus_metrics.create_custom_metric(
            convergence_config
        )

    def record_fno_metrics(
        self, model_id: str, modes: dict[str, int], forward_time: float, resolution: str
    ):
        """Record FNO-specific metrics."""
        if hasattr(self, "fno_forward_time") and self.fno_forward_time:
            self.fno_forward_time.labels(
                model_id=model_id, resolution=resolution
            ).observe(forward_time)

    def record_deeponet_metrics(self, model_id: str, branch_size: int, trunk_size: int):
        """Record DeepONet-specific metrics."""
        if hasattr(self, "deeponet_branch_size") and self.deeponet_branch_size:
            self.deeponet_branch_size.labels(model_id=model_id).set(branch_size)
        if hasattr(self, "deeponet_trunk_size") and self.deeponet_trunk_size:
            self.deeponet_trunk_size.labels(model_id=model_id).set(trunk_size)

    def record_simulation_metrics(
        self,
        simulation_type: str,
        grid_size: str,
        step_time: float,
        convergence_iterations: int,
        tolerance: str,
    ):
        """Record simulation-specific metrics."""
        if hasattr(self, "simulation_step_time") and self.simulation_step_time:
            self.simulation_step_time.labels(
                simulation_type=simulation_type, grid_size=grid_size
            ).observe(step_time)

        if hasattr(self, "simulation_convergence") and self.simulation_convergence:
            self.simulation_convergence.labels(
                simulation_type=simulation_type, tolerance=tolerance
            ).observe(convergence_iterations)

    def record_pde_accuracy(self, equation_type: str, method: str, error: float):
        """Record PDE accuracy metrics."""
        if hasattr(self, "pde_accuracy") and self.pde_accuracy:
            self.pde_accuracy.labels(equation_type=equation_type, method=method).set(
                error
            )

    def record_computational_efficiency(
        self, algorithm: str, problem_size: str, efficiency: float
    ):
        """Record computational efficiency metrics."""
        if hasattr(self, "computational_efficiency") and self.computational_efficiency:
            self.computational_efficiency.labels(
                algorithm=algorithm, problem_size=problem_size
            ).set(efficiency)


def _raise_unsupported_metric_error(metric_type: str) -> None:
    """Helper function to raise unsupported metric type error."""
    raise ValueError(f"Unsupported metric type: {metric_type}")
