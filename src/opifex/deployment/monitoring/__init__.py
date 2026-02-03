"""
Monitoring and Observability Infrastructure for Opifex Production Deployment.

This module provides enterprise-grade monitoring, logging, and observability
capabilities for the Opifex framework deployment in Kubernetes environments.

Features:
- Prometheus metrics collection and configuration
- Grafana dashboard management and deployment
- Structured logging with ELK stack integration
- Application performance monitoring (APM) with tracing
- Health check endpoints and alerting systems
- Custom metrics for scientific computing workloads

Components:
- metrics: Prometheus metrics collection and custom metrics
- logging: Structured logging configuration and ELK integration
- dashboards: Grafana dashboard management and templates
- alerts: Alerting rules and notification configuration
- health: Health check endpoints and monitoring
- tracing: Distributed tracing with OpenTelemetry
"""

from typing import TYPE_CHECKING


# Core monitoring components
if TYPE_CHECKING:
    from .dashboards import Dashboard, GrafanaManager, Panel
    from .health import HealthChecker, HealthStatus, ServiceHealth
    from .logging import LogContext, StructuredLogger
    from .metrics import CustomMetrics, PrometheusMetrics

try:
    from .alerts import (
        Alert,
        AlertManager,
        AlertRule,
        AlertSeverity,
        AlertStatus,
        get_global_alert_manager,
        setup_global_alert_manager,
    )
    from .dashboards import (
        create_neural_operator_dashboard,
        Dashboard,
        GrafanaManager,
        Panel,
    )
    from .health import (
        HealthChecker,
        HealthCheckResult,
        HealthStatus,
        ServiceHealth,
    )
    from .logging import (
        get_global_logger,
        get_logger,
        JsonFormatter,
        log_error,
        log_inference_request,
        log_model_load,
        log_training_start,
        log_training_step,
        LogContext,
        LogEntry,
        LoggingConfig,
        setup_global_logger,
        StructuredLogger,
    )
    from .metrics import (
        CustomMetrics,
        MetricConfig,
        PrometheusMetrics,
    )
except ImportError:
    # Graceful fallback for missing monitoring dependencies
    pass

__all__ = [
    "Alert",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "CustomMetrics",
    "Dashboard",
    "GrafanaManager",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
    "JsonFormatter",
    "LogContext",
    "LogEntry",
    "LoggingConfig",
    "MetricConfig",
    "Panel",
    "PrometheusMetrics",
    "ServiceHealth",
    "StructuredLogger",
    "create_neural_operator_dashboard",
    "get_global_alert_manager",
    "get_global_logger",
    "get_logger",
    "log_error",
    "log_inference_request",
    "log_model_load",
    "log_training_start",
    "log_training_step",
    "setup_global_alert_manager",
    "setup_global_logger",
]
