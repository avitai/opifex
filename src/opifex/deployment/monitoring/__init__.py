"""Monitoring and Observability Infrastructure for Opifex Production Deployment.

Provides monitoring, logging, and observability for Opifex deployment:
- Prometheus metrics collection and configuration
- Grafana dashboard management and deployment
- Structured logging with ELK stack integration
- Health check endpoints and alerting systems
"""

import logging as _logging


_logger = _logging.getLogger(__name__)

try:
    from opifex.deployment.monitoring.alerts import (
        Alert,
        AlertManager,
        AlertRule,
        AlertSeverity,
        AlertStatus,
        get_global_alert_manager,
        setup_global_alert_manager,
    )
    from opifex.deployment.monitoring.dashboards import (
        create_neural_operator_dashboard,
        Dashboard,
        GrafanaManager,
        Panel,
    )
    from opifex.deployment.monitoring.health import (
        HealthChecker,
        HealthCheckResult,
        HealthStatus,
        ServiceHealth,
    )
    from opifex.deployment.monitoring.logging import (
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
    from opifex.deployment.monitoring.metrics import (
        CustomMetrics,
        MetricConfig,
        PrometheusMetrics,
    )
except ImportError:
    _logger.debug(
        "Monitoring dependencies not installed. Install with: uv pip install opifex[platform]"
    )

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
