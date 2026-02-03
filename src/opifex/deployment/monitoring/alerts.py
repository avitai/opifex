"""
Alert Management System for Opifex Framework.

This module provides enterprise-grade alerting capabilities specifically
designed for scientific computing workloads, including system resource
monitoring, training failure detection, and model performance alerts.
"""

import json
import smtplib
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, UTC
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any


# Optional dependencies with proper type checking
try:
    import requests  # type: ignore[import-untyped]

    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAS_REQUESTS = False  # type: ignore[misc]


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    description: str
    query: str  # Prometheus query
    severity: AlertSeverity
    threshold: float
    duration: str = "5m"  # Duration to wait before firing
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_prometheus_rule(self) -> dict[str, Any]:
        """Convert to Prometheus alerting rule format."""
        return {
            "alert": self.name,
            "expr": f"{self.query} > {self.threshold}",
            "for": self.duration,
            "labels": {"severity": self.severity.value, **self.labels},
            "annotations": {"description": self.description, **self.annotations},
        }


@dataclass
class Alert:
    """Individual alert instance."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime
    resolved_timestamp: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.resolved_timestamp:
            data["resolved_timestamp"] = self.resolved_timestamp.isoformat()
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        return data


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, name: str):
        self.name = name

    def send_alert(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class EmailNotification(NotificationChannel):
    """Email notification channel."""

    def __init__(
        self,
        name: str,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        to_addresses: list[str],
        from_address: str | None = None,
    ):
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_addresses = to_addresses
        self.from_address = from_address or username

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = (
                f"[{alert.severity.value.upper()}] Opifex Alert: {alert.rule_name}"
            )

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True

        except Exception:
            # Log error instead of print for production systems
            return False

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        color = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.INFO: "#17a2b8",
        }.get(alert.severity, "#6c757d")

        return f"""
        <html>
        <body>
            <h2 style="color: {color};">Opifex Framework Alert</h2>
            <table border="1" style="border-collapse: collapse;">
                <tr><td><strong>Alert Name</strong></td><td>{alert.rule_name}</td></tr>
                <tr><td><strong>Severity</strong></td><td style="color: {color};">
                {alert.severity.value.upper()}</td></tr>
                <tr><td><strong>Status</strong></td><td>{alert.status.value}</td></tr>
                <tr><td><strong>Message</strong></td><td>{alert.message}</td></tr>
                <tr><td><strong>Timestamp</strong></td><td>
                {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>
                {
            f"<tr><td><strong>Value</strong></td><td>{alert.value}</td></tr>"
            if alert.value
            else ""
        }
            </table>
            <br>
            <h3>Labels</h3>
            <table border="1" style="border-collapse: collapse;">
                {
            "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in alert.labels.items()
            )
        }
            </table>
            <br>
            <h3>Annotations</h3>
            <table border="1" style="border-collapse: collapse;">
                {
            "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in alert.annotations.items()
            )
        }
            </table>
        </body>
        </html>
        """


class SlackNotification(NotificationChannel):
    """Slack notification channel."""

    def __init__(self, name: str, webhook_url: str, channel: str = "#alerts"):
        super().__init__(name)
        self.webhook_url = webhook_url
        self.channel = channel

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not HAS_REQUESTS or requests is None:
            # Log error instead of print for production systems
            return False

        try:
            # Create Slack message payload
            color = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "good",
            }.get(alert.severity, "#808080")

            payload = {
                "channel": self.channel,
                "username": "Opifex Alerts",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True,
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                        ]
                        + (
                            [
                                {
                                    "title": "Value",
                                    "value": str(alert.value),
                                    "short": True,
                                }
                            ]
                            if alert.value
                            else []
                        ),
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            return True

        except Exception:
            # Log error instead of print for production systems
            return False


class AlertManager:
    """Central alert management system."""

    def __init__(self):
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.notification_channels: list[NotificationChannel] = []
        self.suppressed_rules: dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> None:
        """Remove alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
        if rule_name in self.active_alerts:
            del self.active_alerts[rule_name]

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self.notification_channels.append(channel)

    def suppress_rule(self, rule_name: str, duration_minutes: int = 60) -> None:
        """Suppress alerts for a rule temporarily."""
        self.suppressed_rules[rule_name] = datetime.now(UTC) + timedelta(
            minutes=duration_minutes
        )

    def is_rule_suppressed(self, rule_name: str) -> bool:
        """Check if rule is currently suppressed."""
        if rule_name not in self.suppressed_rules:
            return False

        if datetime.now(UTC) > self.suppressed_rules[rule_name]:
            del self.suppressed_rules[rule_name]
            return False

        return True

    def fire_alert(
        self,
        rule_name: str,
        message: str,
        value: float | None = None,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> bool:
        """Fire an alert."""
        if rule_name not in self.rules:
            return False

        rule = self.rules[rule_name]

        if not rule.enabled or self.is_rule_suppressed(rule_name):
            return False

        # Check if alert is already active
        if rule_name in self.active_alerts:
            return False

        # Create alert
        alert = Alert(
            rule_name=rule_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            timestamp=datetime.now(UTC),
            labels={**rule.labels, **(labels or {})},
            annotations={**rule.annotations, **(annotations or {})},
            value=value,
        )

        # Store active alert
        self.active_alerts[rule_name] = alert

        # Send notifications
        for channel in self.notification_channels:
            with suppress(Exception):
                channel.send_alert(alert)

        return True

    def resolve_alert(self, rule_name: str) -> bool:
        """Resolve an active alert."""
        if rule_name not in self.active_alerts:
            return False

        alert = self.active_alerts[rule_name]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_timestamp = datetime.now(UTC)

        # Send resolution notifications
        for channel in self.notification_channels:
            with suppress(Exception):
                channel.send_alert(alert)

        # Remove from active alerts
        del self.active_alerts[rule_name]
        return True

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def export_prometheus_rules(self, output_file: str) -> None:
        """Export rules to Prometheus format."""
        groups = [
            {
                "name": "opifex_alerts",
                "rules": [
                    rule.to_prometheus_rule()
                    for rule in self.rules.values()
                    if rule.enabled
                ],
            }
        ]

        prometheus_config = {"groups": groups}

        with open(output_file, "w") as f:
            json.dump(prometheus_config, f, indent=2)


# Predefined alert rules for scientific computing
def get_system_alert_rules() -> list[AlertRule]:
    """Get system resource alert rules."""
    return [
        AlertRule(
            name="HighCPUUsage",
            description="CPU usage is above 80%",
            query="100 - (avg(rate(cpu_idle_seconds_total[5m])) * 100)",
            severity=AlertSeverity.WARNING,
            threshold=80,
            duration="5m",
            labels={"category": "system"},
            annotations={"summary": "High CPU usage detected"},
        ),
        AlertRule(
            name="HighMemoryUsage",
            description="Memory usage is above 85%",
            query=(
                "(1 - (node_memory_MemAvailable_bytes / "
                "node_memory_MemTotal_bytes)) * 100"
            ),
            severity=AlertSeverity.WARNING,
            threshold=85,
            duration="5m",
            labels={"category": "system"},
            annotations={"summary": "High memory usage detected"},
        ),
        AlertRule(
            name="CriticalMemoryUsage",
            description="Memory usage is above 95%",
            query=(
                "(1 - (node_memory_MemAvailable_bytes / "
                "node_memory_MemTotal_bytes)) * 100"
            ),
            severity=AlertSeverity.CRITICAL,
            threshold=95,
            duration="2m",
            labels={"category": "system"},
            annotations={
                "summary": "Critical memory usage - immediate attention required"
            },
        ),
        AlertRule(
            name="HighDiskUsage",
            description="Disk usage is above 90%",
            query=(
                "(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100"
            ),
            severity=AlertSeverity.WARNING,
            threshold=90,
            duration="10m",
            labels={"category": "system"},
            annotations={"summary": "High disk usage detected"},
        ),
    ]


def get_training_alert_rules() -> list[AlertRule]:
    """Get training-specific alert rules."""
    return [
        AlertRule(
            name="TrainingLossSpike",
            description="Training loss increased significantly",
            query="increase(opifex_training_loss[10m])",
            severity=AlertSeverity.WARNING,
            threshold=0.1,  # 10% increase
            duration="5m",
            labels={"category": "training"},
            annotations={"summary": "Training loss spike detected"},
        ),
        AlertRule(
            name="TrainingStalled",
            description="No training progress for extended period",
            query="time() - opifex_training_last_update_timestamp",
            severity=AlertSeverity.WARNING,
            threshold=1800,  # 30 minutes
            duration="5m",
            labels={"category": "training"},
            annotations={"summary": "Training appears to have stalled"},
        ),
        AlertRule(
            name="ModelAccuracyDrop",
            description="Model accuracy dropped significantly",
            query="decrease(opifex_training_accuracy[30m])",
            severity=AlertSeverity.WARNING,
            threshold=0.05,  # 5% drop
            duration="10m",
            labels={"category": "training"},
            annotations={"summary": "Model accuracy drop detected"},
        ),
        AlertRule(
            name="GPUMemoryExhaustion",
            description="GPU memory usage is critically high",
            query="opifex_gpu_memory_usage_mb / opifex_gpu_memory_total_mb * 100",
            severity=AlertSeverity.CRITICAL,
            threshold=95,
            duration="2m",
            labels={"category": "gpu"},
            annotations={"summary": "GPU memory exhaustion - training may fail"},
        ),
    ]


def get_inference_alert_rules() -> list[AlertRule]:
    """Get inference-specific alert rules."""
    return [
        AlertRule(
            name="HighInferenceLatency",
            description="Inference latency is too high",
            query=(
                "histogram_quantile(0.95, "
                "rate(opifex_inference_duration_seconds_bucket[5m]))"
            ),
            severity=AlertSeverity.WARNING,
            threshold=1.0,  # 1 second
            duration="5m",
            labels={"category": "inference"},
            annotations={"summary": "High inference latency detected"},
        ),
        AlertRule(
            name="InferenceErrorRate",
            description="High rate of inference errors",
            query=(
                "rate(opifex_inference_errors_total[5m]) / "
                "rate(opifex_inference_requests_total[5m]) * 100"
            ),
            severity=AlertSeverity.WARNING,
            threshold=5,  # 5% error rate
            duration="5m",
            labels={"category": "inference"},
            annotations={"summary": "High inference error rate detected"},
        ),
        AlertRule(
            name="ModelUnavailable",
            description="Model endpoint is not responding",
            query='up{job="opifex-model-server"}',
            severity=AlertSeverity.CRITICAL,
            threshold=0,  # Down
            duration="1m",
            labels={"category": "availability"},
            annotations={"summary": "Model server is unavailable"},
        ),
    ]


def setup_default_alerts(alert_manager: AlertManager) -> None:
    """Set up default alert rules."""
    all_rules = (
        get_system_alert_rules()
        + get_training_alert_rules()
        + get_inference_alert_rules()
    )

    for rule in all_rules:
        alert_manager.add_rule(rule)


def create_alert_manager(
    email_config: dict[str, Any] | None = None,
    slack_config: dict[str, Any] | None = None,
) -> AlertManager:
    """Create and configure alert manager."""
    manager = AlertManager()

    # Add notification channels
    if email_config:
        email_channel = EmailNotification(
            name="email",
            smtp_server=email_config["smtp_server"],
            smtp_port=email_config["smtp_port"],
            username=email_config["username"],
            password=email_config["password"],
            to_addresses=email_config["to_addresses"],
            from_address=email_config.get("from_address"),
        )
        manager.add_notification_channel(email_channel)

    if slack_config:
        slack_channel = SlackNotification(
            name="slack",
            webhook_url=slack_config["webhook_url"],
            channel=slack_config.get("channel", "#alerts"),
        )
        manager.add_notification_channel(slack_channel)

    # Set up default rules
    setup_default_alerts(manager)

    return manager


# Global alert manager instance
_global_alert_manager: AlertManager | None = None


def get_global_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _global_alert_manager  # noqa: PLW0603
    if _global_alert_manager is None:
        _global_alert_manager = create_alert_manager()
    return _global_alert_manager


def setup_global_alert_manager(
    email_config: dict[str, Any] | None = None,
    slack_config: dict[str, Any] | None = None,
) -> AlertManager:
    """Set up global alert manager."""
    global _global_alert_manager  # noqa: PLW0603
    _global_alert_manager = create_alert_manager(email_config, slack_config)
    return _global_alert_manager


# Convenience functions for common alerts
def alert_high_cpu(value: float) -> bool:
    """Fire high CPU usage alert."""
    manager = get_global_alert_manager()
    return manager.fire_alert("HighCPUUsage", f"CPU usage is {value:.1f}%", value=value)


def alert_high_memory(value: float) -> bool:
    """Fire high memory usage alert."""
    manager = get_global_alert_manager()
    rule_name = "CriticalMemoryUsage" if value > 95 else "HighMemoryUsage"
    return manager.fire_alert(rule_name, f"Memory usage is {value:.1f}%", value=value)


def alert_training_loss_spike(current_loss: float, previous_loss: float) -> bool:
    """Fire training loss spike alert."""
    manager = get_global_alert_manager()
    increase_pct = ((current_loss - previous_loss) / previous_loss) * 100
    return manager.fire_alert(
        "TrainingLossSpike",
        (
            f"Training loss increased by {increase_pct:.1f}% "
            f"(from {previous_loss:.4f} to {current_loss:.4f})"
        ),
        value=increase_pct,
    )


def alert_inference_latency(latency: float) -> bool:
    """Fire high inference latency alert."""
    manager = get_global_alert_manager()
    return manager.fire_alert(
        "HighInferenceLatency",
        f"Inference latency is {latency:.2f} seconds",
        value=latency,
    )
