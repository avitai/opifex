"""Tests for alert management system."""

from opifex.deployment.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_levels(self):
        """All severity levels exist."""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"


class TestAlertStatus:
    """Tests for AlertStatus enum."""

    def test_status_values(self):
        """All status values exist."""
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.SUPPRESSED.value == "suppressed"


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_create_rule(self):
        """Create rule with required fields."""
        rule = AlertRule(
            name="high_loss",
            description="Training loss exceeded threshold",
            query="opifex_training_loss > 10",
            severity=AlertSeverity.WARNING,
            threshold=10.0,
        )
        assert rule.name == "high_loss"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.enabled is True

    def test_rule_to_prometheus_format(self):
        """Rule converts to Prometheus-compatible dict."""
        rule = AlertRule(
            name="gpu_oom",
            description="GPU out of memory",
            query="gpu_memory_used_pct > 95",
            severity=AlertSeverity.CRITICAL,
            threshold=95.0,
            duration="1m",
        )
        prom = rule.to_prometheus_rule()
        assert "alert" in prom
        assert prom["alert"] == "gpu_oom"
        assert "expr" in prom
        assert "for" in prom

    def test_rule_defaults(self):
        """Rule has sensible defaults."""
        rule = AlertRule(
            name="test",
            description="test",
            query="up == 0",
            severity=AlertSeverity.INFO,
            threshold=0.0,
        )
        assert rule.duration == "5m"
        assert rule.labels == {}
        assert rule.annotations == {}


class TestAlertManager:
    """Tests for AlertManager."""

    def test_create_manager(self):
        """Manager initializes with empty state."""
        mgr = AlertManager()
        assert len(mgr.rules) == 0
        assert len(mgr.active_alerts) == 0

    def test_add_rule(self):
        """Rules can be registered by name."""
        mgr = AlertManager()
        rule = AlertRule(
            name="test_rule",
            description="test",
            query="metric > 5",
            severity=AlertSeverity.WARNING,
            threshold=5.0,
        )
        mgr.add_rule(rule)
        assert "test_rule" in mgr.rules
        assert mgr.rules["test_rule"].threshold == 5.0

    def test_fire_alert(self):
        """Alerts can be fired and tracked."""
        mgr = AlertManager()
        rule = AlertRule(
            name="loss_spike",
            description="Loss spiked",
            query="loss > 100",
            severity=AlertSeverity.CRITICAL,
            threshold=100.0,
        )
        mgr.add_rule(rule)
        mgr.fire_alert("loss_spike", "Loss is 150.0", value=150.0)

        assert "loss_spike" in mgr.active_alerts
        alert = mgr.active_alerts["loss_spike"]
        assert alert.status == AlertStatus.ACTIVE

    def test_resolve_alert(self):
        """Alerts can be resolved."""
        mgr = AlertManager()
        rule = AlertRule(
            name="test",
            description="test",
            query="x > 1",
            severity=AlertSeverity.INFO,
            threshold=1.0,
        )
        mgr.add_rule(rule)
        mgr.fire_alert("test", "value exceeded")
        assert "test" in mgr.active_alerts

        resolved = mgr.resolve_alert("test")
        assert resolved is True
        assert "test" not in mgr.active_alerts

    def test_resolve_nonexistent_returns_false(self):
        """Resolving nonexistent alert returns False."""
        mgr = AlertManager()
        assert mgr.resolve_alert("nonexistent") is False
