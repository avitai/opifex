"""Tests for Grafana dashboard management."""

from opifex.deployment.monitoring.dashboards import Dashboard, Panel


class TestPanel:
    """Tests for Panel dataclass."""

    def test_create_default_panel(self):
        """Create panel with minimal required fields."""
        panel = Panel(id=1, title="Test Panel")
        assert panel.id == 1
        assert panel.title == "Test Panel"
        assert panel.type == "graph"
        assert panel.datasource == "prometheus"

    def test_to_dict_converts_snake_to_camel(self):
        """to_dict converts snake_case keys to camelCase for Grafana."""
        panel = Panel(id=1, title="Test", grid_pos={"h": 8, "w": 12, "x": 0, "y": 0})
        d = panel.to_dict()
        assert "gridPos" in d
        assert "grid_pos" not in d
        assert "fieldConfig" in d
        assert "yAxes" in d
        assert "xAxis" in d

    def test_to_dict_sets_default_grid_pos(self):
        """to_dict fills default grid position when empty."""
        panel = Panel(id=1, title="Test", height=300, span=12)
        d = panel.to_dict()
        assert d["gridPos"]["w"] == 12
        assert d["gridPos"]["h"] == 20  # 300 // 15

    def test_to_dict_sets_default_tooltip(self):
        """to_dict fills default tooltip when empty."""
        panel = Panel(id=1, title="Test")
        d = panel.to_dict()
        assert d["tooltip"]["shared"] is True

    def test_to_dict_sets_default_legend(self):
        """to_dict fills default legend when empty."""
        panel = Panel(id=1, title="Test")
        d = panel.to_dict()
        assert d["legend"]["show"] is True

    def test_panel_with_targets(self):
        """Panel accepts Prometheus query targets."""
        targets = [{"expr": "rate(requests_total[5m])", "legendFormat": "RPS"}]
        panel = Panel(id=1, title="Requests", targets=targets)
        assert len(panel.targets) == 1
        assert panel.targets[0]["expr"] == "rate(requests_total[5m])"


class TestDashboard:
    """Tests for Dashboard dataclass."""

    def test_create_default_dashboard(self):
        """Create dashboard with defaults."""
        db = Dashboard()
        assert db.id is None
        assert db.panels == []

    def test_dashboard_with_title(self):
        """Dashboard accepts title."""
        db = Dashboard(title="Neural Operator Metrics")
        assert db.title == "Neural Operator Metrics"

    def test_add_panels(self):
        """Dashboard holds multiple panels."""
        panels = [Panel(id=1, title="Loss"), Panel(id=2, title="Throughput")]
        db = Dashboard(title="Training", panels=panels)
        assert len(db.panels) == 2
