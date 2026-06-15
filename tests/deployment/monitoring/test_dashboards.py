"""Tests for Grafana dashboard management."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from opifex.deployment.monitoring.dashboards import Dashboard, GrafanaManager, Panel


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


class TestGrafanaManagerCreateDashboard:
    """Tests for GrafanaManager.create_dashboard real API call."""

    def test_posts_payload_to_grafana_api_and_returns_response(self):
        """create_dashboard POSTs the dashboard payload and returns the API JSON."""
        manager = GrafanaManager(
            grafana_url="http://grafana.example:3000/",
            api_key="secret-token",
        )
        dashboard = Dashboard(title="Neural Operators", uid="opifex-neural")
        api_response = {
            "id": 42,
            "uid": "opifex-neural",
            "url": "/d/opifex-neural/neural-operators",
            "status": "success",
            "version": 1,
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_response

        with patch(
            "opifex.deployment.monitoring.dashboards.requests.post",
            return_value=mock_response,
        ) as mock_post:
            result = manager.create_dashboard(dashboard)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == "http://grafana.example:3000/api/dashboards/db"
        sent_payload = call_args.kwargs["json"]
        assert sent_payload["overwrite"] is True
        assert sent_payload["dashboard"] == dashboard.to_dict()
        assert call_args.kwargs["headers"] == manager.headers
        mock_response.raise_for_status.assert_called_once()
        assert result == api_response

    def test_uses_basic_auth_when_no_api_key(self):
        """create_dashboard falls back to basic auth when no API key is configured."""
        manager = GrafanaManager(
            grafana_url="http://grafana.example:3000",
            username="admin",
            password="pw",  # noqa: S106
        )
        dashboard = Dashboard(title="Metrics", uid="opifex-metrics")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "uid": "opifex-metrics"}

        with patch(
            "opifex.deployment.monitoring.dashboards.requests.post",
            return_value=mock_response,
        ) as mock_post:
            manager.create_dashboard(dashboard)

        assert mock_post.call_args.kwargs["auth"] == ("admin", "pw")

    def test_raises_on_non_2xx_response(self):
        """create_dashboard raises when the Grafana API returns a non-2xx status."""
        manager = GrafanaManager(grafana_url="http://grafana.example:3000", api_key="tok")
        dashboard = Dashboard(title="Bad", uid="bad-uid")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("412 Precondition Failed")

        with (
            patch(
                "opifex.deployment.monitoring.dashboards.requests.post",
                return_value=mock_response,
            ),
            pytest.raises(requests.HTTPError),
        ):
            manager.create_dashboard(dashboard)
