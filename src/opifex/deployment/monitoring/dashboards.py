"""
Opifex Grafana Dashboard Management.

Automated dashboard creation and management for scientific computing workloads.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


# Optional dependencies with proper type checking
try:
    import requests  # type: ignore[import-untyped]

    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAS_REQUESTS = False  # type: ignore[misc]


@dataclass
class Panel:
    """Grafana panel configuration."""

    id: int
    title: str
    type: str = "graph"
    span: int = 12
    height: int = 300
    datasource: str = "prometheus"
    targets: list[dict[str, Any]] = field(default_factory=list)
    y_axes: list[dict[str, Any]] = field(default_factory=list)
    x_axis: dict[str, Any] = field(default_factory=dict)
    legend: dict[str, Any] = field(default_factory=dict)
    tooltip: dict[str, Any] = field(default_factory=dict)
    grid_pos: dict[str, Any] = field(default_factory=dict)
    field_config: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert panel to Grafana JSON format."""
        panel_dict = asdict(self)

        # Convert snake_case to camelCase for Grafana compatibility
        if "grid_pos" in panel_dict:
            panel_dict["gridPos"] = panel_dict.pop("grid_pos")
        if "field_config" in panel_dict:
            panel_dict["fieldConfig"] = panel_dict.pop("field_config")
        if "y_axes" in panel_dict:
            panel_dict["yAxes"] = panel_dict.pop("y_axes")
        if "x_axis" in panel_dict:
            panel_dict["xAxis"] = panel_dict.pop("x_axis")

        # Set default grid position if not specified
        if not panel_dict["gridPos"]:
            panel_dict["gridPos"] = {
                "h": self.height // 15,
                "w": self.span,
                "x": 0,
                "y": 0,
            }

        # Set default tooltip
        if not panel_dict["tooltip"]:
            panel_dict["tooltip"] = {
                "shared": True,
                "sort": 2,
                "value_type": "individual",
            }

        # Set default legend
        if not panel_dict["legend"]:
            panel_dict["legend"] = {
                "avg": False,
                "current": False,
                "max": False,
                "min": False,
                "show": True,
                "total": False,
                "values": False,
            }

        return panel_dict


@dataclass
class Dashboard:
    """Grafana dashboard configuration."""

    id: int | None = None
    uid: str | None = None
    title: str = "Opifex Dashboard"
    tags: list[str] = field(default_factory=lambda: ["opifex", "machine-learning"])
    timezone: str = "browser"
    panels: list[Panel] = field(default_factory=list)
    templating: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(
        default_factory=lambda: {"from": "now-1h", "to": "now"}
    )
    refresh: str = "30s"
    schema_version: int = 27
    version: int = 1
    editable: bool = True

    def add_panel(self, panel: Panel) -> None:
        """Add panel to dashboard."""
        self.panels.append(panel)

        # Auto-arrange panels in grid
        panels_per_row = 2 if panel.span <= 12 else 1
        panel_index = len(self.panels) - 1

        row = panel_index // panels_per_row
        col = panel_index % panels_per_row

        panel.grid_pos = {
            "h": panel.height // 15,
            "w": panel.span,
            "x": col * (24 // panels_per_row),
            "y": row * (panel.height // 15),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert dashboard to Grafana JSON format."""
        dashboard_dict = asdict(self)
        dashboard_dict["panels"] = [panel.to_dict() for panel in self.panels]

        # Set default templating if not specified
        if not dashboard_dict["templating"]:
            dashboard_dict["templating"] = {"list": []}

        return dashboard_dict


class GrafanaManager:
    """Grafana dashboard management system."""

    def __init__(
        self,
        grafana_url: str = "http://localhost:3000",
        api_key: str | None = None,
        username: str = "admin",
        password: str = "changeme",  # nosec # noqa: S107
    ):
        """
        Initialize Grafana manager.

        Args:
            grafana_url: Grafana server URL
            api_key: Grafana API key (preferred)
            username: Grafana username (fallback)
            password: Grafana password (fallback)
        """
        self.grafana_url = grafana_url.rstrip("/")
        self.api_key = api_key
        self.username = username
        self.password = password

        # Set up authentication headers
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            self.headers = {"Content-Type": "application/json"}
            self.auth = (self.username, self.password)

    def create_dashboard(self, dashboard: Dashboard) -> dict[str, Any]:
        """Create dashboard in Grafana."""
        # Simulate API response if requests not available
        if not HAS_REQUESTS:
            return {
                "status": "success",
                "message": f"Dashboard '{dashboard.title}' created (simulation mode)",
                "uid": dashboard.uid,
                "url": f"{self.grafana_url}/d/{dashboard.uid}",
            }

        # Implementation would use requests here with payload
        # payload = {
        #     "dashboard": dashboard.to_dict(),
        #     "overwrite": True,
        #     "message": f"Created {dashboard.title} dashboard",
        # }
        # TODO: Implement actual API call with payload
        return {"status": "created", "uid": dashboard.uid}


def create_neural_operator_dashboard() -> Dashboard:
    """Create neural operator performance dashboard."""
    dashboard = Dashboard(
        title="Opifex Neural Operators",
        tags=["opifex", "neural-operators", "performance"],
        uid="opifex-neural-operators",
    )

    # Training Loss Panel
    loss_panel = Panel(
        id=1,
        title="Training Loss",
        type="graph",
        span=12,
        targets=[
            {
                "expr": "opifex_training_loss",
                "format": "time_series",
                "legendFormat": "Loss - {{model_name}}",
                "refId": "A",
            }
        ],
    )
    dashboard.add_panel(loss_panel)

    return dashboard
