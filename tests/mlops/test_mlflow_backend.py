"""Tests for MLflow backend experiment tracking."""

import pytest

from opifex.mlops.experiment import ExperimentConfig, Framework, PhysicsDomain


try:
    from opifex.mlops.backends.mlflow_backend import MLflowBackend

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]


@pytest.mark.skipif(not _MLFLOW_AVAILABLE, reason="mlflow not installed")
class TestMLflowBackendInit:
    """Tests for MLflowBackend initialization."""

    def test_creates_with_local_tracking(self, tmp_path, monkeypatch):
        """Initializes with local file-based MLflow tracking."""
        tracking_uri = f"file://{tmp_path / 'mlruns'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

        config = ExperimentConfig(
            name="test_experiment",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
        )
        backend = MLflowBackend(config)
        assert backend.config.name == "test_experiment"
        assert backend.experiment_id is not None

    def test_stores_config(self, tmp_path, monkeypatch):
        """Config is accessible after init."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")

        config = ExperimentConfig(
            name="unit_test",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
        )
        backend = MLflowBackend(config)
        assert backend.config.framework == Framework.JAX

    def test_client_initialized(self, tmp_path, monkeypatch):
        """MLflow client is created on init."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")

        config = ExperimentConfig(
            name="client_test",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
        )
        backend = MLflowBackend(config)
        assert backend.client is not None
