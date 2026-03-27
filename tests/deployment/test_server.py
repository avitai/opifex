"""Tests for model serving server components.

Tests the AppState container and verifies the FastAPI app can be imported.
Full endpoint testing requires running the server and is out of scope
for unit tests.
"""

from opifex.deployment.server import app, AppState


class TestAppState:
    """Tests for AppState container."""

    def test_initial_state_is_none(self):
        """All state attributes start as None."""
        state = AppState()
        assert state.config is None
        assert state.inference_engine is None
        assert state.model_registry is None


class TestFastAPIApp:
    """Tests for FastAPI app configuration."""

    def test_app_title(self):
        """App has correct title."""
        assert "Opifex" in app.title

    def test_app_has_docs(self):
        """App exposes docs endpoint."""
        assert app.docs_url == "/docs"

    def test_app_has_redoc(self):
        """App exposes redoc endpoint."""
        assert app.redoc_url == "/redoc"
