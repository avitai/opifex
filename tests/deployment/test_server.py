"""Tests for model serving server components.

Tests the AppState container and verifies the FastAPI app is built by the
``create_app`` factory (import is side-effect-free — there is no module-level
app). Full endpoint testing requires running the server and is out of scope
for unit tests.
"""

from opifex.deployment.server import AppState, create_app


class TestAppState:
    """Tests for AppState container."""

    def test_initial_state_is_none(self):
        """All state attributes start as None."""
        state = AppState()
        assert state.config is None
        assert state.inference_engine is None
        assert state.model_registry is None


class TestFastAPIApp:
    """Tests for FastAPI app configuration produced by ``create_app``."""

    def test_app_title(self):
        """App has correct title."""
        assert "Opifex" in create_app().title

    def test_app_has_docs(self):
        """App exposes docs endpoint."""
        assert create_app().docs_url == "/docs"

    def test_app_has_redoc(self):
        """App exposes redoc endpoint."""
        assert create_app().redoc_url == "/redoc"

    def test_app_state_attached(self):
        """The factory attaches a fresh ``AppState`` to ``app.state``."""
        app = create_app()
        assert isinstance(app.state.app_state, AppState)
        assert app.state.app_state.inference_engine is None
