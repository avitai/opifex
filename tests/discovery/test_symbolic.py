"""Tests for symbolic regression wrapper.

TDD: Tests for the PySR bridge. Since PySR requires Julia, tests use
the mock fallback when PySR is not installed.
"""

import jax
import jax.numpy as jnp

from opifex.discovery.symbolic import SymbolicRegressionConfig, SymbolicRegressor


class TestSymbolicRegressionConfig:
    """Tests for SymbolicRegressionConfig."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = SymbolicRegressionConfig()
        assert config.max_complexity == 20
        assert "+" in config.binary_operators
        assert "sin" in config.unary_operators

    def test_frozen(self):
        """Config is immutable."""
        import pytest

        config = SymbolicRegressionConfig()
        with pytest.raises(AttributeError):
            config.max_complexity = 10  # type: ignore[misc]


class TestSymbolicRegressor:
    """Tests for SymbolicRegressor."""

    def test_fit_and_predict(self):
        """Regressor fits data and produces predictions."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (100, 2))
        y = x[:, 0] ** 2 + x[:, 1]  # Simple known function

        reg = SymbolicRegressor()
        reg.fit(x, y)

        y_pred = reg.predict(x)
        assert y_pred.shape == y.shape

    def test_best_equation_is_string(self):
        """Best equation is returned as a string."""
        x = jnp.ones((50, 2))
        y = jnp.ones(50) * 2.0

        reg = SymbolicRegressor()
        reg.fit(x, y)

        eq = reg.best_equation()
        assert isinstance(eq, str)
        assert len(eq) > 0

    def test_custom_config(self):
        """Regressor respects custom configuration."""
        config = SymbolicRegressionConfig(max_complexity=5, niterations=10)
        reg = SymbolicRegressor(config=config)
        assert reg.config.max_complexity == 5
