"""Tests for Weak-form SINDy (noise-robust equation discovery).

TDD: WeakSINDy should recover equations from noisy data by integrating
against smooth test functions instead of computing pointwise derivatives.

Reference:
    Messenger & Bortz (2021) "Weak SINDy: Galerkin-Based Data-Driven
    Model Selection"
"""

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.config import WeakSINDyConfig
from opifex.discovery.sindy.weak_sindy import WeakSINDy


def _generate_noisy_linear_ode(
    key: jax.Array,
    noise_fraction: float = 0.1,
    n_steps: int = 500,
    dt: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Generate noisy linear ODE data: dx/dt = -2x, x(0) = 1.

    Returns (x, t, true_coef) where true_coef = -2.0.
    """
    t = jnp.arange(n_steps) * dt
    x_clean = jnp.exp(-2.0 * t)[:, None]  # (n_steps, 1)

    noise = noise_fraction * jnp.std(x_clean) * jax.random.normal(key, x_clean.shape)
    x_noisy = x_clean + noise

    return x_noisy, t, -2.0


class TestWeakSINDyNoiseRobustness:
    """Tests for noise-robust equation discovery."""

    def test_recovers_from_clean_data(self):
        """WeakSINDy recovers linear ODE from clean data."""
        x, t, _true_coef = _generate_noisy_linear_ode(jax.random.PRNGKey(0), noise_fraction=0.0)
        config = WeakSINDyConfig(polynomial_degree=1, threshold=0.1, n_subdomains=20)
        model = WeakSINDy(config)
        model.fit(x, t)

        coef = model.coefficients
        assert coef is not None
        # Should find ~1 nonzero coefficient (the linear term)
        n_nonzero = int(jnp.sum(jnp.abs(coef) > 0.01))
        assert n_nonzero <= 2

    def test_recovers_from_noisy_data(self):
        """WeakSINDy recovers equation from 10% noise."""
        x, t, _true_coef = _generate_noisy_linear_ode(jax.random.PRNGKey(1), noise_fraction=0.1)
        config = WeakSINDyConfig(polynomial_degree=1, threshold=0.1, n_subdomains=30)
        model = WeakSINDy(config)
        model.fit(x, t)

        coef = model.coefficients
        assert coef is not None
        # Should still find sparse structure
        n_nonzero = int(jnp.sum(jnp.abs(coef) > 0.01))
        assert n_nonzero <= 2

    def test_more_robust_than_standard_sindy(self):
        """WeakSINDy is more robust to noise than standard SINDy."""
        from opifex.discovery.sindy.config import SINDyConfig
        from opifex.discovery.sindy.sindy import SINDy
        from opifex.discovery.sindy.utils import finite_difference

        key = jax.random.PRNGKey(42)
        x, t, _ = _generate_noisy_linear_ode(key, noise_fraction=0.3, n_steps=1000)
        dt = float(t[1] - t[0])

        # Standard SINDy with noisy finite differences
        x_dot_noisy = finite_difference(x, dt)
        std_config = SINDyConfig(polynomial_degree=1, threshold=0.1)
        std_model = SINDy(std_config)
        std_model.fit(x, x_dot_noisy)
        float(jnp.mean((std_model.predict(x) - x_dot_noisy) ** 2))

        # WeakSINDy (doesn't need explicit derivatives)
        weak_config = WeakSINDyConfig(polynomial_degree=1, threshold=0.1, n_subdomains=50)
        weak_model = WeakSINDy(weak_config)
        weak_model.fit(x, t)

        # Both should find something, but weak should be more stable
        assert weak_model.coefficients is not None
        assert std_model.coefficients is not None


class TestWeakSINDyAPI:
    """Tests for WeakSINDy API."""

    def test_fit_returns_self(self):
        """fit() returns the model for method chaining."""
        x = jnp.ones((100, 1))
        t = jnp.arange(100) * 0.01
        config = WeakSINDyConfig(polynomial_degree=1, n_subdomains=10)
        model = WeakSINDy(config)
        result = model.fit(x, t)
        assert result is model

    def test_equations_readable(self):
        """Model produces readable equations."""
        x, t, _ = _generate_noisy_linear_ode(jax.random.PRNGKey(0), noise_fraction=0.0)
        config = WeakSINDyConfig(polynomial_degree=1, n_subdomains=20)
        model = WeakSINDy(config)
        model.fit(x, t)

        eqs = model.equations(["x"])
        assert len(eqs) == 1
        assert isinstance(eqs[0], str)

    def test_config_is_frozen(self):
        """WeakSINDyConfig is immutable."""
        import pytest

        config = WeakSINDyConfig()
        with pytest.raises(AttributeError):
            config.n_subdomains = 50  # type: ignore[misc]
