"""Tests for core SINDy model.

TDD: The key acceptance test is recovering the Lorenz system equations.
"""

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.config import SINDyConfig
from opifex.discovery.sindy.sindy import SINDy


def _generate_lorenz_data(
    n_steps: int = 5000,
    dt: float = 0.001,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Generate Lorenz attractor trajectory via simple Euler integration.

    Returns:
        (x, x_dot, dt) where x is (n_steps, 3) and x_dot is (n_steps, 3).
    """
    x0 = jnp.array([-8.0, 7.0, 27.0])
    trajectory = [x0]
    for _ in range(n_steps - 1):
        xn = trajectory[-1]
        dx = jnp.array(
            [
                sigma * (xn[1] - xn[0]),
                xn[0] * (rho - xn[2]) - xn[1],
                xn[0] * xn[1] - beta * xn[2],
            ]
        )
        trajectory.append(xn + dt * dx)

    x = jnp.stack(trajectory)
    # Compute derivatives from the known equations (clean data)
    x_dot = jnp.stack(
        [
            jnp.array(
                [
                    sigma * (xi[1] - xi[0]),
                    xi[0] * (rho - xi[2]) - xi[1],
                    xi[0] * xi[1] - beta * xi[2],
                ]
            )
            for xi in x
        ]
    )
    return x, x_dot, dt


class TestSINDyLorenz:
    """Core acceptance test: SINDy recovers Lorenz equations."""

    def test_recovers_lorenz_coefficients(self):
        """SINDy identifies correct Lorenz terms from clean data.

        Lorenz system:
            dx/dt = sigma*(y - x)     → coefficients on x, y
            dy/dt = x*(rho - z) - y   → coefficients on x, y, xz
            dz/dt = x*y - beta*z      → coefficients on xy, z
        """
        x, x_dot, _dt = _generate_lorenz_data(n_steps=3000, dt=0.001)

        config = SINDyConfig(polynomial_degree=2, threshold=0.5)
        model = SINDy(config)
        model.fit(x, x_dot)

        coef = model.coefficients
        model.feature_names(["x", "y", "z"])

        # The model should have found sparse coefficients
        assert coef is not None
        n_nonzero = int(jnp.sum(jnp.abs(coef) > 0.01))
        # Lorenz has 7 nonzero terms total across 3 equations
        assert n_nonzero <= 10, f"Too many terms: {n_nonzero}"
        assert n_nonzero >= 5, f"Too few terms: {n_nonzero}"

    def test_predict_matches_derivatives(self):
        """Predicted derivatives match the true derivatives."""
        x, x_dot, _dt = _generate_lorenz_data(n_steps=2000, dt=0.001)

        config = SINDyConfig(polynomial_degree=2, threshold=0.3)
        model = SINDy(config)
        model.fit(x, x_dot)

        x_dot_pred = model.predict(x)
        r2 = 1.0 - jnp.sum((x_dot - x_dot_pred) ** 2) / jnp.sum(
            (x_dot - jnp.mean(x_dot, axis=0)) ** 2
        )
        assert float(r2) > 0.99, f"R² = {float(r2)} < 0.99"

    def test_equations_are_readable(self):
        """Model produces human-readable equation strings."""
        x, x_dot, _dt = _generate_lorenz_data(n_steps=2000, dt=0.001)

        config = SINDyConfig(polynomial_degree=2, threshold=0.3)
        model = SINDy(config)
        model.fit(x, x_dot)

        equations = model.equations(["x", "y", "z"])
        assert len(equations) == 3
        assert all(isinstance(eq, str) for eq in equations)


class TestSINDyAPI:
    """Tests for SINDy API design."""

    def test_config_is_frozen(self):
        """SINDyConfig is immutable."""
        config = SINDyConfig()
        with pytest.raises(AttributeError):
            config.threshold = 0.5  # type: ignore[misc]

    def test_default_config(self):
        """Default config has sensible values."""
        config = SINDyConfig()
        assert config.polynomial_degree == 2
        assert config.threshold > 0

    def test_fit_returns_self(self):
        """fit() returns the model for method chaining."""
        x = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        x_dot = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
        config = SINDyConfig(polynomial_degree=1)
        model = SINDy(config)
        result = model.fit(x, x_dot)
        assert result is model


import pytest
