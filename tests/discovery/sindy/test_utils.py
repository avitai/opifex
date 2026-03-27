"""Tests for SINDy numerical utilities.

TDD: Tests for differentiation and data preprocessing.
"""

import jax.numpy as jnp

from opifex.discovery.sindy.utils import finite_difference, smooth_data


class TestFiniteDifference:
    """Tests for numerical differentiation."""

    def test_constant_has_zero_derivative(self):
        """Constant function has zero derivative."""
        x = jnp.ones((100, 2))
        dt = 0.01
        x_dot = finite_difference(x, dt)
        assert jnp.allclose(x_dot, 0.0, atol=1e-10)

    def test_linear_has_constant_derivative(self):
        """Linear x(t) = t has derivative 1."""
        t = jnp.linspace(0, 1, 100)
        x = t[:, None]  # (100, 1)
        dt = t[1] - t[0]
        x_dot = finite_difference(x, dt)
        # Interior points should be ~1.0
        assert jnp.allclose(x_dot[2:-2], 1.0, atol=1e-4)

    def test_sin_derivative_is_cos(self):
        """Derivative of sin(t) ≈ cos(t)."""
        t = jnp.linspace(0, 2 * jnp.pi, 500)
        x = jnp.sin(t)[:, None]
        dt = t[1] - t[0]
        x_dot = finite_difference(x, dt)
        expected = jnp.cos(t)[:, None]
        # Check interior points (boundaries have lower accuracy)
        assert jnp.allclose(x_dot[5:-5], expected[5:-5], atol=0.01)

    def test_output_shape_matches_input(self):
        """Output has same shape as input."""
        x = jnp.ones((50, 3))
        x_dot = finite_difference(x, 0.01)
        assert x_dot.shape == x.shape

    def test_multi_feature(self):
        """Works with multiple features simultaneously."""
        t = jnp.linspace(0, 1, 100)
        x = jnp.column_stack([t, t**2])
        dt = t[1] - t[0]
        x_dot = finite_difference(x, dt)
        # dx0/dt ≈ 1, dx1/dt ≈ 2*t
        assert jnp.allclose(x_dot[5:-5, 0], 1.0, atol=0.05)


class TestSmoothData:
    """Tests for data smoothing."""

    def test_smooth_reduces_noise(self):
        """Smoothing reduces noise variance."""
        key = __import__("jax").random.PRNGKey(0)
        clean = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 200))
        noisy = clean + 0.1 * __import__("jax").random.normal(key, clean.shape)

        smoothed = smooth_data(noisy[:, None], window_size=5)

        noise_before = jnp.std(noisy - clean)
        noise_after = jnp.std(smoothed[:, 0] - clean)
        assert float(noise_after) < float(noise_before)

    def test_preserves_shape(self):
        """Smoothing preserves data shape."""
        x = jnp.ones((100, 3))
        smoothed = smooth_data(x, window_size=5)
        assert smoothed.shape == x.shape
