"""Tests for differential operators on grid fields.

TDD: Operators must be numerically correct on known analytical solutions.
All operators must be JIT-compatible and differentiable.
"""

import jax.numpy as jnp

from opifex.fields.field import Box, CenteredGrid, Extrapolation
from opifex.fields.operations import curl_2d, divergence, gradient, laplacian


class TestGradient:
    """Tests for spatial gradient operator."""

    def test_constant_has_zero_gradient(self):
        """Gradient of constant field is zero."""
        values = jnp.ones((32, 32)) * 5.0
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        grad = gradient(field)
        assert grad.values.shape == (32, 32, 2)
        assert jnp.allclose(grad.values, 0.0, atol=1e-10)

    def test_linear_x_has_constant_gradient(self):
        """Gradient of u(x,y) = x is [1, 0]."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = coords[..., 0]  # u = x
        field = CenteredGrid(values, box, Extrapolation.NEUMANN)

        grad = gradient(field)
        # Interior gradient should be ~[1, 0]
        interior = grad.values[5:-5, 5:-5]
        assert jnp.allclose(interior[..., 0], 1.0, atol=0.02)
        assert jnp.allclose(interior[..., 1], 0.0, atol=0.02)

    def test_gradient_of_sin(self):
        """Gradient of sin(2πx) ≈ 2π·cos(2πx)."""
        n = 128
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = jnp.sin(2 * jnp.pi * coords[..., 0])
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        grad = gradient(field)
        expected_gx = 2 * jnp.pi * jnp.cos(2 * jnp.pi * coords[..., 0])
        assert jnp.allclose(grad.values[..., 0], expected_gx, atol=0.1)


class TestLaplacian:
    """Tests for Laplacian operator."""

    def test_constant_has_zero_laplacian(self):
        """Laplacian of constant is zero."""
        values = jnp.ones((32, 32)) * 3.0
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        lap = laplacian(field)
        assert jnp.allclose(lap.values, 0.0, atol=1e-10)

    def test_quadratic_has_constant_laplacian(self):
        """Laplacian of u(x,y) = x² + y² is 4."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = coords[..., 0] ** 2 + coords[..., 1] ** 2
        field = CenteredGrid(values, box, Extrapolation.NEUMANN)

        lap = laplacian(field)
        # Interior should be ~4.0 (sum of second derivatives)
        interior = lap.values[5:-5, 5:-5]
        assert jnp.allclose(interior, 4.0, atol=0.1)

    def test_sin_laplacian(self):
        """Laplacian of sin(2πx)sin(2πy) ≈ -8π²·sin(2πx)sin(2πy)."""
        n = 128
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = jnp.sin(2 * jnp.pi * coords[..., 0]) * jnp.sin(2 * jnp.pi * coords[..., 1])
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        lap = laplacian(field)
        expected = -8 * jnp.pi**2 * values
        assert jnp.allclose(lap.values, expected, atol=5.0)  # FD error


class TestDivergence:
    """Tests for divergence operator."""

    def test_constant_vector_has_zero_divergence(self):
        """Divergence of constant vector field is zero."""
        values = jnp.ones((32, 32, 2))
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        div = divergence(field)
        assert jnp.allclose(div.values, 0.0, atol=1e-10)

    def test_identity_field_has_constant_divergence(self):
        """Divergence of v = (x, y) is 2."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = coords  # v = (x, y)
        field = CenteredGrid(values, box, Extrapolation.NEUMANN)

        div = divergence(field)
        interior = div.values[5:-5, 5:-5]
        assert jnp.allclose(interior, 2.0, atol=0.05)


class TestCurl2D:
    """Tests for 2D curl (vorticity)."""

    def test_irrotational_field(self):
        """Curl of gradient field is zero (irrotational)."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        # v = ∇(x² + y²) = (2x, 2y) — irrotational
        values = 2.0 * coords
        field = CenteredGrid(values, box, Extrapolation.NEUMANN)

        curl = curl_2d(field)
        interior = curl.values[5:-5, 5:-5]
        assert jnp.allclose(interior, 0.0, atol=0.1)

    def test_rigid_rotation(self):
        """Curl of rigid rotation v = (-y, x) is 2."""
        n = 64
        box = Box(lower=(-1.0, -1.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = jnp.stack([-coords[..., 1], coords[..., 0]], axis=-1)
        field = CenteredGrid(values, box, Extrapolation.NEUMANN)

        curl = curl_2d(field)
        interior = curl.values[5:-5, 5:-5]
        assert jnp.allclose(interior, 2.0, atol=0.1)
