"""Tests for field abstractions.

TDD: Field types must be JAX pytrees, support physical coordinates,
and carry domain metadata.
"""

import jax
import jax.numpy as jnp

from opifex.fields.field import Box, CenteredGrid, Extrapolation


class TestBox:
    """Tests for domain Box."""

    def test_unit_box_2d(self):
        """Unit box [0,1]^2."""
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        assert box.spatial_dim == 2
        assert jnp.allclose(box.size, jnp.array([1.0, 1.0]))

    def test_box_center(self):
        """Box center is computed correctly."""
        box = Box(lower=(-1.0, -2.0), upper=(1.0, 2.0))
        assert jnp.allclose(box.center, jnp.array([0.0, 0.0]))

    def test_box_3d(self):
        """3D box."""
        box = Box(lower=(0.0, 0.0, 0.0), upper=(1.0, 2.0, 3.0))
        assert box.spatial_dim == 3


class TestExtrapolation:
    """Tests for boundary extrapolation."""

    def test_zero_extrapolation(self):
        """Zero extrapolation pads with zeros."""
        ext = Extrapolation.ZERO
        assert ext.value == "zero"

    def test_periodic_extrapolation(self):
        """Periodic extrapolation wraps around."""
        ext = Extrapolation.PERIODIC
        assert ext.value == "periodic"


class TestCenteredGrid:
    """Tests for CenteredGrid field type."""

    def test_create_from_values(self):
        """Create grid from explicit values."""
        values = jnp.ones((32, 32))
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        grid = CenteredGrid(values=values, box=box)
        assert grid.shape == (32, 32)
        assert grid.spatial_dim == 2

    def test_resolution(self):
        """Grid resolution matches values shape."""
        values = jnp.zeros((64, 128))
        box = Box(lower=(0.0, 0.0), upper=(1.0, 2.0))
        grid = CenteredGrid(values=values, box=box)
        assert grid.resolution == (64, 128)

    def test_cell_size(self):
        """Cell size is box_size / resolution."""
        values = jnp.zeros((10, 20))
        box = Box(lower=(0.0, 0.0), upper=(1.0, 2.0))
        grid = CenteredGrid(values=values, box=box)
        dx = grid.dx
        assert jnp.allclose(dx, jnp.array([0.1, 0.1]))

    def test_is_pytree(self):
        """CenteredGrid works as a JAX pytree."""
        values = jnp.ones((8, 8))
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        grid = CenteredGrid(values=values, box=box)

        # Should be flattenable and unflattenable
        leaves = jax.tree_util.tree_leaves(grid)
        assert len(leaves) >= 1

    def test_arithmetic(self):
        """Grid supports element-wise arithmetic."""
        v1 = jnp.ones((8, 8))
        v2 = jnp.ones((8, 8)) * 2.0
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))

        g1 = CenteredGrid(values=v1, box=box)
        g2 = CenteredGrid(values=v2, box=box)

        g3 = g1 + g2
        assert jnp.allclose(g3.values, 3.0 * jnp.ones((8, 8)))

    def test_scalar_multiply(self):
        """Grid supports scalar multiplication."""
        values = jnp.ones((8, 8)) * 3.0
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        grid = CenteredGrid(values=values, box=box)
        scaled = grid * 2.0
        assert jnp.allclose(scaled.values, 6.0 * jnp.ones((8, 8)))
