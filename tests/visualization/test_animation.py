"""Tests for animation visualization functionality.

This module tests the animation module, covering:
- Animation creation from field sequences
- Multi-channel input handling
- Time points integration
- File saving
"""

import jax.numpy as jnp
import matplotlib as mpl


# Use non-interactive backend for testing
mpl.use("Agg")

from opifex.visualization.animation import create_physics_animation


class TestCreatePhysicsAnimation:
    """Test create_physics_animation function."""

    def test_basic_animation_creation(self):
        """Test basic animation creation."""
        field_sequence = jnp.ones((5, 16, 16))

        anim = create_physics_animation(field_sequence)

        assert anim is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_animation_with_time_points(self):
        """Test animation with custom time points."""
        field_sequence = jnp.ones((10, 16, 16))
        time_points = jnp.linspace(0, 1, 10)

        anim = create_physics_animation(field_sequence, time_points=time_points)

        assert anim is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_animation_with_custom_title(self):
        """Test animation with custom title."""
        field_sequence = jnp.ones((5, 16, 16))

        anim = create_physics_animation(field_sequence, title="Custom Title")

        assert anim is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_animation_with_4d_input(self):
        """Test animation with 4D input (extracts first channel)."""
        field_sequence = jnp.ones((5, 16, 16, 3))

        anim = create_physics_animation(field_sequence)

        assert anim is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_animation_save_path(self, temp_directory):
        """Test saving animation to file."""
        field_sequence = jnp.ones((3, 8, 8))  # Small for fast test
        save_path = str(temp_directory / "animation.gif")

        anim = create_physics_animation(field_sequence, save_path=save_path)

        assert anim is not None
        assert (temp_directory / "animation.gif").exists()
        import matplotlib.pyplot as plt

        plt.close("all")
