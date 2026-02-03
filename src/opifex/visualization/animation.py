"""
Animation utilities for Opifex visualization.

Provides tools for creating animations of physics simulations and training progress.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)


def create_physics_animation(
    field_sequence: jax.Array,
    time_points: jax.Array | None = None,
    title: str = "Physics Animation",
    save_path: str | None = None,
) -> Any:
    """
    Create animation of physics field evolution.

    Args:
        field_sequence: Array of shape (time, height, width) or
            (time, height, width, channels)
        time_points: Optional time values for each frame
        title: Animation title
        save_path: Optional path to save animation

    Returns:
        Animation object
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        logger.warning("Matplotlib not available for animation")
        return None

    if field_sequence.ndim == 4:
        # Multi-channel, use first channel
        field_sequence = field_sequence[..., 0]

    n_frames = field_sequence.shape[0]

    if time_points is None:
        time_points = jnp.arange(n_frames)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize plot
    im = ax.imshow(field_sequence[0], animated=True, origin="lower")
    ax.set_title(f"{title} - t={time_points[0]:.3f}")
    plt.colorbar(im, ax=ax)

    def animate(frame):
        im.set_array(field_sequence[frame])
        ax.set_title(f"{title} - t={time_points[frame]:.3f}")
        return [im]

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=100, blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=10)
        logger.info(f"Animation saved to {save_path}")

    return anim
