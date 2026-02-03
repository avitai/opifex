"""
Visualization package for Opifex framework.

Provides comprehensive visualization tools for scientific computing including
field plotting, animation, and performance analysis.
"""

from opifex.visualization.animation import create_physics_animation
from opifex.visualization.field_plotting import (
    plot_2d_field,
    plot_field_comparison,
    plot_field_evolution,
    plot_spectral_analysis,
    plot_vector_field,
)
from opifex.visualization.performance import (
    plot_flops_analysis,
    plot_memory_usage,
    plot_model_complexity_comparison,
)


__all__ = [
    # Animation tools
    "create_physics_animation",
    # Field plotting
    "plot_2d_field",
    "plot_field_comparison",
    "plot_field_evolution",
    # Performance visualization
    "plot_flops_analysis",
    "plot_memory_usage",
    "plot_model_complexity_comparison",
    "plot_spectral_analysis",
    "plot_vector_field",
]
