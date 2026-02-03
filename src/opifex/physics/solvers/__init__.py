"""
Opifex PDE Solvers

JAX-native PDE solvers for various physical equations and systems.
Optimized for neural operator training data generation and validation.
"""

from .burgers import Burgers2DSolver
from .diffusion_advection import solve_diffusion_advection_2d
from .shallow_water import solve_shallow_water_2d


__all__ = [
    "Burgers2DSolver",
    "solve_diffusion_advection_2d",
    "solve_shallow_water_2d",
]
