"""
Opifex PDE Solvers

JAX-native PDE solvers for various physical equations and systems.
Optimized for neural operator training data generation and validation.
"""

from opifex.physics.solvers.burgers import Burgers2DSolver
from opifex.physics.solvers.diffusion_advection import solve_diffusion_advection_2d
from opifex.physics.solvers.navier_stokes import (
    create_double_shear_layer,
    create_lid_driven_cavity_ic,
    create_taylor_green_vortex,
    solve_navier_stokes_2d,
)
from opifex.physics.solvers.shallow_water import solve_shallow_water_2d


__all__ = [
    "Burgers2DSolver",
    "create_double_shear_layer",
    "create_lid_driven_cavity_ic",
    "create_taylor_green_vortex",
    "solve_diffusion_advection_2d",
    "solve_navier_stokes_2d",
    "solve_shallow_water_2d",
]
