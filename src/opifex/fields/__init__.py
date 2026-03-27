"""Field abstractions for scientific computing on structured grids.

Provides immutable JAX pytree field types and differential operators
inspired by PhiFlow, implemented in pure JAX.

Reference:
    Holl et al. "PhiFlow: A Differentiable PDE Solving Framework"
"""

from opifex.fields.advection import maccormack, semi_lagrangian
from opifex.fields.field import Box, CenteredGrid, Extrapolation
from opifex.fields.operations import curl_2d, divergence, gradient, laplacian
from opifex.fields.pressure import pressure_solve_jacobi, pressure_solve_spectral


__all__ = [
    "Box",
    "CenteredGrid",
    "Extrapolation",
    "curl_2d",
    "divergence",
    "gradient",
    "laplacian",
    "maccormack",
    "pressure_solve_jacobi",
    "pressure_solve_spectral",
    "semi_lagrangian",
]
