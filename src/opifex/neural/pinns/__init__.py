"""
Physics-Informed Neural Networks (PINNs)

This module implements specialized PINN architectures for solving partial
differential equations with physics constraints.
"""

from opifex.neural.pinns.gp_pinn import gp_pinn_predictive_posterior
from opifex.neural.pinns.multi_scale import (
    create_heat_equation_pinn,
    create_navier_stokes_pinn,
    create_poisson_pinn,
    MultiScalePINN,
    SimplePINN,
)


__all__ = [
    "MultiScalePINN",
    "SimplePINN",
    "create_heat_equation_pinn",
    "create_navier_stokes_pinn",
    "create_poisson_pinn",
    "gp_pinn_predictive_posterior",
]
