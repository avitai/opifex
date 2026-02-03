"""
Physics-Informed Neural Networks (PINNs)

This module implements specialized PINN architectures for solving partial
differential equations with physics constraints.
"""

from opifex.neural.pinns.multi_scale import (
    create_heat_equation_pinn,
    create_navier_stokes_pinn,
    MultiScalePINN,
)


__all__ = ["MultiScalePINN", "create_heat_equation_pinn", "create_navier_stokes_pinn"]
