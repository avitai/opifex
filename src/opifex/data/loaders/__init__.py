"""
Grain data loader factories for Opifex PDEs.

This module provides factory functions to create configured Grain DataLoaders.
"""

from opifex.data.loaders.factory import (
    create_burgers_loader,
    create_darcy_loader,
    create_diffusion_loader,
    create_navier_stokes_loader,
    create_shallow_water_loader,
)


__all__ = [
    "create_burgers_loader",
    "create_darcy_loader",
    "create_diffusion_loader",
    "create_navier_stokes_loader",
    "create_shallow_water_loader",
]
