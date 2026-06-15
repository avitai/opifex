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

# rMD17 is a real downloaded dataset wrapped in datarax pipelines (not a
# Grain on-demand PDE source); its factory lives with the source module but
# is re-exported here alongside the other ``create_*_loader`` factories.
from opifex.data.sources.rmd17_source import create_rmd17_loader


__all__ = [
    "create_burgers_loader",
    "create_darcy_loader",
    "create_diffusion_loader",
    "create_navier_stokes_loader",
    "create_rmd17_loader",
    "create_shallow_water_loader",
]
