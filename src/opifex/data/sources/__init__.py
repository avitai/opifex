"""
Grain data sources for Opifex PDEs.

This module provides Grain-compliant data sources for various PDE problems.
"""

from opifex.data.sources.burgers_source import BurgersDataSource
from opifex.data.sources.darcy_source import DarcyDataSource
from opifex.data.sources.diffusion_source import DiffusionDataSource
from opifex.data.sources.shallow_water_source import ShallowWaterDataSource


__all__ = [
    "BurgersDataSource",
    "DarcyDataSource",
    "DiffusionDataSource",
    "ShallowWaterDataSource",
]
