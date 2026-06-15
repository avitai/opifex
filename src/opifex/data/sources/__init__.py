"""
Grain data sources for Opifex PDEs.

This module provides Grain-compliant data sources for various PDE problems,
plus scientific data sources extending datarax's DataSourceModule.
"""

from opifex.data.sources.burgers_source import BurgersDataSource
from opifex.data.sources.darcy_source import DarcyDataSource
from opifex.data.sources.diffusion_source import DiffusionDataSource
from opifex.data.sources.navier_stokes_source import NavierStokesDataSource
from opifex.data.sources.rmd17_source import (
    create_rmd17_loader,
    download_rmd17_molecule,
    parse_rmd17_npz,
    RMD17Config,
    RMD17Data,
    RMD17Loaders,
)
from opifex.data.sources.scientific import (
    PDEBenchConfig,
    PDEBenchSource,
    VTKMeshConfig,
    VTKMeshSource,
)
from opifex.data.sources.shallow_water_source import ShallowWaterDataSource


__all__ = [
    "BurgersDataSource",
    "DarcyDataSource",
    "DiffusionDataSource",
    "NavierStokesDataSource",
    "PDEBenchConfig",
    "PDEBenchSource",
    "RMD17Config",
    "RMD17Data",
    "RMD17Loaders",
    "ShallowWaterDataSource",
    "VTKMeshConfig",
    "VTKMeshSource",
    "create_rmd17_loader",
    "download_rmd17_molecule",
    "parse_rmd17_npz",
]
