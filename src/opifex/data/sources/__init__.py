"""
Grain data sources for Opifex PDEs.

This module provides Grain-compliant data sources for various PDE problems,
plus scientific data sources extending datarax's DataSourceModule.
"""

from opifex.data.sources.burgers_source import BurgersDataSource
from opifex.data.sources.darcy_source import DarcyDataSource
from opifex.data.sources.diffusion_source import DiffusionDataSource
from opifex.data.sources.navier_stokes_source import NavierStokesDataSource
from opifex.data.sources.qh9_blocks import (
    cut_fock_to_blocks,
    reconstruct_fock_from_blocks,
)
from opifex.data.sources.qh9_padded_source import (
    create_qh9_padded_sources,
    iterate_padded_batches,
    QH9PaddedConfig,
    QH9PaddedSource,
    QH9PaddedSplits,
)
from opifex.data.sources.qh9_source import (
    load_qh9_data,
    matrix_transform_def2svp,
    qh9_random_split,
    QH9Data,
    QH9Example,
    read_qh9_sqlite,
)
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
    "QH9Data",
    "QH9Example",
    "QH9PaddedConfig",
    "QH9PaddedSource",
    "QH9PaddedSplits",
    "RMD17Config",
    "RMD17Data",
    "RMD17Loaders",
    "ShallowWaterDataSource",
    "VTKMeshConfig",
    "VTKMeshSource",
    "create_qh9_padded_sources",
    "create_rmd17_loader",
    "cut_fock_to_blocks",
    "download_rmd17_molecule",
    "iterate_padded_batches",
    "load_qh9_data",
    "matrix_transform_def2svp",
    "parse_rmd17_npz",
    "qh9_random_split",
    "read_qh9_sqlite",
    "reconstruct_fock_from_blocks",
]
