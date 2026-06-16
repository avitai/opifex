"""Data sources for Opifex.

Synthetic PDE datasets are generated eagerly via jit+vmap
(:mod:`opifex.data.sources.pde_generation`) and served through datarax
pipelines by :mod:`opifex.data.loaders`. Atomistic/quantum and file-backed
sources (rMD17, QH9, PDEBench, VTK) extend datarax's ``DataSourceModule``.
"""

from opifex.data.sources.pde_generation import (
    generate_burgers,
    generate_darcy,
    generate_diffusion,
    generate_navier_stokes,
    generate_shallow_water,
)
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


__all__ = [
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
    "VTKMeshConfig",
    "VTKMeshSource",
    "create_qh9_padded_sources",
    "create_rmd17_loader",
    "cut_fock_to_blocks",
    "download_rmd17_molecule",
    "generate_burgers",
    "generate_darcy",
    "generate_diffusion",
    "generate_navier_stokes",
    "generate_shallow_water",
    "iterate_padded_batches",
    "load_qh9_data",
    "matrix_transform_def2svp",
    "parse_rmd17_npz",
    "qh9_random_split",
    "read_qh9_sqlite",
    "reconstruct_fock_from_blocks",
]
