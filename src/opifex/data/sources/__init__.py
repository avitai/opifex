"""Data sources for Opifex.

Synthetic PDE datasets are generated eagerly via jit+vmap
(:mod:`opifex.data.sources.pde_generation`) and served through datarax
pipelines by :mod:`opifex.data.loaders`. Atomistic/quantum and file-backed
sources (rMD17, QH9, PDEBench, VTK) extend datarax's ``DataSourceModule``.

Every export is listed in ``__init__.pyi``, which type checkers read. The sources that need no
optional dependency are imported with the package; the PDEBench names load
:mod:`opifex.data.sources.pdebench` on first use (scientific-python SPEC 1, as
``datarax.sources`` loads its optional integrations), so ``import opifex.data.sources`` does not
import ``h5py``.
"""

import lazy_loader

from opifex.data.sources.pde_generation import (
    generate_burgers as generate_burgers,
    generate_darcy as generate_darcy,
    generate_diffusion as generate_diffusion,
    generate_navier_stokes as generate_navier_stokes,
    generate_shallow_water as generate_shallow_water,
)
from opifex.data.sources.qh9_blocks import (
    cut_fock_to_blocks as cut_fock_to_blocks,
    reconstruct_fock_from_blocks as reconstruct_fock_from_blocks,
)
from opifex.data.sources.qh9_padded_source import (
    create_qh9_padded_sources as create_qh9_padded_sources,
    iterate_padded_batches as iterate_padded_batches,
    QH9PaddedConfig as QH9PaddedConfig,
    QH9PaddedSource as QH9PaddedSource,
    QH9PaddedSplits as QH9PaddedSplits,
)
from opifex.data.sources.qh9_source import (
    load_qh9_data as load_qh9_data,
    matrix_transform_def2svp as matrix_transform_def2svp,
    qh9_random_split as qh9_random_split,
    QH9Data as QH9Data,
    QH9Example as QH9Example,
    read_qh9_sqlite as read_qh9_sqlite,
)
from opifex.data.sources.rmd17_source import (
    create_rmd17_loader as create_rmd17_loader,
    download_rmd17_molecule as download_rmd17_molecule,
    parse_rmd17_npz as parse_rmd17_npz,
    RMD17Config as RMD17Config,
    RMD17Data as RMD17Data,
    RMD17Loaders as RMD17Loaders,
)
from opifex.data.sources.scientific import (
    create_vtk_mesh_loader as create_vtk_mesh_loader,
    VTKMeshConfig as VTKMeshConfig,
    VTKMeshSource as VTKMeshSource,
)


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
