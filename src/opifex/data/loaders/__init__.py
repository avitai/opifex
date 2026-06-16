"""datarax loader factories for Opifex's synthetic PDE datasets.

Each ``create_*_loader`` generates a dataset via jit+vmap and serves it through
datarax ``MemorySource`` + ``Pipeline``, returning a :class:`PDELoaders`
(train + val) — the same pattern as :func:`create_rmd17_loader`.
"""

from opifex.data.loaders.factory import (
    create_burgers_loader,
    create_darcy_loader,
    create_diffusion_loader,
    create_navier_stokes_loader,
    create_shallow_water_loader,
    PDELoaders,
)
from opifex.data.sources.rmd17_source import create_rmd17_loader


__all__ = [
    "PDELoaders",
    "create_burgers_loader",
    "create_darcy_loader",
    "create_diffusion_loader",
    "create_navier_stokes_loader",
    "create_rmd17_loader",
    "create_shallow_water_loader",
]
