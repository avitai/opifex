"""datarax-backed loader factories for opifex's synthetic PDE datasets.

Each ``create_*_loader`` generates a dataset eagerly via jit+vmap
(:mod:`opifex.data.sources.pde_generation`) and serves it through datarax
``MemorySource`` + ``Pipeline``, returning a :class:`PDELoaders` (train + val)
— the same datarax pattern as :func:`opifex.data.sources.rmd17_source.create_rmd17_loader`.
The dataset contract is uniform: each batch is ``{"input": (b, C_in, *spatial),
"output": (b, C_out, *spatial)}`` (channels-first), the operator being
conditioning -> final-time solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: TC002  # pyproject dep — keep eager (see CLAUDE.md)
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from opifex.data.sources.pde_generation import (
    generate_burgers,
    generate_darcy,
    generate_diffusion,
    generate_navier_stokes,
    generate_shallow_water,
)


@dataclass(frozen=True)
class PDELoaders:
    """datarax train/val pipelines plus metadata for a synthetic PDE dataset.

    Attributes:
        train: datarax ``Pipeline`` over the training split (shuffled).
        val: datarax ``Pipeline`` over the validation split (sequential).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        resolution: Spatial resolution of the fields.
    """

    train: Pipeline
    val: Pipeline
    n_train: int
    n_val: int
    resolution: int


def _build_pipeline(
    data: dict[str, np.ndarray], *, batch_size: int, shuffle: bool, seed: int
) -> Pipeline:
    """Wrap a split's ``{"input", "output"}`` arrays in a datarax source + pipeline."""
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data=data,
        rngs=nnx.Rngs(shuffle=seed),
    )
    return Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(seed))


def _make_loaders(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
    resolution: int,
) -> PDELoaders:
    """Split a generated dataset into train/val datarax pipelines."""
    n_samples = data["input"].shape[0]
    n_val = max(1, round(n_samples * val_fraction))
    n_train = n_samples - n_val
    train = {key: value[:n_train] for key, value in data.items()}
    val = {key: value[n_train:] for key, value in data.items()}
    return PDELoaders(
        train=_build_pipeline(train, batch_size=batch_size, shuffle=True, seed=seed),
        val=_build_pipeline(val, batch_size=batch_size, shuffle=False, seed=seed + 1),
        n_train=n_train,
        n_val=n_val,
        resolution=resolution,
    )


def create_burgers_loader(
    *,
    n_samples: int = 1000,
    resolution: int = 128,
    batch_size: int = 32,
    viscosity_range: tuple[float, float] = (0.1, 0.1),
    time_final: float = 1.0,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PDELoaders:
    """Create train/val datarax pipelines for the 1D Burgers operator-learning task."""
    data = generate_burgers(
        n_samples=n_samples,
        resolution=resolution,
        viscosity_range=viscosity_range,
        time_final=time_final,
        seed=seed,
    )
    return _make_loaders(
        data, batch_size=batch_size, val_fraction=val_fraction, seed=seed, resolution=resolution
    )


def create_darcy_loader(
    *,
    n_samples: int = 1000,
    resolution: int = 64,
    batch_size: int = 32,
    coeff_range: tuple[float, float] = (0.1, 1.0),
    field_type: str = "smooth",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PDELoaders:
    """Create train/val datarax pipelines for the 2D Darcy-flow operator-learning task."""
    data = generate_darcy(
        n_samples=n_samples,
        resolution=resolution,
        coeff_range=coeff_range,
        field_type=field_type,
        seed=seed,
    )
    return _make_loaders(
        data, batch_size=batch_size, val_fraction=val_fraction, seed=seed, resolution=resolution
    )


def create_diffusion_loader(
    *,
    n_samples: int = 1000,
    resolution: int = 64,
    batch_size: int = 32,
    diffusion_range: tuple[float, float] = (0.01, 0.1),
    advection_range: tuple[float, float] = (-1.0, 1.0),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PDELoaders:
    """Create train/val datarax pipelines for the 2D diffusion-advection task."""
    data = generate_diffusion(
        n_samples=n_samples,
        resolution=resolution,
        diffusion_range=diffusion_range,
        advection_range=advection_range,
        seed=seed,
    )
    return _make_loaders(
        data, batch_size=batch_size, val_fraction=val_fraction, seed=seed, resolution=resolution
    )


def create_navier_stokes_loader(
    *,
    n_samples: int = 1000,
    resolution: int = 64,
    batch_size: int = 32,
    viscosity_range: tuple[float, float] = (0.001, 0.01),
    time_range: tuple[float, float] = (0.0, 1.0),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PDELoaders:
    """Create train/val datarax pipelines for the 2D Navier-Stokes task."""
    data = generate_navier_stokes(
        n_samples=n_samples,
        resolution=resolution,
        viscosity_range=viscosity_range,
        time_range=time_range,
        seed=seed,
    )
    return _make_loaders(
        data, batch_size=batch_size, val_fraction=val_fraction, seed=seed, resolution=resolution
    )


def create_shallow_water_loader(
    *,
    n_samples: int = 1000,
    resolution: int = 64,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> PDELoaders:
    """Create train/val datarax pipelines for the 2D shallow-water task."""
    data = generate_shallow_water(n_samples=n_samples, resolution=resolution, seed=seed)
    return _make_loaders(
        data, batch_size=batch_size, val_fraction=val_fraction, seed=seed, resolution=resolution
    )


__all__ = [
    "PDELoaders",
    "create_burgers_loader",
    "create_darcy_loader",
    "create_diffusion_loader",
    "create_navier_stokes_loader",
    "create_shallow_water_loader",
]
