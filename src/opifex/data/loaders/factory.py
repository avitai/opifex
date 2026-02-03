"""
Grain DataLoader factory functions for Opifex PDEs.

This module provides factory functions to create configured Grain DataLoaders
with appropriate data sources, samplers, and transformations.
"""

from typing import Any

import grain.python as grain  # type: ignore  # noqa: PGH003

from opifex.data.sources.burgers_source import BurgersDataSource
from opifex.data.sources.darcy_source import DarcyDataSource
from opifex.data.sources.diffusion_source import DiffusionDataSource
from opifex.data.sources.navier_stokes_source import NavierStokesDataSource
from opifex.data.sources.shallow_water_source import ShallowWaterDataSource
from opifex.data.transforms.pde_transforms import (
    AddNoiseAugmentation,
    NormalizeTransform,
    SpectralTransform,
)


def create_burgers_loader(
    n_samples: int = 1000,
    batch_size: int = 32,
    resolution: int = 64,
    time_steps: int = 5,
    viscosity_range: tuple[float, float] = (0.01, 0.1),
    time_range: tuple[float, float] = (0.0, 2.0),
    dimension: str = "2d",
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 4,
    enable_normalization: bool = True,
    normalization_mean: float = 0.0,
    normalization_std: float = 1.0,
    enable_spectral: bool = False,
    enable_augmentation: bool = False,
    augmentation_noise_level: float = 0.01,
    num_epochs: int = 1,
    **kwargs: Any,
) -> grain.DataLoader:
    """
    Create Grain DataLoader for Burgers equation dataset.

    This factory creates a complete data loading pipeline with:
    1. BurgersDataSource - on-demand PDE solution generation
    2. IndexSampler - efficient shuffling and sharding
    3. Optional transforms - normalization, spectral features, augmentation
    4. Batching - efficient batch creation

    Args:
        n_samples: Total number of samples in dataset
        batch_size: Batch size for training
        resolution: Spatial resolution for discretization
        time_steps: Number of time steps in solution
        viscosity_range: Tuple of (min_viscosity, max_viscosity)
        time_range: Tuple of (start_time, end_time)
        dimension: Either "1d" or "2d"
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility
        worker_count: Number of parallel worker processes
        enable_normalization: Apply z-score normalization
        normalization_mean: Mean for normalization
        normalization_std: Std for normalization
        enable_spectral: Add FFT features
        enable_augmentation: Add noise augmentation
        augmentation_noise_level: Noise level for augmentation
        num_epochs: Number of epochs to iterate (default: 1)
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        grain.DataLoader: Configured data loader ready for iteration

    Example:
        >>> loader = create_burgers_loader(
        ...     n_samples=10000,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     worker_count=4,
        ... )
        >>> for batch in loader:
        ...     x_batch = batch["input"]
        ...     y_batch = batch["output"]
        ...     # Train model
    """
    # 1. Data source - generates PDE solutions on-demand
    data_source = BurgersDataSource(
        n_samples=n_samples,
        resolution=resolution,
        time_steps=time_steps,
        viscosity_range=viscosity_range,
        time_range=time_range,
        dimension=dimension,
        seed=seed,
    )

    # 2. Sampler - handles shuffling and multi-process sharding
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    )

    # 3. Operations pipeline - transforms and batching
    operations = []

    # Optional: Add noise augmentation (before normalization)
    if enable_augmentation:
        operations.append(AddNoiseAugmentation(noise_level=augmentation_noise_level))

    # Optional: Normalize data
    if enable_normalization:
        operations.append(
            NormalizeTransform(mean=normalization_mean, std=normalization_std)
        )

    # Optional: Add spectral features
    if enable_spectral:
        operations.append(SpectralTransform())

    # Always add batching as the last operation
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

    # 4. Create DataLoader
    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=20,  # Pre-fetch buffer size
    )


def create_darcy_loader(
    n_samples: int = 1000,
    batch_size: int = 32,
    resolution: int = 85,
    viscosity_range: tuple[float, float] = (0.5, 2.0),
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 0,
    enable_normalization: bool = True,
    num_epochs: int = 1,
    **kwargs: Any,
) -> grain.DataLoader:
    """Create Grain DataLoader for Darcy flow dataset."""
    data_source = DarcyDataSource(
        n_samples=n_samples,
        resolution=resolution,
        viscosity_range=viscosity_range,
        seed=seed,
    )

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    )

    operations = []
    if enable_normalization:
        operations.append(NormalizeTransform(mean=0.0, std=1.0))
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=20,
    )


def create_diffusion_loader(
    n_samples: int = 1000,
    batch_size: int = 32,
    resolution: int = 64,
    time_steps: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 0,
    num_epochs: int = 1,
    **kwargs: Any,
) -> grain.DataLoader:
    """Create Grain DataLoader for diffusion-advection dataset."""
    data_source = DiffusionDataSource(
        n_samples=n_samples,
        resolution=resolution,
        time_steps=time_steps,
        seed=seed,
    )

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    )

    operations = [grain.Batch(batch_size=batch_size, drop_remainder=True)]

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=20,
    )


def create_shallow_water_loader(
    n_samples: int = 1000,
    batch_size: int = 32,
    resolution: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 0,
    num_epochs: int = 1,
    **kwargs: Any,
) -> grain.DataLoader:
    """Create Grain DataLoader for shallow water equations dataset."""
    data_source = ShallowWaterDataSource(
        n_samples=n_samples,
        resolution=resolution,
        seed=seed,
    )

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    )

    operations = [grain.Batch(batch_size=batch_size, drop_remainder=True)]

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=20,
    )


def create_navier_stokes_loader(
    n_samples: int = 1000,
    batch_size: int = 32,
    resolution: int = 64,
    time_steps: int = 5,
    reynolds_range: tuple[float, float] = (100.0, 1000.0),
    time_range: tuple[float, float] = (0.0, 1.0),
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 0,
    enable_normalization: bool = False,
    normalization_mean: float = 0.0,
    normalization_std: float = 1.0,
    num_epochs: int = 1,
    **kwargs: Any,
) -> grain.DataLoader:
    """
    Create Grain DataLoader for 2D Navier-Stokes equations dataset.

    This factory creates a complete data loading pipeline for incompressible
    2D Navier-Stokes solutions with:
    1. NavierStokesDataSource - on-demand NS solution generation
    2. IndexSampler - efficient shuffling and sharding
    3. Optional transforms - normalization
    4. Batching - efficient batch creation

    Args:
        n_samples: Total number of samples in dataset
        batch_size: Batch size for training
        resolution: Spatial resolution for discretization
        time_steps: Number of time steps in solution trajectory
        reynolds_range: Tuple of (min_reynolds, max_reynolds)
        time_range: Tuple of (start_time, end_time)
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility
        worker_count: Number of parallel worker processes
        enable_normalization: Apply z-score normalization
        normalization_mean: Mean for normalization
        normalization_std: Std for normalization
        num_epochs: Number of epochs to iterate (default: 1)
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        grain.DataLoader: Configured data loader ready for iteration

    Example:
        >>> loader = create_navier_stokes_loader(
        ...     n_samples=1000,
        ...     batch_size=16,
        ...     resolution=64,
        ...     reynolds_range=(100, 500),
        ...     seed=42,
        ... )
        >>> for batch in loader:
        ...     input_vel = batch["input"]   # (batch, 2, res, res)
        ...     output_vel = batch["output"] # (batch, time_steps, 2, res, res)
    """
    data_source = NavierStokesDataSource(
        n_samples=n_samples,
        resolution=resolution,
        time_steps=time_steps,
        reynolds_range=reynolds_range,
        time_range=time_range,
        seed=seed,
    )

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    )

    operations = []
    if enable_normalization:
        operations.append(
            NormalizeTransform(mean=normalization_mean, std=normalization_std)
        )
    operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count,
        worker_buffer_size=20,
    )


__all__ = [
    "create_burgers_loader",
    "create_darcy_loader",
    "create_diffusion_loader",
    "create_navier_stokes_loader",
    "create_shallow_water_loader",
]
