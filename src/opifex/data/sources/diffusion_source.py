"""
Grain-compliant data source for diffusion-advection equation.

This module provides a RandomAccessDataSource implementation for generating
diffusion-advection solutions on-demand, following Grain's interface requirements.
"""

from typing import Any, SupportsIndex

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.solvers.diffusion_advection import solve_diffusion_advection_2d


class DiffusionDataSource(grain.RandomAccessDataSource):
    """
    Grain-compliant data source for diffusion-advection equation.

    Generates PDE solutions on-demand using existing physics module solver (DRY!).

    Args:
        n_samples: Total number of samples
        resolution: Spatial resolution
        time_steps: Number of time steps in trajectory
        diffusion_range: Range of diffusion coefficients
        advection_range: Range of advection velocities
        dimension: Either "1d" or "2d"
        seed: Random seed
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 64,
        time_steps: int = 5,
        diffusion_range: tuple[float, float] = (0.01, 0.1),
        advection_range: tuple[float, float] = (0.1, 1.0),
        dimension: str = "2d",
        seed: int = 42,
    ):
        """Initialize Diffusion data source."""
        if dimension not in ["1d", "2d"]:
            raise ValueError(f"dimension must be '1d' or '2d', got '{dimension}'")

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        self.n_samples = n_samples
        self.resolution = resolution
        self.time_steps = time_steps
        self.diffusion_range = diffusion_range
        self.advection_range = advection_range
        self.dimension = dimension
        self.seed = seed

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def _generate_initial_condition_2d(self, key):
        """Generate 2D initial condition."""
        x = jnp.linspace(0, 1, self.resolution)
        y = jnp.linspace(0, 1, self.resolution)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        key1, key2, key3 = jax.random.split(key, 3)
        ic_type = jax.random.randint(key1, (), 0, 3)

        def gaussian():
            cx = 0.3 + 0.4 * jax.random.uniform(key2)
            cy = 0.3 + 0.4 * jax.random.uniform(key3)
            width = 0.05 + 0.1 * jax.random.uniform(key2)
            return jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width**2))

        def square():
            cx = 0.3 + 0.4 * jax.random.uniform(key2)
            cy = 0.3 + 0.4 * jax.random.uniform(key3)
            size = 0.1 + 0.2 * jax.random.uniform(key2)
            return jnp.where(
                (jnp.abs(X - cx) < size / 2) & (jnp.abs(Y - cy) < size / 2),
                1.0,
                0.0,
            )

        def wave():
            fx = 2 + 4 * jax.random.uniform(key2)
            fy = 2 + 4 * jax.random.uniform(key3)
            return jnp.sin(fx * jnp.pi * X) * jnp.sin(fy * jnp.pi * Y)

        return jax.lax.switch(ic_type, [gaussian, square, wave])

    def __getitem__(self, index: SupportsIndex | slice) -> dict[str, Any]:
        """Generate sample deterministically from index."""
        if isinstance(index, slice):
            raise TypeError("Slicing not supported, use integer index")

        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")

        if index < 0 or index >= self.n_samples:
            raise IndexError(
                f"Index {index} out of bounds for source with {self.n_samples} samples"
            )

        # Deterministic key from index
        key = jax.random.PRNGKey(self.seed + index)
        key_ic, key_diff, key_vx, key_vy = jax.random.split(key, 4)

        # Always generate 2D initial condition (for existing solver compatibility)
        ic_2d = self._generate_initial_condition_2d(key_ic)

        # Generate diffusion coefficient
        diffusion_coeff = jax.random.uniform(
            key_diff,
            minval=self.diffusion_range[0],
            maxval=self.diffusion_range[1],
        )

        # Generate advection velocities
        vx = jax.random.uniform(
            key_vx, minval=-self.advection_range[1], maxval=self.advection_range[1]
        )
        vy = jax.random.uniform(
            key_vy, minval=-self.advection_range[1], maxval=self.advection_range[1]
        )

        # Solve using physics module solver (DRY!) - always in 2D
        trajectory_2d = []
        dt = 0.01
        steps_per_snapshot = max(1, 100 // self.time_steps)

        current_state = ic_2d

        for i in range(self.time_steps):
            trajectory_2d.append(current_state)
            if i < self.time_steps - 1:
                current_state = solve_diffusion_advection_2d(
                    current_state,
                    float(diffusion_coeff),
                    (float(vx), float(vy)),
                    dt=dt,
                    n_steps=steps_per_snapshot,
                    grid_spacing=1.0 / self.resolution,
                )

        # Stack trajectory
        output_2d = jnp.stack(trajectory_2d)

        # Extract 1D if needed (take middle row)
        if self.dimension == "1d":
            ic = ic_2d[self.resolution // 2, :]
            output = output_2d[:, self.resolution // 2, :]
        else:
            ic = ic_2d
            output = output_2d

        # Convert to NumPy for Grain
        return {
            "input": np.array(ic),
            "output": np.array(output),
            "metadata": {
                "index": index,
                "dimension": self.dimension,
                "resolution": self.resolution,
                "pde_type": "diffusion_advection",
            },
        }


__all__ = ["DiffusionDataSource"]
