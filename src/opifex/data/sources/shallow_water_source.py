"""
Grain-compliant data source for shallow water equations.

This module provides a RandomAccessDataSource implementation for generating
shallow water solutions on-demand, following Grain's interface requirements.
"""

from typing import Any, SupportsIndex

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.solvers.shallow_water import solve_shallow_water_2d


class ShallowWaterDataSource(grain.RandomAccessDataSource):
    """
    Grain-compliant data source for shallow water equations.

    Generates PDE solutions on-demand using existing physics module solver (DRY!).

    Args:
        n_samples: Total number of samples
        resolution: Spatial resolution
        time_steps: Number of time steps (not used - solver returns final state)
        seed: Random seed
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 64,
        time_steps: int = 1,  # Shallow water solver returns final state only
        seed: int = 42,
    ):
        """Initialize ShallowWater data source."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        self.n_samples = n_samples
        self.resolution = resolution
        self.time_steps = time_steps
        self.seed = seed

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def _generate_initial_condition(self, key):
        """Generate random initial conditions for (h, u, v)."""
        x = jnp.linspace(0, 1, self.resolution)
        y = jnp.linspace(0, 1, self.resolution)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        key_h, key_u, key_v = jax.random.split(key, 3)

        # Height field (small perturbation around mean depth)
        cx_h = jax.random.uniform(key_h, (), minval=0.3, maxval=0.7)
        cy_h = jax.random.uniform(key_h, (), minval=0.3, maxval=0.7)
        h = 1.0 + 0.1 * jnp.exp(-((X - cx_h) ** 2 + (Y - cy_h) ** 2) / 0.05)

        # U velocity field
        cx_u = jax.random.uniform(key_u, (), minval=0.3, maxval=0.7)
        u = 0.1 * jnp.sin(2 * jnp.pi * X) * jnp.exp(-((Y - cx_u) ** 2) / 0.1)

        # V velocity field
        cy_v = jax.random.uniform(key_v, (), minval=0.3, maxval=0.7)
        v = 0.1 * jnp.sin(2 * jnp.pi * Y) * jnp.exp(-((X - cy_v) ** 2) / 0.1)

        return h, u, v

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

        # Generate initial conditions
        h_init, u_init, v_init = self._generate_initial_condition(key)

        # Solve using physics module solver (DRY!)
        h_final, u_final, v_final = solve_shallow_water_2d(
            h_init,
            u_init,
            v_init,
            g=9.81,
            dt=0.001,
            n_steps=100,
            grid_spacing=float(1.0 / self.resolution),
        )

        # Stack initial and final states
        # Input: (h, u, v) at t=0
        input_state = jnp.stack([h_init, u_init, v_init])
        # Output: (h, u, v) at t=final
        output_state = jnp.stack([h_final, u_final, v_final])

        # Convert to NumPy for Grain
        return {
            "input": np.array(input_state),
            "output": np.array(output_state),
            "metadata": {
                "index": index,
                "resolution": self.resolution,
                "pde_type": "shallow_water",
            },
        }


__all__ = ["ShallowWaterDataSource"]
