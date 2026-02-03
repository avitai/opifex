"""
Grain-compliant data source for 2D Navier-Stokes equations.

This module provides a RandomAccessDataSource implementation for generating
2D incompressible Navier-Stokes solutions on-demand, following Grain's
interface requirements.
"""

from typing import Any, SupportsIndex

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.solvers.navier_stokes import (
    create_double_shear_layer,
    create_taylor_green_vortex,
    solve_navier_stokes_2d,
)


class NavierStokesDataSource(grain.RandomAccessDataSource):
    """
    Grain-compliant data source for 2D incompressible Navier-Stokes equations.

    Generates PDE solutions on-demand (lazy evaluation) while being deterministic:
    same index always returns the same sample.

    The data source generates various flow configurations including Taylor-Green
    vortices, double shear layers, and random vortical flows.

    Args:
        n_samples: Total number of samples in the dataset
        resolution: Spatial resolution for discretization
        time_steps: Number of time steps in solution trajectory
        reynolds_range: Tuple of (min_reynolds, max_reynolds) for Reynolds number
        time_range: Tuple of (start_time, end_time)
        seed: Random seed for deterministic generation

    Example:
        >>> source = NavierStokesDataSource(n_samples=100, resolution=64, seed=42)
        >>> sample = source[0]  # Returns dict with 'input', 'output', etc.
        >>> len(source)  # Returns 100
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 64,
        time_steps: int = 5,
        reynolds_range: tuple[float, float] = (100.0, 1000.0),
        time_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
    ):
        """Initialize Navier-Stokes data source."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        self.n_samples = n_samples
        self.resolution = resolution
        self.time_steps = time_steps
        self.reynolds_range = reynolds_range
        self.time_range = time_range
        self.seed = seed

        # Characteristic velocity and length scale for Reynolds number
        self.U_ref = 1.0  # Reference velocity
        self.L_ref = 2 * jnp.pi  # Domain size

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def _generate_initial_condition(
        self, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Generate random initial condition for NS equations."""
        key1, key2, key3 = jax.random.split(key, 3)

        # Choose initial condition type
        ic_type = int(jax.random.randint(key1, (), 0, 3))

        if ic_type == 0:
            # Taylor-Green vortex with random amplitude
            amplitude = float(jax.random.uniform(key2, (), minval=0.5, maxval=2.0))
            u0, v0 = create_taylor_green_vortex(self.resolution, amplitude)
        elif ic_type == 1:
            # Double shear layer with random perturbation
            perturbation = float(jax.random.uniform(key2, (), minval=0.01, maxval=0.1))
            thickness = float(jax.random.uniform(key3, (), minval=0.02, maxval=0.1))
            u0, v0 = create_double_shear_layer(self.resolution, thickness, perturbation)
        else:
            # Random vortical flow (superposition of vortices)
            x = jnp.linspace(0, 2 * jnp.pi, self.resolution, endpoint=False)
            y = jnp.linspace(0, 2 * jnp.pi, self.resolution, endpoint=False)
            X, Y = jnp.meshgrid(x, y, indexing="ij")

            # Random parameters for vortices
            n_vortices = 2
            u0 = jnp.zeros((self.resolution, self.resolution))
            v0 = jnp.zeros((self.resolution, self.resolution))

            for i in range(n_vortices):
                subkey = jax.random.fold_in(key2, i)
                subkeys = jax.random.split(subkey, 4)

                cx = float(jax.random.uniform(subkeys[0], (), minval=1, maxval=5))
                cy = float(jax.random.uniform(subkeys[1], (), minval=1, maxval=5))
                strength = float(
                    jax.random.uniform(subkeys[2], (), minval=-1.0, maxval=1.0)
                )
                width = float(
                    jax.random.uniform(subkeys[3], (), minval=0.3, maxval=1.0)
                )

                r_sq = (X - cx) ** 2 + (Y - cy) ** 2
                decay = jnp.exp(-r_sq / (2 * width**2))

                # Vortex velocity field
                u0 = u0 - strength * (Y - cy) * decay
                v0 = v0 + strength * (X - cx) * decay

        return u0, v0

    def __getitem__(self, index: SupportsIndex | slice) -> dict[str, Any]:
        """
        Generate sample deterministically from index.

        Args:
            index: Index of the sample to generate

        Returns:
            Dictionary containing:
                - input: Initial velocity field (2, resolution, resolution)
                - output: Solution trajectory (time_steps, 2, resolution, resolution)
                - reynolds: Reynolds number used
                - time_points: Time points of the solution
                - metadata: Additional information

        Raises:
            IndexError: If index is out of bounds
            TypeError: If index is not an integer
        """
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
        key_ic, key_re = jax.random.split(key)

        # Generate initial condition
        u0, v0 = self._generate_initial_condition(key_ic)

        # Generate Reynolds number and compute viscosity
        reynolds = float(
            jax.random.uniform(
                key_re,
                minval=self.reynolds_range[0],
                maxval=self.reynolds_range[1],
            )
        )
        # Re = U * L / nu => nu = U * L / Re
        nu = self.U_ref * self.L_ref / reynolds

        # Solve NS equations
        u_traj, v_traj = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=nu,
            time_range=self.time_range,
            time_steps=self.time_steps,
            resolution=self.resolution,
        )

        # Stack u and v into channels
        # Input: (2, resolution, resolution) - initial condition
        # Output: (time_steps, 2, resolution, resolution) - trajectory
        input_data = np.stack([np.array(u0), np.array(v0)], axis=0)

        # Skip initial condition in trajectory (it's already in input)
        output_u = np.array(u_traj[1:])  # (time_steps, res, res)
        output_v = np.array(v_traj[1:])  # (time_steps, res, res)
        output_data = np.stack(
            [output_u, output_v], axis=1
        )  # (time_steps, 2, res, res)

        return {
            "input": input_data,
            "output": output_data,
            "reynolds": reynolds,
            "viscosity": float(nu),
            "time_points": np.array(
                jnp.linspace(self.time_range[0], self.time_range[1], self.time_steps)
            ),
            "metadata": {
                "index": index,
                "resolution": self.resolution,
            },
        }


__all__ = ["NavierStokesDataSource"]
