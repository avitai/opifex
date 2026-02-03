"""
Grain-compliant data source for Burgers equation.

This module provides a RandomAccessDataSource implementation for generating
Burgers equation solutions on-demand, following Grain's interface requirements.
"""

from typing import Any, SupportsIndex

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.solvers.burgers import solve_burgers_1d, solve_burgers_2d


class BurgersDataSource(grain.RandomAccessDataSource):
    """
    Grain-compliant data source for Burgers equation.

    Generates PDE solutions on-demand (lazy evaluation) while being deterministic:
    same index always returns the same sample.

    This implementation follows DRY principles by reusing the existing
    BurgersEquationDataset solver logic.

    Args:
        n_samples: Total number of samples in the dataset
        resolution: Spatial resolution for discretization
        time_steps: Number of time steps in solution trajectory
        viscosity_range: Tuple of (min_viscosity, max_viscosity)
        time_range: Tuple of (start_time, end_time)
        dimension: Either "1d" or "2d" for the problem dimension
        seed: Random seed for deterministic generation

    Example:
        >>> source = BurgersDataSource(n_samples=1000, resolution=64, seed=42)
        >>> sample = source[0]  # Returns dict with 'input', 'output', etc.
        >>> len(source)  # Returns 1000
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 64,
        time_steps: int = 5,
        viscosity_range: tuple[float, float] = (0.01, 0.1),
        time_range: tuple[float, float] = (0.0, 2.0),
        dimension: str = "2d",
        seed: int = 42,
    ):
        """Initialize Burgers data source."""
        if dimension not in ["1d", "2d"]:
            raise ValueError(f"dimension must be '1d' or '2d', got '{dimension}'")

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        self.n_samples = n_samples
        self.resolution = resolution
        self.time_steps = time_steps
        self.viscosity_range = viscosity_range
        self.time_range = time_range
        self.dimension = dimension
        self.seed = seed

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def _generate_initial_condition(self, key, dimension):
        """Generate random initial condition."""
        if dimension == "1d":
            x = jnp.linspace(-1, 1, self.resolution)
            key1, key2, key3 = jax.random.split(key, 3)
            ic_type = jax.random.randint(key1, (), 0, 3)

            def gaussian():
                center = jax.random.uniform(key2, (), minval=-0.5, maxval=0.5)
                width = jax.random.uniform(key3, (), minval=0.1, maxval=0.3)
                return jnp.exp(-((x - center) ** 2) / width**2)

            def sine():
                freq = jax.random.uniform(key2, (), minval=1.0, maxval=3.0)
                phase = jax.random.uniform(key3, (), minval=0.0, maxval=2 * jnp.pi)
                return jnp.sin(freq * jnp.pi * x + phase)

            def step():
                center = jax.random.uniform(key2, (), minval=-0.3, maxval=0.3)
                width = jax.random.uniform(key3, (), minval=0.05, maxval=0.15)
                return jnp.tanh((x - center) / width)

            return jax.lax.switch(ic_type, [gaussian, sine, step])
        # 2D
        x = jnp.linspace(-1, 1, self.resolution)
        y = jnp.linspace(-1, 1, self.resolution)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        key1, key2, key3, key4 = jax.random.split(key, 4)
        ic_type = jax.random.randint(key1, (), 0, 3)

        def gaussian():
            cx = jax.random.uniform(key2, (), minval=-0.5, maxval=0.5)
            cy = jax.random.uniform(key3, (), minval=-0.5, maxval=0.5)
            width = jax.random.uniform(key4, (), minval=0.2, maxval=0.4)
            return jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / width**2)

        def vortex():
            return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.exp(-(X**2 + Y**2))

        def wave():
            fx = jax.random.uniform(key2, (), minval=1.0, maxval=2.0)
            fy = jax.random.uniform(key3, (), minval=1.0, maxval=2.0)
            phase = jax.random.uniform(key4, (), minval=0.0, maxval=2 * jnp.pi)
            return jnp.sin(fx * jnp.pi * X) * jnp.cos(fy * jnp.pi * Y + phase)

        return jax.lax.switch(ic_type, [gaussian, vortex, wave])

    def __getitem__(self, index: SupportsIndex | slice) -> dict[str, Any]:
        """
        Generate sample deterministically from index.

        Args:
            index: Index of the sample to generate

        Returns:
            Dictionary containing:
                - input: Initial condition
                - output: Solution trajectory
                - viscosity: Viscosity parameter used
                - time_points: Time points of the solution
                - metadata: Additional information

        Raises:
            IndexError: If index is out of bounds
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
        key_ic, key_visc = jax.random.split(key)

        # Generate initial condition
        ic = self._generate_initial_condition(key_ic, self.dimension)

        # Generate viscosity parameter
        viscosity = jax.random.uniform(
            key_visc,
            minval=self.viscosity_range[0],
            maxval=self.viscosity_range[1],
        )

        # Solve PDE using physics module solvers (DRY!)
        if self.dimension == "2d":
            solution = solve_burgers_2d(
                ic,
                viscosity,  # pyright: ignore[reportArgumentType]
                self.time_range,
                self.time_steps,
                self.resolution,
            )
        else:
            solution = solve_burgers_1d(
                ic,
                viscosity,  # pyright: ignore[reportArgumentType]
                self.time_range,
                self.time_steps,
                self.resolution,
            )

        # Solver returns time_steps+1 points (includes initial), skip first
        solution = solution[1:]

        # Convert to NumPy for Grain (Grain expects NumPy arrays for multiprocessing)
        # Return as dictionary (Grain convention)
        return {
            "input": np.array(ic),
            "output": np.array(solution),
            "viscosity": float(viscosity),
            "time_points": np.array(
                jnp.linspace(self.time_range[0], self.time_range[1], self.time_steps)
            ),
            "metadata": {
                "index": index,
                "dimension": self.dimension,
                "resolution": self.resolution,
            },
        }


__all__ = ["BurgersDataSource"]
