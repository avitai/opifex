"""
Grain-compliant data source for Darcy flow equation.

This module provides a RandomAccessDataSource implementation for generating
Darcy flow solutions on-demand, following Grain's interface requirements.
"""

from typing import Any, SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np

from opifex.data.sources._base import GrainPDESource
from opifex.physics.solvers.darcy import solve_darcy_flow


class DarcyDataSource(GrainPDESource):
    """
    Grain-compliant data source for Darcy flow equation.

    Generates PDE solutions on-demand (lazy evaluation) while being deterministic:
    same index always returns the same sample.

    Solves: -∇·(a(x)∇u(x)) = f(x) where a(x) is the permeability field.

    Args:
        n_samples: Total number of samples in the dataset
        resolution: Spatial resolution for discretization
        viscosity_range: ``(low, high)`` permeability bounds. For
            ``field_type='smooth'`` the coefficient is scaled into this range; for
            ``field_type='binary'`` these are the two discrete values.
        field_type: ``'smooth'`` (default) for a smooth log-Fourier coefficient
            field, or ``'binary'`` for the high-contrast thresholded-GRF benchmark
            (Li et al. 2020) used by operator-learning showcases.
        seed: Random seed for deterministic generation

    Example:
        >>> source = DarcyDataSource(n_samples=1000, resolution=85, seed=42)
        >>> sample = source[0]  # Returns dict with 'input', 'output', etc.
        >>> len(source)  # Returns 1000
    """

    def __init__(
        self,
        n_samples: int = 1000,
        resolution: int = 85,
        viscosity_range: tuple[float, float] = (0.5, 2.0),
        field_type: str = "smooth",
        seed: int = 42,
    ) -> None:
        """Initialize Darcy data source."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        if field_type not in ("smooth", "binary"):
            raise ValueError(f"field_type must be 'smooth' or 'binary', got {field_type!r}")

        self.n_samples = n_samples
        self.resolution = resolution
        self.viscosity_range = viscosity_range
        self.field_type = field_type
        self.seed = seed

        # Store grid for solver
        x = jnp.linspace(0, 1, resolution)
        y = jnp.linspace(0, 1, resolution)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

        # Isotropic GRF power-spectrum filter for the binary benchmark field
        # (covariance ~ (-Δ + τ²I)^{-1}); DC zeroed so the threshold splits ~50/50.
        freqs = jnp.fft.fftfreq(resolution) * resolution
        kx, ky = jnp.meshgrid(freqs, freqs, indexing="ij")
        self._grf_filter = ((kx**2 + ky**2 + 9.0) ** (-1.0)).at[0, 0].set(0.0)

    def _generate_permeability_field(self, key):
        """Generate a permeability coefficient field.

        ``field_type='binary'`` thresholds a Gaussian random field into the two
        ``viscosity_range`` values (the Li et al. 2020 high-contrast benchmark);
        ``field_type='smooth'`` builds a smooth log-Fourier field scaled into the
        ``viscosity_range``.
        """
        if self.field_type == "binary":
            white = jax.random.normal(key, (self.resolution, self.resolution))
            field = jnp.fft.ifft2(jnp.fft.fft2(white) * self._grf_filter).real
            low, high = self.viscosity_range
            return jnp.where(field >= 0.0, high, low)

        key1, key2, key3 = jax.random.split(key, 3)

        # Random Fourier features for smooth coefficient field
        n_modes = 12
        freqs = jax.random.randint(key1, (n_modes, 2), 1, 6)
        phases = jax.random.uniform(key2, (n_modes,), minval=0, maxval=2 * jnp.pi)
        amplitudes = jax.random.uniform(key3, (n_modes,), minval=0.1, maxval=1.0)

        # Build coefficient field
        coeff_field = jnp.ones((self.resolution, self.resolution))

        for i in range(n_modes):
            coeff_field += amplitudes[i] * jnp.sin(
                freqs[i, 0] * jnp.pi * self.X + freqs[i, 1] * jnp.pi * self.Y + phases[i]
            )

        # Ensure positive definite and bounded within viscosity range
        coeff_field = jnp.exp(coeff_field)
        # Scale to viscosity range
        return self.viscosity_range[0] + (self.viscosity_range[1] - self.viscosity_range[0]) * (
            coeff_field - jnp.min(coeff_field)
        ) / (jnp.max(coeff_field) - jnp.min(coeff_field) + 1e-10)

    def __getitem__(self, index: SupportsIndex | slice) -> dict[str, Any]:
        """
        Generate sample deterministically from index.

        Args:
            index: Index of the sample to generate

        Returns:
            Dictionary containing:
                - input: Permeability coefficient field a(x)
                - output: Pressure solution u(x)
                - metadata: Additional information

        Raises:
            IndexError: If index is out of bounds
        """
        index, key = self._resolve_key(index)

        # Generate permeability field
        permeability = self._generate_permeability_field(key)

        # Solve Darcy flow using physics module solver (DRY!)
        # Jacobi method needs many iterations to converge - 500 is a reasonable balance
        # between accuracy and speed for training data generation
        solution = solve_darcy_flow(permeability, self.resolution, max_iter=500)

        # Convert to NumPy for Grain (Grain expects NumPy arrays for multiprocessing)
        return {
            "input": np.array(permeability),
            "output": np.array(solution),
            "metadata": {
                "index": index,
                "resolution": self.resolution,
                "pde_type": "darcy_flow",
            },
        }


__all__ = ["DarcyDataSource"]
