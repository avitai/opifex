"""Finite Basis Physics-Informed Neural Network (FBPINN).

FBPINN uses smooth window functions to create a partition of unity,
enabling smooth blending of subdomain solutions without explicit
interface conditions.

Key Features:
    - Smooth window functions (cosine, Gaussian)
    - Partition of unity through normalization
    - No explicit interface conditions needed
    - Naturally handles overlapping subdomains

References:
    - Moseley et al. (2023): Finite Basis Physics-Informed Neural Networks
    - Survey Section 8.3.2: FBPINNs
    - GitHub: https://github.com/benmoseley/FBPINNs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import (
    DomainDecompositionPINN,
    Subdomain,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


class WindowFunction(ABC):
    """Abstract base class for window functions.

    Window functions define the influence region of each subdomain network.
    They should be smooth, have compact support within the subdomain, and
    enable partition of unity when combined.
    """

    def __init__(self, subdomain: Subdomain):
        """Initialize window function.

        Args:
            subdomain: The subdomain this window is associated with
        """
        self.subdomain = subdomain
        self.center = subdomain.center
        self.bounds = subdomain.bounds

    @abstractmethod
    def __call__(self, x: Float[Array, ...]) -> Float[Array, " batch"]:
        """Evaluate window function at given points.

        Args:
            x: Input coordinates, shape (batch, dim)

        Returns:
            Window values, shape (batch,)
        """


class CosineWindow(WindowFunction):
    """Cosine-based window function.

    w(x) = 0.5 * (1 + cos(π * r)) for r < 1, else 0

    where r is the normalized distance from the subdomain center,
    scaled by the subdomain half-width.

    This creates a smooth bump function that is 1 at the center
    and 0 at the boundary.
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, " batch"]:
        """Evaluate cosine window at given points."""
        # Compute normalized distance from center (0 at center, 1 at boundary)
        # For each dimension, compute distance normalized by half-width
        half_widths = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        normalized_dist = jnp.abs(x - self.center) / half_widths

        # Take maximum normalized distance across dimensions
        # This creates a "box" influence region
        r = jnp.max(normalized_dist, axis=-1)

        # Cosine window: 0.5 * (1 + cos(π * r)) for r < 1
        # At r=0 (center): 0.5 * (1 + 1) = 1
        # At r=1 (boundary): 0.5 * (1 + cos(π)) = 0.5 * (1 - 1) = 0
        return jnp.where(
            r < 1.0,
            0.5 * (1.0 + jnp.cos(jnp.pi * r)),
            jnp.zeros_like(r),
        )


class GaussianWindow(WindowFunction):
    """Gaussian-based window function.

    w(x) = exp(-||x - center||² / (2 * σ²))

    where σ controls the width of the Gaussian.
    """

    def __init__(self, subdomain: Subdomain, sigma: float = 0.25):
        """Initialize Gaussian window.

        Args:
            subdomain: The subdomain this window is associated with
            sigma: Standard deviation of the Gaussian (relative to subdomain size)
        """
        super().__init__(subdomain)
        # Scale sigma by the average subdomain half-width
        half_widths = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        self.sigma = sigma * jnp.mean(half_widths)

    def __call__(self, x: Float[Array, ...]) -> Float[Array, " batch"]:
        """Evaluate Gaussian window at given points."""
        # Compute squared distance from center
        diff = x - self.center
        dist_sq = jnp.sum(diff**2, axis=-1)

        # Gaussian window
        return jnp.exp(-dist_sq / (2.0 * self.sigma**2))


@dataclass(frozen=True)
class FBPINNConfig:
    """Configuration for FBPINN training.

    Attributes:
        window_type: Type of window function ("cosine" or "gaussian")
        normalize_windows: Whether to normalize window weights to sum to 1
        overlap_factor: Factor controlling subdomain overlap (for auto-partitioning)
        gaussian_sigma: Sigma parameter for Gaussian windows
    """

    window_type: Literal["cosine", "gaussian"] = "cosine"
    normalize_windows: bool = True
    overlap_factor: float = 0.2
    gaussian_sigma: float = 0.25


def create_window_function(
    subdomain: Subdomain,
    window_type: str = "cosine",
    **kwargs,
) -> WindowFunction:
    """Factory function to create window functions.

    Args:
        subdomain: Subdomain for the window
        window_type: Type of window ("cosine" or "gaussian")
        **kwargs: Additional arguments for specific window types

    Returns:
        WindowFunction instance
    """
    if window_type == "cosine":
        return CosineWindow(subdomain)
    if window_type == "gaussian":
        sigma = kwargs.get("sigma", kwargs.get("gaussian_sigma", 0.25))
        return GaussianWindow(subdomain, sigma=sigma)
    raise ValueError(f"Unknown window type: {window_type}")


class FBPINN(DomainDecompositionPINN):
    """Finite Basis Physics-Informed Neural Network.

    FBPINN decomposes the computational domain into overlapping subdomains,
    using smooth window functions to blend subdomain network outputs.
    This creates a partition of unity that ensures smooth global solutions.

    The output is computed as:
        u(x) = Σᵢ wᵢ(x) * uᵢ(x) / Σⱼ wⱼ(x)

    where wᵢ(x) is the window function for subdomain i and uᵢ(x) is the
    network output for subdomain i.

    Attributes:
        config: FBPINN configuration
        windows: List of window functions for each subdomain

    Example:
        >>> subdomains = [
        ...     Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),
        ...     Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
        ... ]
        >>> model = FBPINN(
        ...     input_dim=1, output_dim=1,
        ...     subdomains=subdomains, interfaces=[],
        ...     hidden_dims=[32, 32], rngs=nnx.Rngs(0)
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subdomains: Sequence[Subdomain],
        interfaces: Sequence,
        hidden_dims: Sequence[int],
        *,
        config: FBPINNConfig | None = None,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize FBPINN.

        Args:
            input_dim: Spatial dimension
            output_dim: Solution dimension
            subdomains: List of subdomain definitions (should overlap)
            interfaces: List of interface definitions (optional for FBPINN)
            hidden_dims: Hidden layer dimensions for subdomain networks
            config: FBPINN configuration. Uses defaults if None.
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=hidden_dims,
            activation=activation,
            rngs=rngs,
        )
        self.config = config or FBPINNConfig()

        # Create window function for each subdomain
        self.windows = [
            create_window_function(
                subdomain,
                window_type=self.config.window_type,
                gaussian_sigma=self.config.gaussian_sigma,
            )
            for subdomain in subdomains
        ]

    def compute_window_weights(
        self, x: Float[Array, ...]
    ) -> Float[Array, "batch num_subdomains"]:
        """Compute window weights for all subdomains.

        Args:
            x: Input coordinates

        Returns:
            Window weights, shape (batch, num_subdomains)
        """
        # Evaluate all window functions
        weights = jnp.stack(
            [window(x) for window in self.windows],
            axis=-1,
        )  # (batch, num_subdomains)

        # Normalize to partition of unity if configured
        if self.config.normalize_windows:
            weights_sum = jnp.sum(weights, axis=-1, keepdims=True)
            # Avoid division by zero
            weights = jnp.where(
                weights_sum > 1e-8,
                weights / weights_sum,
                weights,
            )

        return weights

    def _compute_subdomain_weights(
        self, x: Float[Array, ...]
    ) -> Float[Array, "batch num_subdomains"]:
        """Override base class method to use window functions.

        This is called by the parent class __call__ method.
        """
        return self.compute_window_weights(x)

    def __call__(self, x: Float[Array, ...]) -> Float[Array, "batch out"]:
        """Compute solution at given points using window-weighted blending.

        For each point, computes the weighted sum of all subdomain network
        outputs, where weights come from the window functions.

        Args:
            x: Input coordinates, shape (batch, input_dim)

        Returns:
            Solution values, shape (batch, output_dim)
        """
        # Get outputs from all subdomain networks
        subdomain_outputs = self.get_subdomain_outputs(x)

        # Compute window weights
        weights = self.compute_window_weights(x)

        # Weighted combination
        # subdomain_outputs is list of (batch, out) arrays
        stacked = jnp.stack(subdomain_outputs, axis=-1)  # (batch, out, num_subdomains)
        weights_expanded = weights[:, None, :]  # (batch, 1, num_subdomains)

        return jnp.sum(stacked * weights_expanded, axis=-1)
