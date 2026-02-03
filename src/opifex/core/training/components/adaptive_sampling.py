"""Residual-based Adaptive Distribution (RAD) sampling for PINNs.

This module implements adaptive collocation point sampling strategies
for physics-informed neural networks, based on PDE residual magnitudes.

Key Features:
    - RAD: Residual-weighted sampling distribution
    - RAR-D: Residual-based Adaptive Refinement with Distribution

The key insight is that collocation points should be concentrated in
regions where the PDE residual is high, as these are the areas where
the neural network needs more training signal.

References:
    - Survey Section 5.2: Residual-based Adaptive Sampling
    - Wu et al. (2023): A comprehensive study of non-adaptive and
      residual-based adaptive sampling for physics-informed neural networks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray


@dataclass(frozen=True)
class RADConfig:
    """Configuration for Residual-based Adaptive Distribution sampling.

    Attributes:
        beta: Exponent for residual weighting. Higher values concentrate
              sampling more strongly on high-residual regions.
              ξ_j = |r_j|^β / Σ_k |r_k|^β
        resample_frequency: Number of training steps between resampling
        min_probability: Minimum sampling probability to ensure coverage
        temperature: Temperature for probability smoothing
    """

    beta: float = 1.0
    resample_frequency: int = 100
    min_probability: float = 1e-6
    temperature: float = 1.0


def compute_sampling_distribution(
    residuals: Float[Array, ...],
    beta: float = 1.0,
    min_probability: float = 1e-6,
) -> Float[Array, ...]:
    """Compute sampling distribution from residual magnitudes.

    The sampling probability for each point is proportional to the
    residual magnitude raised to the power beta:
        p_j = |r_j|^β / Σ_k |r_k|^β

    Args:
        residuals: PDE residual magnitudes at each collocation point
        beta: Exponent for residual weighting
        min_probability: Minimum probability to ensure all points have
                        some chance of being sampled

    Returns:
        Sampling probabilities that sum to 1
    """
    # Take absolute value and add small epsilon for numerical stability
    abs_residuals = jnp.abs(residuals) + 1e-10

    # Apply power weighting
    weighted = jnp.power(abs_residuals, beta)

    # Normalize to get probabilities
    probabilities = weighted / jnp.sum(weighted)

    # Ensure minimum probability for all points
    probabilities = jnp.maximum(probabilities, min_probability)

    # Re-normalize after applying minimum
    return probabilities / jnp.sum(probabilities)


class RADSampler:
    """Residual-based Adaptive Distribution sampler.

    This sampler draws collocation points from the domain with
    probability proportional to the PDE residual magnitude.

    Attributes:
        config: RAD configuration

    Example:
        >>> sampler = RADSampler()
        >>> domain_points = jnp.linspace(0, 1, 100).reshape(-1, 1)
        >>> residuals = compute_pde_residual(model, domain_points)
        >>> key = jax.random.key(0)
        >>> sampled = sampler.sample(domain_points, residuals, batch_size=32, key=key)
    """

    def __init__(self, config: RADConfig | None = None):
        """Initialize RAD sampler.

        Args:
            config: RAD configuration. Uses defaults if None.
        """
        self.config = config or RADConfig()

    def sample(
        self,
        domain_points: Float[Array, ...],
        residuals: Float[Array, ...],
        batch_size: int,
        key: PRNGKeyArray,
    ) -> Float[Array, ...]:
        """Sample collocation points based on residual distribution.

        Args:
            domain_points: Candidate points in the domain
            residuals: PDE residual magnitudes at each point
            batch_size: Number of points to sample
            key: JAX random key

        Returns:
            Sampled collocation points
        """
        # Compute sampling probabilities
        probs = compute_sampling_distribution(
            residuals,
            beta=self.config.beta,
            min_probability=self.config.min_probability,
        )

        # Sample indices based on probabilities
        indices = jax.random.choice(
            key,
            len(domain_points),
            shape=(batch_size,),
            p=probs,
            replace=True,
        )

        return domain_points[indices]

    def compute_weights(
        self,
        residuals: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Compute importance weights for residual-weighted loss.

        These weights can be used to weight the loss function instead
        of resampling the collocation points.

        Args:
            residuals: PDE residual magnitudes

        Returns:
            Importance weights
        """
        return compute_sampling_distribution(
            residuals,
            beta=self.config.beta,
            min_probability=self.config.min_probability,
        )


@dataclass(frozen=True)
class RARDConfig:
    """Configuration for RAR-D refinement.

    Attributes:
        num_new_points: Number of new points to add per refinement
        percentile_threshold: Only refine near points above this percentile
        noise_scale: Scale of random perturbation for new points
    """

    num_new_points: int = 10
    percentile_threshold: float = 90.0
    noise_scale: float = 0.1


class RARDRefiner:
    """Residual-based Adaptive Refinement with Distribution.

    This refiner adds new collocation points near regions with high
    PDE residual, adaptively increasing resolution where needed.

    Attributes:
        config: RAR-D configuration

    Example:
        >>> refiner = RARDRefiner(num_new_points=20)
        >>> refined_points = refiner.refine(points, residuals, bounds, key)
    """

    def __init__(
        self,
        config: RARDConfig | None = None,
        num_new_points: int | None = None,
        noise_scale: float | None = None,
    ):
        """Initialize RAR-D refiner.

        Args:
            config: RAR-D configuration. Uses defaults if None.
            num_new_points: Override for number of new points
            noise_scale: Override for noise scale
        """
        if config is not None:
            self.config = config
        else:
            self.config = RARDConfig(
                num_new_points=num_new_points or 10,
                noise_scale=noise_scale or 0.1,
            )

    def refine(
        self,
        current_points: Float[Array, ...],
        residuals: Float[Array, ...],
        bounds: Float[Array, "dim 2"],
        key: PRNGKeyArray,
    ) -> Float[Array, ...]:
        """Add new points near high-residual regions.

        Args:
            current_points: Current collocation points
            residuals: PDE residual magnitudes at each point
            bounds: Domain bounds, shape (dim, 2) with [min, max]
            key: JAX random key

        Returns:
            Refined point set including new points
        """
        n_points, dim = current_points.shape
        num_new = self.config.num_new_points

        # Find high-residual points (above threshold percentile)
        threshold = jnp.percentile(jnp.abs(residuals), self.config.percentile_threshold)
        high_residual_mask = jnp.abs(residuals) >= threshold

        # Get high-residual point indices
        jnp.where(high_residual_mask, size=n_points)[0]
        n_high = jnp.sum(high_residual_mask)

        # If no high residual points, use all points
        n_high = jnp.maximum(n_high, 1)

        # Sample base points from high-residual regions
        key1, key2 = jax.random.split(key)

        # Use residual-weighted sampling for selecting base points
        probs = compute_sampling_distribution(residuals, beta=1.0)
        base_indices = jax.random.choice(
            key1,
            n_points,
            shape=(num_new,),
            p=probs,
            replace=True,
        )
        base_points = current_points[base_indices]

        # Add random perturbation
        domain_sizes = bounds[:, 1] - bounds[:, 0]
        noise = jax.random.normal(key2, shape=(num_new, dim))
        noise = noise * self.config.noise_scale * domain_sizes

        new_points = base_points + noise

        # Clip to bounds
        new_points = jnp.clip(
            new_points,
            bounds[:, 0],
            bounds[:, 1],
        )

        # Concatenate with existing points
        return jnp.concatenate([current_points, new_points], axis=0)

    def identify_refinement_regions(
        self,
        residuals: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Identify which points are in refinement regions.

        Args:
            residuals: PDE residual magnitudes

        Returns:
            Boolean mask indicating refinement regions
        """
        threshold = jnp.percentile(jnp.abs(residuals), self.config.percentile_threshold)
        return jnp.abs(residuals) >= threshold
