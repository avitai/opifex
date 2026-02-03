"""
Grain transforms for PDE data processing.

This module provides Grain-compliant MapTransform implementations
for normalizing, augmenting, and processing PDE data.
"""

import dataclasses
from typing import Any

import grain.python as grain
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class NormalizeTransform(grain.MapTransform):
    """
    Normalize PDE data using mean and standard deviation.

    Applies z-score normalization: (x - mean) / (std + epsilon)
    to both 'input' and 'output' fields in the sample dictionary.

    Args:
        mean: Mean value for normalization
        std: Standard deviation for normalization
        epsilon: Small constant to prevent division by zero

    Example:
        >>> transform = NormalizeTransform(mean=0.0, std=1.0)
        >>> sample = {"input": jnp.array([1.0, 2.0, 3.0])}
        >>> normalized = transform.map(sample)
    """

    mean: float = 0.0
    std: float = 1.0
    epsilon: float = 1e-8

    def map(self, features: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Apply normalization to input and output fields.

        Args:
            features: Dictionary containing PDE data

        Returns:
            Dictionary with normalized input and output fields
        """
        # Create copy to avoid modifying original
        result = features.copy()

        # Normalize input if present
        if "input" in features:
            result["input"] = (features["input"] - self.mean) / (
                self.std + self.epsilon
            )

        # Normalize output if present
        if "output" in features:
            result["output"] = (features["output"] - self.mean) / (
                self.std + self.epsilon
            )

        return result


@dataclasses.dataclass
class SpectralTransform(grain.MapTransform):
    """
    Add spectral (FFT) features to PDE data.

    Computes the real FFT of the input field and adds it as a new
    'input_fft' key, preserving the original input.

    Example:
        >>> transform = SpectralTransform()
        >>> sample = {"input": jnp.array([1.0, 2.0, 3.0, 4.0])}
        >>> with_fft = transform.map(sample)
        >>> "input_fft" in with_fft  # True
    """

    def map(self, features: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Compute FFT and add to features.

        Args:
            features: Dictionary containing PDE data

        Returns:
            Dictionary with additional 'input_fft' field
        """
        # Create copy to avoid modifying original
        result = features.copy()

        # Add FFT of input if present
        if "input" in features:
            # Use real FFT (rfft) since PDE data is real-valued
            # FFT along last axis for both 1D and 2D data
            result["input_fft"] = jnp.fft.rfft(features["input"], axis=-1)

        return result


@dataclasses.dataclass
class AddNoiseAugmentation(grain.MapTransform):
    """
    Add Gaussian noise to input for data augmentation.

    Adds random Gaussian noise to the 'input' field only, leaving
    'output' unchanged. Useful for training robust models.

    Args:
        noise_level: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility (default: 0)

    Example:
        >>> augment = AddNoiseAugmentation(noise_level=0.01)
        >>> sample = {"input": jnp.array([1.0, 2.0, 3.0])}
        >>> noisy = augment.map(sample)
    """

    noise_level: float = 0.01
    seed: int = 0

    def map(self, features: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Add Gaussian noise to input field.

        Args:
            features: Dictionary containing PDE data

        Returns:
            Dictionary with noisy input field
        """
        # Create copy to avoid modifying original
        result = features.copy()

        # Add noise to input if present
        if "input" in features:
            key = jax.random.PRNGKey(self.seed)
            noise = jax.random.normal(key, features["input"].shape) * self.noise_level
            result["input"] = features["input"] + noise

        return result


__all__ = ["AddNoiseAugmentation", "NormalizeTransform", "SpectralTransform"]
