"""Gaussian (z-score) normalization shared by the neural-operator examples.

A single tested helper replacing the per-field ``(x - mean) / std`` /
``pred * std + mean`` blocks copy-pasted across the operator examples. Fit the
statistics once on a training field and reuse them for the test field; map model
predictions back to physical units with :meth:`GaussianNormalizer.denormalize`.
Input and output fields use separate normalizers because they have different
scales.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


_STD_EPS: float = 1e-8


@struct.dataclass
class GaussianNormalizer:
    """Immutable scalar mean/std for z-score normalization.

    Attributes:
        mean: Field mean used for centering.
        std: Field standard deviation (already regularised) used for scaling.
    """

    mean: jax.Array
    std: jax.Array

    @classmethod
    def fit(cls, data: jax.Array) -> GaussianNormalizer:
        """Compute global mean and (epsilon-guarded) std over ``data``.

        Args:
            data: Field to fit statistics on, any shape.

        Returns:
            A normalizer carrying the fitted statistics.
        """
        return cls(mean=jnp.mean(data), std=jnp.std(data) + _STD_EPS)

    def normalize(self, x: jax.Array) -> jax.Array:
        """Return ``(x - mean) / std``."""
        return (x - self.mean) / self.std

    def denormalize(self, x: jax.Array) -> jax.Array:
        """Invert :meth:`normalize`: ``x * std + mean``."""
        return x * self.std + self.mean
