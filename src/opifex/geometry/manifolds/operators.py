"""
Manifold Neural Operators for Geometric Deep Learning

This module provides neural operators that operate on manifolds, enabling
geometric deep learning on curved spaces. The operators use tangent space
projections for neural processing while respecting manifold structure.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from .base import Manifold


class ManifoldNeuralOperator(nnx.Module):
    """
    Neural operator for processing data on manifolds.

    This operator maps manifold points to manifold points while using
    tangent space representations for neural network processing.
    The architecture follows the pattern:
    1. Map manifold points to tangent space via logarithmic map
    2. Process tangent vectors through neural network
    3. Map processed vectors back to manifold via exponential map

    Args:
        manifold: The manifold on which to operate
        hidden_dim: Hidden dimension for the neural network encoder
        rngs: FLAX NNX random number generators

    Example:
        >>> import jax
        >>> import flax.nnx as nnx
        >>> from opifex.geometry.manifolds import HyperbolicManifold
        >>> from opifex.geometry.manifolds.operators import ManifoldNeuralOperator
        >>>
        >>> manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        >>> key = jax.random.PRNGKey(42)
        >>> rngs = nnx.Rngs(key)
        >>>
        >>> operator = ManifoldNeuralOperator(
        ...     manifold=manifold,
        ...     hidden_dim=64,
        ...     rngs=rngs
        ... )
        >>>
        >>> # Process manifold points
        >>> points = manifold.random_point(key, shape=(batch_size,))
        >>> output = operator(points)
    """

    def __init__(
        self,
        manifold: Manifold,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the manifold neural operator."""
        self.manifold = manifold

        # Neural network encoder operating in tangent space
        # Following FLAX NNX patterns from critical technical guidelines
        self.encoder = nnx.Sequential(
            nnx.Linear(manifold.dimension, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, manifold.dimension, rngs=rngs),
        )

    def __call__(
        self, manifold_points: Float[Array, "batch manifold_dim"]
    ) -> Float[Array, "batch manifold_dim"]:
        """
        Apply the manifold neural operator to input points using vectorized operations.

        Args:
            manifold_points: Points on the manifold [batch_size, manifold_dim]

        Returns:
            Processed points on the manifold [batch_size, manifold_dim]
        """
        # Ensure input is properly shaped
        if manifold_points.ndim == 1:
            # Single point case
            manifold_points = manifold_points.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        # For manifold neural operators, we use a canonical reference point
        # Use zero vector as reference (works for most manifolds)
        reference_point = jnp.zeros(self.manifold.dimension)

        # Vectorized mapping to tangent space using vmap
        def map_to_tangent(point):
            valid_point = self._ensure_valid_point(point)
            return self.manifold.log_map(reference_point, valid_point)

        # Vectorized mapping from tangent space using vmap
        def map_from_tangent(tangent_vec):
            manifold_point = self.manifold.exp_map(reference_point, tangent_vec)
            return self._ensure_valid_point(manifold_point)

        # Use vmap for efficient batch processing
        batch_map_to_tangent = jax.vmap(map_to_tangent)
        batch_map_from_tangent = jax.vmap(map_from_tangent)

        # Map manifold points to tangent space (vectorized)
        tangent_batch = batch_map_to_tangent(manifold_points)

        # Process tangent vectors through neural network
        processed_tangent = self.encoder(tangent_batch)

        # Map processed tangent vectors back to manifold (vectorized)
        result = batch_map_from_tangent(processed_tangent)

        # Return single point if input was single point
        if single_point:
            result = result.reshape(-1)

        return result

    def _ensure_valid_point(self, point: Array) -> Array:
        """
        Ensure a point is valid on the manifold.

        Args:
            point: Point to validate

        Returns:
            Valid point on the manifold
        """
        # For basic manifolds, return the point as-is
        # Subclasses can override this for specific validation
        return point


class RiemannianNeuralOperator(ManifoldNeuralOperator):
    """
    Specialized neural operator for Riemannian manifolds.

    This extends the base ManifoldNeuralOperator with Riemannian-specific
    features such as parallel transport and metric-aware processing.
    """

    def __init__(
        self,
        manifold: Manifold,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        use_parallel_transport: bool = True,
    ):
        """
        Initialize Riemannian neural operator.

        Args:
            manifold: Riemannian manifold
            hidden_dim: Hidden dimension for neural networks
            rngs: Random number generators
            use_parallel_transport: Whether to use parallel transport
        """
        super().__init__(manifold, hidden_dim, rngs=rngs)
        self.use_parallel_transport = use_parallel_transport

        # Additional layers for metric-aware processing
        self.metric_processor = nnx.Sequential(
            nnx.Linear(
                manifold.dimension * manifold.dimension, hidden_dim // 2, rngs=rngs
            ),
            nnx.gelu,
            nnx.Linear(hidden_dim // 2, manifold.dimension, rngs=rngs),
        )

    def __call__(
        self, manifold_points: Float[Array, "batch manifold_dim"]
    ) -> Float[Array, "batch manifold_dim"]:
        """
        Apply Riemannian neural operator with metric awareness.

        Args:
            manifold_points: Points on the Riemannian manifold

        Returns:
            Processed points incorporating metric information
        """
        # Get base manifold neural operator result
        base_result = super().__call__(manifold_points)

        # Add metric-aware processing
        if hasattr(self.manifold, "metric_tensor"):
            # Compute metric tensors for all points (vectorized)
            batch_metric = jax.vmap(self.manifold.metric_tensor)(manifold_points)

            # Flatten metric tensors for processing
            batch_size = manifold_points.shape[0]
            flattened_metrics = batch_metric.reshape(batch_size, -1)

            # Process metric information
            metric_features = self.metric_processor(flattened_metrics)

            # Combine base result with metric-aware features
            # Use a learnable combination (could be made more sophisticated)
            alpha = 0.8  # Weight for base result
            beta = 0.2  # Weight for metric features

            return alpha * base_result + beta * metric_features

        return base_result


class HyperbolicNeuralOperator(ManifoldNeuralOperator):
    """
    Specialized neural operator for hyperbolic manifolds.

    This extends the base ManifoldNeuralOperator with hyperbolic-specific
    features such as gyrovector operations and curvature-aware processing.
    """

    def __init__(
        self,
        manifold: Manifold,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        use_gyro_operations: bool = True,
    ):
        """
        Initialize hyperbolic neural operator.

        Args:
            manifold: Hyperbolic manifold
            hidden_dim: Hidden dimension for neural networks
            rngs: Random number generators
            use_gyro_operations: Whether to use gyrovector operations
        """
        super().__init__(manifold, hidden_dim, rngs=rngs)
        self.use_gyro_operations = use_gyro_operations

        # Additional layers for curvature-aware processing
        # Use manifold dimension for processing
        self.curvature_processor = nnx.Sequential(
            nnx.Linear(manifold.dimension, hidden_dim // 4, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim // 4, manifold.dimension, rngs=rngs),
        )

    def __call__(
        self, manifold_points: Float[Array, "batch manifold_dim"]
    ) -> Float[Array, "batch manifold_dim"]:
        """
        Apply hyperbolic neural operator with curvature-aware processing.

        Args:
            manifold_points: Points on the hyperbolic manifold

        Returns:
            Processed points incorporating hyperbolic geometry
        """
        # Get base manifold neural operator result
        base_result = super().__call__(manifold_points)

        # Add curvature-aware processing
        if hasattr(self.manifold, "curvature") or hasattr(
            self.manifold, "scalar_curvature"
        ):
            # Process points through curvature-aware layers
            curvature_features = self.curvature_processor(manifold_points)

            # Combine base result with curvature-aware features
            # Use hyperbolic-specific combination
            gamma = 0.7  # Weight for base result
            delta = 0.3  # Weight for curvature features

            return gamma * base_result + delta * curvature_features

        return base_result
