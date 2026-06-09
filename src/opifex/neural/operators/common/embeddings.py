"""Grid Embeddings for Neural Operators - Opifex Framework.

JAX/Flax NNX implementation of grid embeddings for neural operators,
based on neuraloperator reference implementation.

This module provides:
- GridEmbeddingND: N-dimensional positional embedding (1D/2D/3D and beyond)
- GridEmbedding2D: Two-dimensional specialization of GridEmbeddingND
- Utility functions for regular grid generation

All implementations are optimized for JAX transformations and Flax NNX patterns.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx


class EmbeddingBase(nnx.Module, ABC):
    """Abstract base class for embeddings."""

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels after embedding."""


class GridEmbeddingND(EmbeddingBase):
    """Append one normalized coordinate channel per spatial axis (any N-D grid).

    For an input of shape ``(batch, d_1, ..., d_n, channels)`` this concatenates
    ``dim`` extra channels along the last axis, where appended channel ``k`` is
    the coordinate that varies along spatial axis ``k`` and spans
    ``grid_boundaries[k]`` from its lower to its upper bound (``indexing="ij"``).
    Works for 1D, 2D, and 3D (and beyond).

    Spectral convolutions are translation-equivariant, so injecting absolute
    grid coordinates is the standard way to give a neural operator position
    awareness (Li et al., 2020).

    Reference: ``neuralop.layers.embeddings.GridEmbeddingND`` from the
    `neuraloperator <https://github.com/neuraloperator/neuraloperator>`_ library,
    adapted to JAX/Flax NNX. The channels-last layout follows opifex's own
    ``fno._positional.append_grid_coordinates`` convention transposed to the
    framework's channels-last operator inputs. Unlike the PyTorch reference,
    which uses ``linspace(start, stop, res + 1)[:-1]`` (left-closed cells), this
    implementation includes both endpoints so the corner coordinates equal the
    configured boundaries exactly.

    Args:
        in_channels: Number of input channels.
        dim: Number of spatial dimensions.
        grid_boundaries: Per-axis ``[start, end]`` coordinate bounds; defaults
            to ``[[0.0, 1.0]] * dim``. Must have length ``dim``.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 2,
        grid_boundaries: list[list[float]] | None = None,
    ) -> None:
        super().__init__()
        if grid_boundaries is None:
            grid_boundaries = [[0.0, 1.0] for _ in range(dim)]
        if len(grid_boundaries) != dim:
            msg = f"Expected grid_boundaries to have length {dim}, got {len(grid_boundaries)}"
            raise ValueError(msg)
        self.in_channels = in_channels
        self.dim = dim
        self.grid_boundaries = grid_boundaries

    @property
    def out_channels(self) -> int:
        """Number of output channels after embedding (input + dim coordinate channels)."""
        return self.in_channels + self.dim

    def _generate_grid(self, spatial_shape: tuple[int, ...]) -> list[jax.Array]:
        """Generate the per-axis N-D coordinate meshgrid.

        Args:
            spatial_shape: Spatial dimensions ``(d_1, ..., d_n)``.

        Returns:
            One coordinate array per spatial axis, each of shape ``spatial_shape``.
        """
        if len(spatial_shape) != self.dim:
            msg = f"Expected {self.dim} spatial dimensions, got {len(spatial_shape)}"
            raise ValueError(msg)

        coord_arrays = [
            jnp.linspace(start, end, size)
            for size, (start, end) in zip(spatial_shape, self.grid_boundaries, strict=True)
        ]
        return list(jnp.meshgrid(*coord_arrays, indexing="ij"))

    def get_grid(self, spatial_shape: tuple[int, ...]) -> list[jax.Array]:
        """Return the per-axis coordinate meshgrid for ``spatial_shape``."""
        return self._generate_grid(spatial_shape)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed grid coordinates into the input tensor.

        Args:
            x: Input tensor of shape ``(batch, *spatial_dims, in_channels)``.

        Returns:
            Tensor of shape ``(batch, *spatial_dims, in_channels + dim)`` with the
            per-axis coordinate channels appended along the last axis.
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]  # All dimensions except batch and channels

        grids = self._generate_grid(spatial_shape)
        grid_tensors = [
            jnp.broadcast_to(
                jnp.expand_dims(grid, axis=(0, -1)),  # (spatial,) -> (1, *spatial, 1)
                (batch_size, *spatial_shape, 1),
            )
            for grid in grids
        ]
        return jnp.concatenate([x, *grid_tensors], axis=-1)


class GridEmbedding2D(GridEmbeddingND):
    """Two-dimensional specialization of :class:`GridEmbeddingND`.

    Expects inputs of shape ``(batch, height, width, channels)`` and appends two
    coordinate channels: one varying along the height axis, one along the width
    axis (matching the N-D / neuraloperator ``indexing="ij"`` convention). Kept
    as a thin alias for the 2D case used throughout the examples.

    Args:
        in_channels: Number of input channels.
        grid_boundaries: Per-axis ``[start, end]`` bounds ``[[h0, h1], [w0, w1]]``;
            defaults to ``[[0.0, 1.0], [0.0, 1.0]]``.
    """

    def __init__(
        self,
        in_channels: int,
        grid_boundaries: list[list[float]] | None = None,
    ) -> None:
        super().__init__(in_channels=in_channels, dim=2, grid_boundaries=grid_boundaries)


class SinusoidalEmbedding(EmbeddingBase):
    """
    Sinusoidal positional embedding in the style of Transformers and NeRFs.

    Expects inputs of shape (batch, n_points, in_channels) or (n_points, in_channels).

    Based on neuraloperator SinusoidalEmbedding but adapted for JAX/Flax NNX.

    Args:
        in_channels: Number of input channels
        num_frequencies: Number of frequencies in encoding
        embedding_type: 'transformer' or 'nerf' style encoding
        max_positions: Maximum positions for transformer encoding
    """

    def __init__(
        self,
        in_channels: int,
        num_frequencies: int | None = None,
        embedding_type: str = "transformer",
        max_positions: int = 10000,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies or in_channels

        allowed_types = ["transformer", "nerf"]
        if embedding_type not in allowed_types:
            raise ValueError(f"embedding_type must be one of {allowed_types}")

        self.embedding_type = embedding_type
        self.max_positions = max_positions

        if embedding_type == "transformer" and max_positions is None:
            raise ValueError("max_positions must be specified for transformer embedding")

    @property
    def out_channels(self) -> int:
        """Output channels = 2 * num_frequencies * input_channels."""
        return 2 * self.num_frequencies * self.in_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply sinusoidal embedding.

        Args:
            x: Input tensor of shape (batch, n_points, in_channels)
               or (n_points, in_channels)

        Returns:
            Embedded tensor with sinusoidal positional encoding
        """
        original_shape = x.shape

        # Handle both batched and unbatched inputs
        if x.ndim == 2:
            x = x[None, ...]  # Add batch dimension
            unbatched = True
        elif x.ndim == 3:
            unbatched = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D with shape {original_shape}")

        batch_size, n_points, _ = x.shape

        # Generate frequencies
        if self.embedding_type == "nerf":
            freqs = 2.0 ** jnp.arange(0, self.num_frequencies) * jnp.pi
        elif self.embedding_type == "transformer":
            # Reference: neuraloperator SinusoidalEmbedding.forward
            # (neuralop/layers/embeddings.py L278): exponent = arange(L) / L * 2.
            freqs = jnp.arange(0, self.num_frequencies) / self.num_frequencies * 2
            freqs = (1.0 / self.max_positions) ** freqs
        else:  # pragma: no cover - embedding_type validated in __init__
            raise AssertionError(f"unreachable: embedding_type={self.embedding_type!r}")

        # Compute frequency encodings
        # x: (batch, n_points, channels), freqs: (num_frequencies,)
        # Result: (batch, n_points, channels, num_frequencies)
        freq_encodings = jnp.einsum("bnc,f->bncf", x, freqs)

        # Apply sin and cos
        sin_encodings = jnp.sin(freq_encodings)
        cos_encodings = jnp.cos(freq_encodings)

        # Stack and reshape to interleave sin and cos
        encodings = jnp.stack([sin_encodings, cos_encodings], axis=-1)
        encodings = encodings.reshape(batch_size, n_points, -1)

        # Remove batch dimension if input was unbatched
        if unbatched:
            encodings = encodings[0]

        return encodings


class FunctionEmbedding(EmbeddingBase):
    """Function space embedding for neural operators.

    Provides embedding capabilities for function spaces in neural operator
    architectures, supporting various coordinate systems and transformations.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels

        # Linear projection for function embedding
        self.projection = nnx.Linear(in_channels, out_channels, rngs=nnx.Rngs(0))

    @property
    def out_channels(self) -> int:
        """Number of output channels after embedding."""
        return self._out_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply function embedding to input.

        Args:
            x: Input tensor of shape (batch, n_points, in_channels) or
               (n_points, in_channels)

        Returns:
            Embedded tensor of shape (batch, n_points, out_channels) or
            (n_points, out_channels)
        """
        return self.projection(x)


# Utility functions for grid generation


def regular_grid_2d(
    spatial_dims: tuple[int, int],
    grid_boundaries: list[list[float]] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Generate regular 2D coordinate grid.

    Args:
        spatial_dims: Spatial dimensions (height, width)
        grid_boundaries: Coordinate boundaries [[x_min, x_max], [y_min, y_max]]

    Returns:
        Tuple of (x_grid, y_grid) coordinate arrays
    """
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0], [0.0, 1.0]]
    h, w = spatial_dims
    x_coords = jnp.linspace(grid_boundaries[0][0], grid_boundaries[0][1], w)
    y_coords = jnp.linspace(grid_boundaries[1][0], grid_boundaries[1][1], h)

    grids = jnp.meshgrid(x_coords, y_coords, indexing="xy")
    return (grids[0], grids[1])


def irregular_grid_from_points(points: jax.Array) -> jax.Array:
    """
    Create grid embedding from irregular point coordinates.

    Args:
        points: Point coordinates of shape (n_points, spatial_dim)

    Returns:
        Grid embedding ready for neural operator input
    """
    return points


def grid_resolutions_from_boundaries(
    resolutions: list[int], grid_boundaries: list[list[float]] | None = None
) -> list[tuple[jax.Array, ...]]:
    """
    Generate multiple grid resolutions for multi-scale training.

    Args:
        resolutions: List of grid resolutions to generate
        grid_boundaries: Coordinate boundaries for each dimension

    Returns:
        List of coordinate grids at different resolutions
    """
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0], [0.0, 1.0]]

    if len(resolutions) != len(grid_boundaries):
        msg = "resolutions and grid_boundaries must have same length"
        raise RuntimeError(msg)

    grids = []
    for res in resolutions:
        if len(grid_boundaries) == 2:  # 2D case
            spatial_dims = (res, res)
            x_grid, y_grid = regular_grid_2d(spatial_dims, grid_boundaries)
            grids.append((x_grid, y_grid))
        else:
            # General N-D case can be extended here
            msg = f"Multi-resolution grids for {len(grid_boundaries)}D not implemented"
            raise NotImplementedError(msg)

    return grids
