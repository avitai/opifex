"""Base classes for Domain Decomposition PINNs.

This module provides the foundational classes for domain decomposition
approaches to physics-informed neural networks.

Key Classes:
    - Subdomain: Represents a subdomain region in the computational domain
    - Interface: Represents the interface between adjacent subdomains
    - DomainDecompositionPINN: Abstract base class for DD-PINN variants

Design Principles:
    - Each subdomain has its own neural network
    - Interfaces enforce continuity and flux matching
    - Window functions provide smooth blending (for FBPINN variants)

References:
    - Survey Section 8.3: Domain Decomposition Methods
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.pinns.dense_stack import DenseStack


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


@dataclass(frozen=True, slots=True, kw_only=True)
class Subdomain:
    """Representation of a subdomain in the computational domain.

    A subdomain is a rectangular region defined by its bounds in each
    spatial dimension.

    Attributes:
        id: Unique identifier for this subdomain
        bounds: Array of shape (dim, 2) with [min, max] for each dimension
        overlap: Optional overlap with neighboring subdomains (for Schwarz methods)
    """

    id: int
    bounds: Float[Array, "dim 2"]
    overlap: float = 0.0

    def contains(self, x: Float[Array, " dim"]) -> Array:
        """Check if a point is inside this subdomain.

        Args:
            x: Point coordinates of shape (dim,)

        Returns:
            Boolean array (scalar) indicating if point is inside subdomain
        """
        return jnp.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1]))

    @property
    def center(self) -> Float[Array, " dim"]:
        """Compute the center of the subdomain."""
        return jnp.mean(self.bounds, axis=1)

    @property
    def volume(self) -> Float[Array, ""]:
        """Compute the volume (area in 2D, length in 1D) of the subdomain."""
        sizes = self.bounds[:, 1] - self.bounds[:, 0]
        return jnp.prod(sizes)


@dataclass(frozen=True, slots=True, kw_only=True)
class Interface:
    """Representation of an interface between two subdomains.

    The interface stores sample points for enforcing continuity conditions
    between adjacent subdomains.

    Attributes:
        subdomain_ids: Tuple of (left_id, right_id) for adjacent subdomains
        points: Sample points on the interface, shape (num_points, dim)
        normal: Outward normal vector from first subdomain, shape (dim,)
    """

    subdomain_ids: tuple[int, int]
    points: Float[Array, "num_points dim"]
    normal: Float[Array, " dim"]


class SubdomainNetwork(nnx.Module):
    """Neural network for a single subdomain.

    A simple MLP that processes inputs for a specific subdomain.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize subdomain network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            rngs: Random number generators
        """
        self.activation = activation

        self.layers = DenseStack(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, ...]) -> Float[Array, "batch out"]:
        """Forward pass through the network."""
        return self.layers(x)


class DomainDecompositionPINN(nnx.Module):
    """Base class for Domain Decomposition PINNs.

    This class provides the infrastructure for training separate networks
    on subdomains with interface coupling conditions.

    Attributes:
        input_dim: Input spatial dimension
        output_dim: Output dimension (solution fields)
        subdomains: List of subdomain definitions
        interfaces: List of interface definitions
        networks: List of subdomain networks
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subdomains: Sequence[Subdomain],
        interfaces: Sequence[Interface],
        hidden_dims: Sequence[int],
        *,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Domain Decomposition PINN.

        Args:
            input_dim: Input spatial dimension
            output_dim: Output dimension
            subdomains: List of subdomain definitions
            interfaces: List of interface definitions
            hidden_dims: Hidden layer dimensions (shared across subdomains)
            activation: Activation function
            rngs: Random number generators
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.subdomains = list(subdomains)
        self.interfaces = list(interfaces)
        self.activation = activation

        # Create network for each subdomain
        networks = []
        for _ in subdomains:
            network = SubdomainNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                rngs=rngs,
            )
            networks.append(network)
        # Use nnx.List for FLAX NNX 0.12.0 compatibility
        self.networks = nnx.List(networks)

    def __call__(self, x: Float[Array, ...]) -> Float[Array, "batch out"]:
        """Compute solution at given points.

        For each point, finds the containing subdomain and evaluates
        the corresponding network.

        Args:
            x: Input coordinates, shape (batch, input_dim)

        Returns:
            Solution values, shape (batch, output_dim)
        """
        # Get outputs from all subdomain networks
        subdomain_outputs = self.get_subdomain_outputs(x)

        # Compute weights based on subdomain membership
        weights = self._compute_subdomain_weights(x)

        # Weighted combination
        stacked = jnp.stack(subdomain_outputs, axis=-1)  # (batch, out, num_subdomains)
        weights_expanded = weights[:, None, :]  # (batch, 1, num_subdomains)

        return jnp.sum(stacked * weights_expanded, axis=-1)

    def get_subdomain_outputs(self, x: Float[Array, ...]) -> list[Float[Array, "batch out"]]:
        """Get outputs from all subdomain networks.

        Args:
            x: Input coordinates

        Returns:
            List of outputs from each subdomain network
        """
        return [network(x) for network in self.networks]

    def _compute_subdomain_weights(
        self, x: Float[Array, ...]
    ) -> Float[Array, "batch num_subdomains"]:
        """Compute weights for subdomain combination.

        Default implementation uses hard assignment based on containment.
        Subclasses (like FBPINN) can override with smooth window functions.

        Args:
            x: Input coordinates

        Returns:
            Weights for each subdomain, shape (batch, num_subdomains)
        """
        batch_size = x.shape[0]
        num_subdomains = len(self.subdomains)
        weights = jnp.zeros((batch_size, num_subdomains))

        for i, subdomain in enumerate(self.subdomains):
            # Check containment for each point
            # Bind subdomain to default argument to avoid late binding closure issue
            in_subdomain = jax.vmap(
                lambda pt, sd=subdomain: jnp.all(
                    (pt >= sd.bounds[:, 0]) & (pt <= sd.bounds[:, 1])
                ).astype(jnp.float32)
            )(x)
            weights = weights.at[:, i].set(in_subdomain)

        # Normalize weights (handle overlapping regions)
        weights_sum = jnp.sum(weights, axis=-1, keepdims=True)
        return jnp.where(weights_sum > 0, weights / (weights_sum + 1e-8), weights)

    def compute_interface_residual(self) -> Float[Array, ""]:
        """Compute interface continuity residual.

        Enforces u_left = u_right at interface points.

        Returns:
            Scalar interface residual (MSE of discontinuity)
        """
        if not self.interfaces:
            return jnp.array(0.0)

        total_residual = jnp.array(0.0)

        for interface in self.interfaces:
            left_id, right_id = interface.subdomain_ids
            points = interface.points

            # Get predictions from both subdomains
            u_left = self.networks[left_id](points)
            u_right = self.networks[right_id](points)

            # Continuity residual
            residual = jnp.mean((u_left - u_right) ** 2)
            total_residual = total_residual + residual

        return total_residual / len(self.interfaces)

    def compute_flux_residual(
        self,
        derivative_fn: Callable[[nnx.Module, Float[Array, ...]], Float[Array, ...]],
    ) -> Float[Array, ""]:
        """Compute interface flux continuity residual.

        Enforces (du/dn)_left = (du/dn)_right at interface points.

        Args:
            derivative_fn: Function to compute gradient of network output

        Returns:
            Scalar flux residual
        """
        if not self.interfaces:
            return jnp.array(0.0)

        total_residual = jnp.array(0.0)

        for interface in self.interfaces:
            left_id, right_id = interface.subdomain_ids
            points = interface.points
            normal = interface.normal

            # Get gradients from both subdomains
            grad_left = derivative_fn(self.networks[left_id], points)
            grad_right = derivative_fn(self.networks[right_id], points)

            # Flux = gradient dot normal
            flux_left = jnp.sum(grad_left * normal, axis=-1)
            flux_right = jnp.sum(grad_right * normal, axis=-1)

            # Flux continuity residual
            residual = jnp.mean((flux_left - flux_right) ** 2)
            total_residual = total_residual + residual

        return total_residual / len(self.interfaces)


def _cell_id(cell: tuple[int, ...], num_partitions: tuple[int, ...]) -> int:
    """Map a per-axis cell index to a flat row-major (C-order) subdomain id.

    The ordering matches ``itertools.product`` and the legacy 2D convention
    ``id = i * ny + j``, generalised to N dimensions via row-major raveling.

    Args:
        cell: Per-axis cell indices, length ``dim``
        num_partitions: Number of partitions in each dimension

    Returns:
        Flat subdomain identifier
    """
    return int(np.ravel_multi_index(cell, num_partitions))


def _face_sample_points(
    cell: tuple[int, ...],
    axis: int,
    partition_bounds: list[Array],
    interface_points: int,
) -> Array:
    """Sample points on the shared face between a cell and its ``axis`` neighbour.

    The face is fixed at the upper edge of ``cell`` along ``axis`` and spans the
    extent of the cell in every other dimension. ``interface_points`` points are
    laid out along each of the (dim-1) transverse axes via a tensor-product
    meshgrid, generalising the 1D/2D face sampling to N-D. A 2D face therefore
    keeps the legacy ``interface_points`` total, while a 3D face yields an
    ``interface_points x interface_points`` grid. For a 1D face (a single point,
    no transverse axes) the point is tiled to preserve the documented shape.

    Args:
        cell: Per-axis cell indices of the lower-side subdomain
        axis: Axis normal to the shared face
        partition_bounds: Per-axis partition edge arrays
        interface_points: Number of sample points along each transverse axis

    Returns:
        Sample coordinates, shape ``(num_points, dim)``
    """
    dim = len(cell)
    face_value = partition_bounds[axis][cell[axis] + 1]
    has_transverse_axis = dim > 1

    axis_grids: list[Array] = []
    for d in range(dim):
        if d == axis:
            axis_grids.append(jnp.array([face_value]))
        else:
            lo = partition_bounds[d][cell[d]]
            hi = partition_bounds[d][cell[d] + 1]
            axis_grids.append(jnp.linspace(lo, hi, interface_points))

    mesh = jnp.meshgrid(*axis_grids, indexing="ij")
    points = jnp.stack([component.ravel() for component in mesh], axis=-1)

    # A 1D face is a single point; tile it to honour the documented point count.
    if not has_transverse_axis:
        points = jnp.tile(points, (interface_points, 1))
    return points


def uniform_partition(
    bounds: Float[Array, "dim 2"],
    num_partitions: tuple[int, ...],
    interface_points: int = 10,
) -> tuple[list[Subdomain], list[Interface]]:
    """Create a uniform N-D partition of a rectangular (hyperrectangular) domain.

    The domain is tiled into a tensor-product grid of axis-aligned subdomains,
    one per grid cell, with subdomain ids enumerated in row-major (C) order.
    Internal faces between axis-adjacent cells become :class:`Interface`
    objects with an axis-aligned unit normal and a grid of sample points on the
    shared face. The construction is dimension-agnostic and works for 1D, 2D,
    3D and higher (no per-dimension special-casing).

    Reference:
        Moseley, Markham, Nissen-Meyer (2023), "Finite Basis Physics-Informed
        Neural Networks", arXiv:2107.07871. The FBPINN subdomain tiling is a
        tensor product across dimensions; see ``RectangularDecompositionND`` in
        the reference implementation (https://github.com/benmoseley/FBPINNs),
        which lays out subdomains via ``np.meshgrid(*subdomain_xs)``.

    Args:
        bounds: Domain bounds, shape ``(dim, 2)`` with ``[min, max]`` per axis
        num_partitions: Number of partitions in each dimension (length ``dim``)
        interface_points: Target number of sample points per interface face

    Returns:
        Tuple of ``(subdomains, interfaces)``
    """
    dim = bounds.shape[0]
    if len(num_partitions) != dim:
        msg = f"num_partitions length {len(num_partitions)} must match bounds dim {dim}"
        raise ValueError(msg)

    # Per-axis partition edges: num_partitions[d] cells -> num_partitions[d]+1 edges.
    partition_bounds = [
        jnp.linspace(bounds[d, 0], bounds[d, 1], num_partitions[d] + 1) for d in range(dim)
    ]

    # Tensor-product subdomain grid, enumerated row-major to match _cell_id.
    cell_ranges = [range(n) for n in num_partitions]
    subdomains = [
        Subdomain(
            id=_cell_id(cell, num_partitions),
            bounds=jnp.array(
                [
                    [partition_bounds[d][cell[d]], partition_bounds[d][cell[d] + 1]]
                    for d in range(dim)
                ]
            ),
        )
        for cell in itertools.product(*cell_ranges)
    ]

    # Internal faces: for each axis, connect cells adjacent along that axis.
    interfaces: list[Interface] = []
    for axis in range(dim):
        for cell in itertools.product(*cell_ranges):
            if cell[axis] + 1 >= num_partitions[axis]:
                continue  # no neighbour on the far side along this axis
            neighbour = (*cell[:axis], cell[axis] + 1, *cell[axis + 1 :])
            normal = jnp.zeros(dim).at[axis].set(1.0)
            interfaces.append(
                Interface(
                    subdomain_ids=(
                        _cell_id(cell, num_partitions),
                        _cell_id(neighbour, num_partitions),
                    ),
                    points=_face_sample_points(cell, axis, partition_bounds, interface_points),
                    normal=normal,
                )
            )

    return subdomains, interfaces
