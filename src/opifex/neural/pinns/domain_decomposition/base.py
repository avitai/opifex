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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


@dataclass
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


@dataclass
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
    ):
        """Initialize subdomain network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            rngs: Random number generators
        """
        self.activation = activation

        layers = []
        dims = [input_dim, *hidden_dims, output_dim]

        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

        # Use nnx.List for FLAX NNX 0.12.0 compatibility
        self.layers = nnx.List(layers)

    def __call__(self, x: Float[Array, ...]) -> Float[Array, "batch out"]:
        """Forward pass through the network."""
        h = x
        for layer in list(self.layers)[:-1]:
            h = layer(h)
            h = self.activation(h)
        return list(self.layers)[-1](h)


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
    ):
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

    def get_subdomain_outputs(
        self, x: Float[Array, ...]
    ) -> list[Float[Array, "batch out"]]:
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


def uniform_partition(  # noqa: PLR0912
    bounds: Float[Array, "dim 2"],
    num_partitions: tuple[int, ...],
    interface_points: int = 10,
) -> tuple[list[Subdomain], list[Interface]]:
    """Create uniform partition of a rectangular domain.

    Args:
        bounds: Domain bounds, shape (dim, 2) with [min, max] for each dimension
        num_partitions: Number of partitions in each dimension
        interface_points: Number of sample points per interface

    Returns:
        Tuple of (subdomains, interfaces)
    """
    dim = bounds.shape[0]
    if len(num_partitions) != dim:
        msg = f"num_partitions length {len(num_partitions)} must match bounds dim {dim}"
        raise ValueError(msg)

    # Compute partition boundaries in each dimension
    partition_bounds = []
    for d in range(dim):
        edges = jnp.linspace(bounds[d, 0], bounds[d, 1], num_partitions[d] + 1)
        partition_bounds.append(edges)

    # Create subdomains
    subdomains = []
    subdomain_id = 0

    if dim == 1:
        for i in range(num_partitions[0]):
            lo, hi = partition_bounds[0][i], partition_bounds[0][i + 1]
            sub_bounds = jnp.array([[lo, hi]])
            subdomains.append(Subdomain(id=subdomain_id, bounds=sub_bounds))
            subdomain_id += 1
    elif dim == 2:
        for i in range(num_partitions[0]):
            for j in range(num_partitions[1]):
                sub_bounds = jnp.array(
                    [
                        [partition_bounds[0][i], partition_bounds[0][i + 1]],
                        [partition_bounds[1][j], partition_bounds[1][j + 1]],
                    ]
                )
                subdomains.append(Subdomain(id=subdomain_id, bounds=sub_bounds))
                subdomain_id += 1
    else:
        raise NotImplementedError("Only 1D and 2D partitioning implemented")

    # Create interfaces
    interfaces = []

    if dim == 1:
        for i in range(num_partitions[0] - 1):
            interface_x = partition_bounds[0][i + 1]
            points = jnp.array([[interface_x]] * interface_points)
            interfaces.append(
                Interface(
                    subdomain_ids=(i, i + 1),
                    points=points,
                    normal=jnp.array([1.0]),
                )
            )
    elif dim == 2:
        # Vertical interfaces (x-direction)
        for i in range(num_partitions[0] - 1):
            for j in range(num_partitions[1]):
                interface_x = partition_bounds[0][i + 1]
                y_min = partition_bounds[1][j]
                y_max = partition_bounds[1][j + 1]

                points = jnp.column_stack(
                    [
                        jnp.full(interface_points, interface_x),
                        jnp.linspace(y_min, y_max, interface_points),
                    ]
                )

                left_id = i * num_partitions[1] + j
                right_id = (i + 1) * num_partitions[1] + j

                interfaces.append(
                    Interface(
                        subdomain_ids=(left_id, right_id),
                        points=points,
                        normal=jnp.array([1.0, 0.0]),
                    )
                )

        # Horizontal interfaces (y-direction)
        for i in range(num_partitions[0]):
            for j in range(num_partitions[1] - 1):
                interface_y = partition_bounds[1][j + 1]
                x_min = partition_bounds[0][i]
                x_max = partition_bounds[0][i + 1]

                points = jnp.column_stack(
                    [
                        jnp.linspace(x_min, x_max, interface_points),
                        jnp.full(interface_points, interface_y),
                    ]
                )

                bottom_id = i * num_partitions[1] + j
                top_id = i * num_partitions[1] + (j + 1)

                interfaces.append(
                    Interface(
                        subdomain_ids=(bottom_id, top_id),
                        points=points,
                        normal=jnp.array([0.0, 1.0]),
                    )
                )

    return subdomains, interfaces
