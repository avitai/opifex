"""Augmented Physics-Informed Neural Network (APINN).

APINN uses a learnable gating network to smoothly blend subdomain solutions,
allowing the model to learn optimal subdomain selection.

Key Features:
    - Learnable gating network for subdomain weighting
    - Temperature-controlled softmax for soft/hard selection
    - Differentiable blending for end-to-end training

References:
    - Survey Section 8.3.3: Augmented PINNs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import (
    DomainDecompositionPINN,
    Interface,
    Subdomain,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


@dataclass(frozen=True)
class APINNConfig:
    """Configuration for APINN training.

    Attributes:
        temperature: Softmax temperature for gating. Lower values give
                    sharper (more discrete) weights, higher values give
                    smoother (more uniform) weights.
        gating_hidden_dims: Hidden dimensions for the gating network
        continuity_weight: Weight for interface continuity loss
    """

    temperature: float = 1.0
    gating_hidden_dims: list[int] = field(default_factory=lambda: [16, 16])
    continuity_weight: float = 1.0


class GatingNetwork(nnx.Module):
    """Gating network for subdomain selection.

    This network takes spatial coordinates and outputs weights
    for blending subdomain solutions.

    Attributes:
        layers: List of linear layers
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        num_subdomains: int,
        hidden_dims: Sequence[int],
        *,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize gating network.

        Args:
            input_dim: Input spatial dimension
            num_subdomains: Number of subdomains to gate
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            rngs: Random number generators
        """
        self.activation = activation
        self.num_subdomains = num_subdomains

        layers = []
        dims = [input_dim, *hidden_dims, num_subdomains]

        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

        self.layers = nnx.List(layers)

    def __call__(
        self, x: Float[Array, "batch dim"], temperature: float = 1.0
    ) -> Float[Array, "batch num_subdomains"]:
        """Compute gating weights.

        Args:
            x: Input coordinates
            temperature: Softmax temperature

        Returns:
            Gating weights for each subdomain (sum to 1)
        """
        h = x
        for layer in list(self.layers)[:-1]:
            h = layer(h)
            h = self.activation(h)

        # Final layer outputs logits
        logits = list(self.layers)[-1](h)

        # Temperature-scaled softmax
        return nnx.softmax(logits / temperature, axis=-1)


class APINN(DomainDecompositionPINN):
    """Augmented Physics-Informed Neural Network.

    APINN uses a learnable gating network to determine how to blend
    solutions from different subdomains. Unlike FBPINN which uses
    fixed window functions, APINN learns the optimal blending.

    The output is computed as:
        u(x) = Σᵢ gᵢ(x) * uᵢ(x)

    where gᵢ(x) are the learned gating weights (sum to 1) and
    uᵢ(x) are the subdomain network outputs.

    Attributes:
        config: APINN configuration
        gating_network: Network that produces blending weights
        input_dim: Spatial dimension
        output_dim: Solution dimension
        subdomains: List of subdomain definitions
        interfaces: List of interface definitions
        networks: List of subdomain networks

    Example:
        >>> subdomains = [
        ...     Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
        ...     Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ... ]
        >>> interfaces = [
        ...     Interface(subdomain_ids=(0, 1), points=jnp.array([[0.5]]),
        ...               normal=jnp.array([1.0]))
        ... ]
        >>> model = APINN(
        ...     input_dim=1, output_dim=1,
        ...     subdomains=subdomains, interfaces=interfaces,
        ...     hidden_dims=[32, 32], rngs=nnx.Rngs(0)
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subdomains: Sequence[Subdomain],
        interfaces: Sequence[Interface],
        hidden_dims: Sequence[int],
        *,
        config: APINNConfig | None = None,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize APINN.

        Args:
            input_dim: Spatial dimension
            output_dim: Solution dimension
            subdomains: List of subdomain definitions
            interfaces: List of interface definitions
            hidden_dims: Hidden layer dimensions for subdomain networks
            config: APINN configuration. Uses defaults if None.
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
        self.config = config or APINNConfig()

        # Create gating network
        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            num_subdomains=len(subdomains),
            hidden_dims=list(self.config.gating_hidden_dims),
            activation=activation,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "batch dim"]) -> Float[Array, "batch out"]:
        """Compute solution with learned gating.

        Args:
            x: Input coordinates

        Returns:
            Blended solution from all subdomains
        """
        # Get outputs from all subdomain networks
        subdomain_outputs = self.get_subdomain_outputs(x)

        # Get gating weights
        weights = self.get_gating_weights(x)

        # Weighted combination: Σᵢ gᵢ(x) * uᵢ(x)
        stacked = jnp.stack(subdomain_outputs, axis=-1)  # (batch, out, num_subdomains)
        weights_expanded = weights[:, None, :]  # (batch, 1, num_subdomains)

        return jnp.sum(stacked * weights_expanded, axis=-1)

    def get_gating_weights(
        self, x: Float[Array, "batch dim"]
    ) -> Float[Array, "batch num_subdomains"]:
        """Get gating weights for given points.

        Args:
            x: Input coordinates

        Returns:
            Gating weights for each subdomain
        """
        return self.gating_network(x, temperature=self.config.temperature)

    def compute_interface_loss(self) -> Float[Array, ""]:
        """Compute interface continuity loss.

        Even with learned gating, we can still encourage continuity
        at explicit interface points.

        Returns:
            Scalar interface loss
        """
        if not self.interfaces:
            return jnp.array(0.0)

        total_loss = jnp.array(0.0)

        for interface in self.interfaces:
            left_id, right_id = interface.subdomain_ids
            points = interface.points

            # Get predictions from both subdomains
            u_left = list(self.networks)[left_id](points)
            u_right = list(self.networks)[right_id](points)

            # Continuity loss
            loss = jnp.mean((u_left - u_right) ** 2)
            total_loss = total_loss + loss

        return self.config.continuity_weight * total_loss / max(len(self.interfaces), 1)
