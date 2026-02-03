"""Extended Physics-Informed Neural Network (XPINN).

XPINN extends the PINN framework to handle domain decomposition with
explicit interface conditions for continuity and flux matching.

Key Features:
    - Separate networks for each subdomain
    - Interface continuity conditions (u_left = u_right)
    - Flux continuity conditions (du/dn_left = du/dn_right)
    - Weighted loss combination for interface enforcement

References:
    - Jagtap & Karniadakis (2020): Extended Physics-Informed Neural Networks
    - Survey Section 8.3.1: XPINNs
    - GitHub: https://github.com/AmeyaJagtap/XPINNs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
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
class XPINNConfig:
    """Configuration for XPINN training.

    Attributes:
        continuity_weight: Weight for interface continuity loss (u_left = u_right)
        flux_weight: Weight for interface flux continuity loss (du/dn matching)
        residual_weight: Weight for PDE residual loss in each subdomain
        average_residual_weight: Weight for residual averaging at interfaces
    """

    continuity_weight: float = 1.0
    flux_weight: float = 1.0
    residual_weight: float = 1.0
    average_residual_weight: float = 0.0  # Optional residual averaging


class XPINN(DomainDecompositionPINN):
    """Extended Physics-Informed Neural Network.

    XPINN decomposes the computational domain into non-overlapping subdomains,
    training a separate neural network for each subdomain. Interface conditions
    enforce solution continuity and flux matching between adjacent subdomains.

    The total loss includes:
        - Data loss (if available)
        - PDE residual loss (per subdomain)
        - Interface continuity loss: ||u_left - u_right||²
        - Interface flux loss: ||∂u/∂n_left - ∂u/∂n_right||²

    Attributes:
        config: XPINN configuration with loss weights
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
        >>> model = XPINN(
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
        config: XPINNConfig | None = None,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize XPINN.

        Args:
            input_dim: Spatial dimension
            output_dim: Solution dimension
            subdomains: List of subdomain definitions
            interfaces: List of interface definitions
            hidden_dims: Hidden layer dimensions for subdomain networks
            config: XPINN configuration. Uses defaults if None.
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
        self.config = config or XPINNConfig()

    def compute_continuity_loss(self) -> Float[Array, ""]:
        """Compute interface continuity loss.

        Enforces u_left = u_right at all interface points.

        Returns:
            Scalar continuity loss (MSE of discontinuity)
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

            # Continuity loss: ||u_left - u_right||²
            loss = jnp.mean((u_left - u_right) ** 2)
            total_loss = total_loss + loss

        return total_loss / max(len(self.interfaces), 1)

    def compute_flux_loss(self) -> Float[Array, ""]:
        """Compute interface flux continuity loss.

        Enforces ∂u/∂n_left = ∂u/∂n_right at all interface points,
        where n is the interface normal direction.

        Returns:
            Scalar flux loss (MSE of flux discontinuity)
        """
        if not self.interfaces:
            return jnp.array(0.0)

        total_loss = jnp.array(0.0)

        for interface in self.interfaces:
            left_id, right_id = interface.subdomain_ids
            points = interface.points
            normal = interface.normal

            # Compute gradients using autodiff
            def compute_gradient(network, x):
                """Compute gradient of network output w.r.t. input."""

                def scalar_output(xi):
                    return network(xi.reshape(1, -1)).squeeze()

                return jax.vmap(jax.grad(scalar_output))(x)

            grad_left = compute_gradient(list(self.networks)[left_id], points)
            grad_right = compute_gradient(list(self.networks)[right_id], points)

            # Flux = gradient dot normal
            flux_left = jnp.sum(grad_left * normal, axis=-1)
            flux_right = jnp.sum(grad_right * normal, axis=-1)

            # Flux continuity loss
            loss = jnp.mean((flux_left - flux_right) ** 2)
            total_loss = total_loss + loss

        return total_loss / max(len(self.interfaces), 1)

    def compute_interface_loss(self) -> Float[Array, ""]:
        """Compute total weighted interface loss.

        Combines continuity and flux losses with configured weights.

        Returns:
            Scalar total interface loss
        """
        continuity_loss = self.compute_continuity_loss()
        flux_loss = self.compute_flux_loss()

        return (
            self.config.continuity_weight * continuity_loss
            + self.config.flux_weight * flux_loss
        )

    def compute_subdomain_residual(
        self,
        subdomain_id: int,
        residual_fn: Callable[
            [
                Callable[[Float[Array, ...]], Float[Array, "batch out"]],
                Float[Array, ...],
            ],
            Float[Array, " batch"],
        ],
        collocation_points: Float[Array, ...],
    ) -> Float[Array, ""]:
        """Compute PDE residual for a specific subdomain.

        Args:
            subdomain_id: ID of the subdomain
            residual_fn: Function that computes PDE residual given network and points
            collocation_points: Points where to evaluate residual

        Returns:
            Scalar residual loss for this subdomain
        """
        network = list(self.networks)[subdomain_id]
        residuals = residual_fn(network, collocation_points)
        return jnp.mean(residuals**2)

    def compute_total_residual(
        self,
        residual_fn: Callable[
            [
                Callable[[Float[Array, ...]], Float[Array, "batch out"]],
                Float[Array, ...],
            ],
            Float[Array, " batch"],
        ],
        collocation_points_per_subdomain: Sequence[Float[Array, ...]],
    ) -> Float[Array, ""]:
        """Compute total PDE residual across all subdomains.

        Args:
            residual_fn: Function that computes PDE residual
            collocation_points_per_subdomain: Collocation points for each subdomain

        Returns:
            Scalar total residual loss
        """
        total_loss = jnp.array(0.0)

        for i, points in enumerate(collocation_points_per_subdomain):
            loss = self.compute_subdomain_residual(i, residual_fn, points)
            total_loss = total_loss + loss

        return total_loss / len(collocation_points_per_subdomain)
