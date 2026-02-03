"""Conservative Physics-Informed Neural Network (cPINN).

cPINN extends XPINN with explicit flux conservation at interfaces,
enforcing strong conservation properties required for conservation laws.

Key Features:
    - Explicit flux computation at interfaces
    - Strong conservation enforcement
    - Weighted combination of continuity and flux losses

References:
    - Jagtap et al. (2020): Conservative physics-informed neural networks
    - Survey Section 8.3.2: Conservative PINNs
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
class CPINNConfig:
    """Configuration for cPINN training.

    Attributes:
        flux_weight: Weight for flux conservation loss at interfaces
        continuity_weight: Weight for solution continuity loss
        conservation_weight: Weight for global conservation enforcement
    """

    flux_weight: float = 1.0
    continuity_weight: float = 1.0
    conservation_weight: float = 0.1


def compute_flux(
    u_fn: Callable[[Float[Array, ...]], Float[Array, "batch out"]],
    x: Float[Array, ...],
) -> Float[Array, ...]:
    """Compute flux (gradient) of a function at given points.

    For scalar output, flux is the gradient: F = ∇u
    For vector output, this computes the Jacobian.

    Args:
        u_fn: Function mapping points to solution values
        x: Points at which to compute flux

    Returns:
        Flux vectors at each point, shape (batch, dim)
    """

    def scalar_grad(xi):
        """Compute gradient for a single point."""

        def scalar_fn(x_single):
            return u_fn(x_single.reshape(1, -1)).squeeze()

        return jax.grad(scalar_fn)(xi)

    return jax.vmap(scalar_grad)(x)


def compute_flux_conservation_residual(
    flux_left: Float[Array, ...],
    flux_right: Float[Array, ...],
    normal: Float[Array, " dim"],
) -> Float[Array, ""]:
    """Compute flux conservation residual at interface.

    For conservation laws, the normal flux must be continuous:
        F_left · n = F_right · n

    Args:
        flux_left: Flux from left subdomain
        flux_right: Flux from right subdomain
        normal: Interface normal vector

    Returns:
        Scalar residual (MSE of flux mismatch)
    """
    # Project flux onto normal direction
    flux_n_left = jnp.sum(flux_left * normal, axis=-1)
    flux_n_right = jnp.sum(flux_right * normal, axis=-1)

    # Conservation residual
    return jnp.mean((flux_n_left - flux_n_right) ** 2)


class CPINN(DomainDecompositionPINN):
    """Conservative Physics-Informed Neural Network.

    cPINN enforces strong conservation at subdomain interfaces by
    explicitly computing and matching fluxes across boundaries.

    The total interface loss includes:
        - Continuity loss: ||u_left - u_right||²
        - Flux conservation loss: ||F_left · n - F_right · n||²

    where F = ∇u is the flux (gradient) of the solution.

    Attributes:
        config: cPINN configuration with loss weights
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
        >>> model = CPINN(
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
        config: CPINNConfig | None = None,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize cPINN.

        Args:
            input_dim: Spatial dimension
            output_dim: Solution dimension
            subdomains: List of subdomain definitions
            interfaces: List of interface definitions
            hidden_dims: Hidden layer dimensions for subdomain networks
            config: cPINN configuration. Uses defaults if None.
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
        self.config = config or CPINNConfig()

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

            # Continuity loss
            loss = jnp.mean((u_left - u_right) ** 2)
            total_loss = total_loss + loss

        return total_loss / max(len(self.interfaces), 1)

    def compute_flux_conservation_loss(self) -> Float[Array, ""]:
        """Compute flux conservation loss at interfaces.

        Enforces F_left · n = F_right · n at all interface points,
        where F = ∇u is the flux.

        Returns:
            Scalar flux conservation loss
        """
        if not self.interfaces:
            return jnp.array(0.0)

        total_loss = jnp.array(0.0)

        for interface in self.interfaces:
            left_id, right_id = interface.subdomain_ids
            points = interface.points
            normal = interface.normal

            # Compute fluxes from both subdomains
            flux_left = compute_flux(list(self.networks)[left_id], points)
            flux_right = compute_flux(list(self.networks)[right_id], points)

            # Flux conservation residual
            loss = compute_flux_conservation_residual(flux_left, flux_right, normal)
            total_loss = total_loss + loss

        return total_loss / max(len(self.interfaces), 1)

    def compute_interface_loss(self) -> Float[Array, ""]:
        """Compute total weighted interface loss.

        Combines continuity and flux conservation losses with configured weights.

        Returns:
            Scalar total interface loss
        """
        continuity_loss = self.compute_continuity_loss()
        flux_loss = self.compute_flux_conservation_loss()

        return (
            self.config.continuity_weight * continuity_loss
            + self.config.flux_weight * flux_loss
        )
