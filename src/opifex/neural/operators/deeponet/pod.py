"""POD-enhanced DeepONet (PODDeepONet).

Replaces the trunk network in DeepONet with pre-computed POD (Proper
Orthogonal Decomposition) basis modes. The branch network learns
coefficients for the POD expansion.

Reference:
    Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2022).
    "A comprehensive and fair comparison of two neural operators."
    GitHub: lu-group/deeponet-fno

This module reuses:
- ``StandardMLP`` from ``opifex.neural.base`` for the branch network
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.base import StandardMLP


logger = logging.getLogger(__name__)


class PODDeepONet(nnx.Module):
    """POD-enhanced DeepONet.

    Uses pre-computed POD basis modes instead of a trunk network.
    The branch network outputs coefficients that are combined with the
    fixed basis via::

        output = branch_coefficients @ pod_basis.T

    Optionally adds a pre-computed mean to the output.

    Args:
        branch_sizes: Layer sizes for the branch network. The last
            element must equal the number of POD modes.
        pod_basis: Pre-computed POD basis modes, shape
            ``(n_locations, n_modes)``.
        output_mean: Optional mean to add to the output, shape
            ``(n_locations,)``.
        activation: Activation function for the branch network.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        branch_sizes: list[int],
        pod_basis: jax.Array,
        *,
        output_mean: jax.Array | None = None,
        activation: str = "gelu",
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        n_locations, n_modes = pod_basis.shape

        # Validate: branch output dim must equal number of POD modes
        if branch_sizes[-1] != n_modes:
            raise ValueError(
                f"branch output dim ({branch_sizes[-1]}) must match "
                f"number of POD modes ({n_modes})"
            )

        self.n_locations = n_locations
        self.n_modes = n_modes

        # Branch network: inputs -> POD coefficients
        self.branch_net = StandardMLP(
            layer_sizes=branch_sizes,
            activation=activation,
            rngs=rngs,
        )

        # POD basis: non-trainable (nnx.Variable, not nnx.Param)
        self.pod_basis_modes = nnx.Variable(pod_basis)

        # Optional output mean: non-trainable
        self.output_mean = (
            nnx.Variable(output_mean) if output_mean is not None else None
        )

        logger.info(
            "PODDeepONet: %d modes, %d locations, branch=%s",
            n_modes,
            n_locations,
            branch_sizes,
        )

    def __call__(
        self,
        branch_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass.

        Args:
            branch_input: Branch network input ``(B, input_dim)``.
            deterministic: If True, disable dropout.

        Returns:
            Output ``(B, n_locations)``.
        """
        # Branch: (B, input_dim) -> (B, n_modes)
        coeffs = self.branch_net(branch_input, deterministic=deterministic)

        # Combine: coeffs @ basis.T -> (B, n_locations)
        # coeffs: (B, n_modes), basis: (n_locations, n_modes)
        out = jnp.einsum("bi,ni->bn", coeffs, self.pod_basis_modes[...])

        # Add mean if provided
        if self.output_mean is not None:
            out = out + self.output_mean[...]

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pod_deeponet(
    branch_sizes: list[int],
    pod_basis: jax.Array,
    *,
    output_mean: jax.Array | None = None,
    activation: str = "gelu",
    rngs: nnx.Rngs,
) -> PODDeepONet:
    """Create a PODDeepONet model.

    Args:
        branch_sizes: Layer sizes for the branch MLP.
        pod_basis: Pre-computed POD basis ``(n_locations, n_modes)``.
        output_mean: Optional pre-computed output mean.
        activation: Activation function name.
        rngs: Flax NNX random number generators.

    Returns:
        Configured PODDeepONet.
    """
    return PODDeepONet(
        branch_sizes=branch_sizes,
        pod_basis=pod_basis,
        output_mean=output_mean,
        activation=activation,
        rngs=rngs,
    )


__all__ = [
    "PODDeepONet",
    "create_pod_deeponet",
]
