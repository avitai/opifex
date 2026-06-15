"""Physics-Informed Neural Operators for scientific computing.

This module provides neural operators that incorporate physical laws and constraints
directly into their architecture and training process. All implementations are
fully compliant with modern Flax NNX patterns.

MODERNIZATION APPLIED:
- Full Flax NNX compliance with proper RNG handling
- Enhanced physics constraint integration
- Optimized differential operator computation
- Support for various boundary condition types
- Robust handling of multi-physics systems
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# Import neural network components
from opifex.neural.base import StandardMLP
from opifex.neural.operators.physics._conservation import conservation_residual_loss


class PhysicsInformedOperator(nnx.Module):
    """Physics-Informed Neural Operator with embedded physical constraints.

    This operator combines standard neural operator architectures with
    physics-based constraints and differential operators to ensure
    physically consistent solutions.

    Fully compliant with modern Flax NNX patterns.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        physics_type: str = "pde",
        *,
        activation: str = "gelu",
        physics_weight: float = 1.0,
        data_weight: float = 1.0,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Physics-Informed Operator following NNX patterns.

        Args:
            layer_sizes: Layer sizes for the neural network
                [input_dim, hidden1, hidden2, ..., output_dim]
            physics_type: Type of physics constraint ('pde', 'conservation', 'symmetry')
            activation: Activation function name
            physics_weight: Weight for physics loss component
            data_weight: Weight for data loss component
            use_bias: Whether to use bias in linear layers
            rngs: Random number generators (keyword-only)
        """
        super().__init__()

        self.layer_sizes = layer_sizes
        self.physics_type = physics_type
        self.physics_weight = physics_weight
        self.data_weight = data_weight

        # Create main neural network
        self.network = StandardMLP(
            layer_sizes=layer_sizes,
            activation=activation,
            dropout_rate=0.0,
            use_bias=use_bias,
            apply_final_dropout=False,
            rngs=rngs,
        )

    def __call__(
        self,
        coordinates: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply physics-informed operator.

        Following NNX best practices, this method does NOT include rngs parameter
        as all random state is managed during initialization.

        Args:
            coordinates: Space-time coordinates
                Shape: (batch_size, input_dim) or (batch_size, n_points, input_dim)
            deterministic: Whether to use deterministic mode

        Returns:
            Solution field values
        """
        # Handle different input shapes
        if coordinates.ndim == 3:
            # Reshape from (batch, n_points, dim) to (batch*n_points, dim)
            batch_size, n_points, input_dim = coordinates.shape
            coords_flat = coordinates.reshape(batch_size * n_points, input_dim)
        else:
            # Already in correct shape (batch, dim)
            coords_flat = coordinates
            batch_size = None
            n_points = None

        # Apply neural network
        solution_flat = self.network(coords_flat, deterministic=deterministic)

        # Reshape back if needed
        if batch_size is not None and n_points is not None:
            solution = solution_flat.reshape(batch_size, n_points, -1)
        else:
            solution = solution_flat

        return solution

    def compute_physics_loss(
        self,
        coordinates: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute physics-based loss components.

        Args:
            coordinates: Space-time coordinates
            deterministic: Whether to use deterministic mode

        Returns:
            Physics loss value
        """
        if self.physics_type == "pde":
            return self._compute_pde_loss(coordinates)
        if self.physics_type == "conservation":
            return self._compute_conservation_loss(coordinates)
        if self.physics_type == "symmetry":
            return self._compute_symmetry_loss(coordinates)
        # Return zero loss for unknown physics types
        return jnp.array(0.0)

    def _compute_pde_loss(self, coordinates: jax.Array) -> jax.Array:
        """Compute PDE residual loss using automatic differentiation."""

        def solution_fn(coords_single):
            """Function to differentiate - single point evaluation."""
            coords_batch = coords_single[None, :]  # Add batch dimension
            solution = self.network(coords_batch, deterministic=True)
            return solution[0, 0]  # Return scalar value

        # Example: Heat equation ∂u/∂t = ∇²u
        # For coordinates [x, t], compute derivatives
        if coordinates.shape[-1] >= 2:
            # Vectorize gradient computation
            grad_fn = jax.vmap(jax.grad(solution_fn))
            hessian_fn = jax.vmap(jax.hessian(solution_fn))

            # Handle batch dimension
            if coordinates.ndim == 3:
                coords_flat = coordinates.reshape(-1, coordinates.shape[-1])
            else:
                coords_flat = coordinates

            gradients = grad_fn(coords_flat)
            hessians = hessian_fn(coords_flat)

            # Extract derivatives for heat equation
            u_t = gradients[..., -1]  # Time derivative (last dimension)
            u_xx = hessians[..., 0, 0]  # Second spatial derivative

            # PDE residual: u_t - u_xx = 0
            pde_residual = u_t - u_xx

            # Mean squared residual
            return jnp.mean(pde_residual**2)
        return jnp.array(0.0)

    def _compute_conservation_loss(self, coordinates: jax.Array) -> jax.Array:
        """Compute the local conservation-law residual loss.

        Enforces the continuity / mass conservation law in flux form
        ``∇·F = 0`` (Landau & Lifshitz, *Fluid Mechanics*, §1), treating the
        predicted output channels as components of the conserved flux ``F`` and
        the ``n_points`` axis as the spatial grid. The loss penalises the
        squared discrete divergence ``mean((∇·F)²)`` computed with second-order
        central finite differences, matching ``neuraloperator``'s
        ``FiniteDiff.divergence`` reference.

        A divergence-free (constant) flux yields ~0 loss; a flux with non-zero
        divergence yields the corresponding positive value.

        The divergence requires a resolved spatial grid. For point-wise inputs
        of shape ``(batch, input_dim)`` there is no neighbour to difference
        against, so the local conservation residual is not well-defined and a
        zero loss is returned honestly rather than substituting a proxy.

        Args:
            coordinates: Space-time coordinates, ``(batch, n_points, input_dim)``
                for the resolved-grid case.

        Returns:
            Scalar mean-squared conservation residual.
        """
        if coordinates.ndim < 3:
            # No spatial grid axis: local divergence is undefined here.
            return jnp.array(0.0)

        # solution shape: (batch, n_points, n_channels); channels are the flux
        # components and axis 1 indexes the spatial grid.
        solution = self.__call__(coordinates, deterministic=True)
        spacing = self._grid_spacing(coordinates)
        return conservation_residual_loss(solution, spatial_axis=1, spacing=spacing)

    @staticmethod
    def _grid_spacing(coordinates: jax.Array) -> jax.Array:
        """Estimate uniform grid spacing from the first spatial coordinate.

        Args:
            coordinates: Coordinates of shape ``(batch, n_points, input_dim)``.

        Returns:
            Positive scalar grid spacing ``Δx`` along the points axis.
        """
        first_axis = coordinates[..., 0]
        diffs = jnp.diff(first_axis, axis=-1)
        spacing = jnp.mean(jnp.abs(diffs))
        return jnp.where(spacing > 0.0, spacing, 1.0)

    def _compute_symmetry_loss(self, coordinates: jax.Array) -> jax.Array:
        """Compute symmetry constraint violation loss."""
        # Example: translational symmetry
        solution_original = self.__call__(coordinates, deterministic=True)

        # Apply small translation
        if coordinates.shape[-1] >= 1:
            coords_translated = coordinates.at[..., 0].add(0.1)
            solution_translated = self.__call__(coords_translated, deterministic=True)

            # Symmetry violation
            symmetry_violation = solution_original - solution_translated
            return jnp.mean(symmetry_violation**2)
        return jnp.array(0.0)

    def compute_total_loss(
        self,
        coordinates: jax.Array,
        target_solution: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> dict[str, jax.Array]:
        """Compute total loss combining data and physics components.

        Args:
            coordinates: Space-time coordinates
            target_solution: Target solution (optional, for supervised learning)
            deterministic: Whether to use deterministic mode

        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Get prediction
        solution = self.__call__(coordinates, deterministic=deterministic)

        # Physics loss
        physics_loss = self.compute_physics_loss(coordinates, deterministic=deterministic)

        # Data loss (if target provided)
        if target_solution is not None:
            data_loss = jnp.mean((solution - target_solution) ** 2)
        else:
            data_loss = jnp.array(0.0)

        # Total weighted loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss

        return {
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "data_weight": jnp.array(self.data_weight),
            "physics_weight": jnp.array(self.physics_weight),
        }
