"""Coarse-to-Fine Multilevel Training Framework.

This module implements multilevel training strategies where models are
trained from coarse (fewer parameters) to fine (more parameters) levels.

Key Features:
    - Network hierarchy with configurable coarsening
    - Transfer operators (prolongation/restriction)
    - Cascade training (sequential level training)

The key insight is that training a coarse network first provides a good
initialization for the fine network, accelerating convergence.

References:
    - Survey Section 8.2: Multilevel Training
    - Multigrid methods in numerical analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


@dataclass(frozen=True)
class MultilevelConfig:
    """Configuration for multilevel training.

    Attributes:
        num_levels: Number of levels in the hierarchy
        coarsening_factor: Factor to reduce width at each coarser level
        level_epochs: Number of epochs to train at each level
        warmup_epochs: Extra epochs at the finest level
    """

    num_levels: int = 3
    coarsening_factor: float = 0.5
    level_epochs: list[int] = field(default_factory=lambda: [100, 200, 300])
    warmup_epochs: int = 0


class MultilevelMLP(nnx.Module):
    """Simple MLP for multilevel hierarchy.

    This is a basic MLP that can be used at each level of the hierarchy.
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
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            rngs: Random number generators
        """
        self.activation = activation

        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

        self.layers = nnx.List(layers)

    def __call__(self, x: Float[Array, ...]) -> Float[Array, "batch out"]:
        """Forward pass."""
        h = x
        for layer in list(self.layers)[:-1]:
            h = layer(h)
            h = self.activation(h)
        return list(self.layers)[-1](h)


def create_network_hierarchy(
    input_dim: int,
    output_dim: int,
    base_hidden_dims: Sequence[int],
    num_levels: int,
    coarsening_factor: float = 0.5,
    *,
    activation: Callable[[Array], Array] = nnx.tanh,
    rngs: nnx.Rngs,
) -> list[MultilevelMLP]:
    """Create hierarchy of networks from coarse to fine.

    The finest level (highest index) uses the base_hidden_dims.
    Coarser levels use progressively smaller networks.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        base_hidden_dims: Hidden dimensions for the finest level
        num_levels: Number of levels in hierarchy
        coarsening_factor: Factor to reduce width at each coarser level
        activation: Activation function
        rngs: Random number generators

    Returns:
        List of networks from coarsest to finest
    """
    hierarchy = []

    for level in range(num_levels):
        # Compute factor for this level (finest = 1.0, coarser = smaller)
        factor = coarsening_factor ** (num_levels - level - 1)

        # Scale hidden dimensions
        scaled_dims = [max(int(d * factor), 1) for d in base_hidden_dims]

        # Create network
        network = MultilevelMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=scaled_dims,
            activation=activation,
            rngs=rngs,
        )
        hierarchy.append(network)

    return hierarchy


def prolongate(
    coarse_model: MultilevelMLP,
    fine_model: MultilevelMLP,
) -> MultilevelMLP:
    """Transfer (prolongate) parameters from coarse to fine model.

    This copies the coarse model parameters to the corresponding subset
    of the fine model parameters. Additional fine model parameters are
    left at their initialized values.

    Args:
        coarse_model: Coarse level model
        fine_model: Fine level model (will be modified in place)

    Returns:
        Fine model with prolongated parameters
    """
    coarse_layers = list(coarse_model.layers)
    fine_layers = list(fine_model.layers)

    layer_pairs = zip(coarse_layers, fine_layers, strict=False)
    for _i, (coarse_layer, fine_layer) in enumerate(layer_pairs):
        # Get coarse parameters
        coarse_kernel = coarse_layer.kernel[...]
        coarse_bias = coarse_layer.bias[...]

        # Get fine layer dimensions
        fine_kernel = fine_layer.kernel[...]
        fine_bias = fine_layer.bias[...]

        # Copy coarse to fine (smaller fits into larger)
        min_in = min(coarse_kernel.shape[0], fine_kernel.shape[0])
        min_out = min(coarse_kernel.shape[1], fine_kernel.shape[1])

        # Update fine kernel
        coarse_slice = coarse_kernel[:min_in, :min_out]
        new_kernel = fine_kernel.at[:min_in, :min_out].set(coarse_slice)
        fine_layer.kernel[...] = new_kernel

        # Update fine bias
        min_bias = min(len(coarse_bias), len(fine_bias))
        new_bias = fine_bias.at[:min_bias].set(coarse_bias[:min_bias])
        fine_layer.bias[...] = new_bias

    return fine_model


def restrict(
    fine_model: MultilevelMLP,
    coarse_model: MultilevelMLP,
) -> MultilevelMLP:
    """Transfer (restrict) parameters from fine to coarse model.

    This copies a subset of the fine model parameters to the coarse model.

    Args:
        fine_model: Fine level model
        coarse_model: Coarse level model (will be modified in place)

    Returns:
        Coarse model with restricted parameters
    """
    fine_layers = list(fine_model.layers)
    coarse_layers = list(coarse_model.layers)

    layer_pairs = zip(fine_layers, coarse_layers, strict=False)
    for _i, (fine_layer, coarse_layer) in enumerate(layer_pairs):
        # Get fine parameters
        fine_kernel = fine_layer.kernel[...]
        fine_bias = fine_layer.bias[...]

        # Get coarse dimensions
        coarse_kernel = coarse_layer.kernel[...]
        coarse_bias = coarse_layer.bias[...]

        # Copy fine to coarse (take subset)
        rows, cols = coarse_kernel.shape[0], coarse_kernel.shape[1]
        coarse_layer.kernel[...] = fine_kernel[:rows, :cols]
        coarse_layer.bias[...] = fine_bias[: len(coarse_bias)]

    return coarse_model


class CascadeTrainer:
    """Cascade trainer for multilevel training.

    Trains models from coarse to fine levels, transferring learned
    parameters between levels.

    Attributes:
        config: Multilevel configuration
        hierarchy: List of models from coarse to fine
        current_level: Current training level

    Example:
        >>> trainer = CascadeTrainer(
        ...     input_dim=1, output_dim=1,
        ...     base_hidden_dims=[64, 64],
        ...     config=MultilevelConfig(num_levels=3),
        ...     rngs=nnx.Rngs(0)
        ... )
        >>> model = trainer.get_current_model()
        >>> # Train model...
        >>> trainer.advance_level()
        >>> finer_model = trainer.get_current_model()
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_hidden_dims: Sequence[int],
        config: MultilevelConfig | None = None,
        *,
        activation: Callable[[Array], Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize cascade trainer.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            base_hidden_dims: Hidden dimensions for finest level
            config: Multilevel configuration
            activation: Activation function
            rngs: Random number generators
        """
        self.config = config or MultilevelConfig()

        self.hierarchy = create_network_hierarchy(
            input_dim=input_dim,
            output_dim=output_dim,
            base_hidden_dims=base_hidden_dims,
            num_levels=self.config.num_levels,
            coarsening_factor=self.config.coarsening_factor,
            activation=activation,
            rngs=rngs,
        )

        self.current_level = 0

    def get_current_model(self) -> MultilevelMLP:
        """Get model at current level.

        Returns:
            Current level model
        """
        return self.hierarchy[self.current_level]

    def advance_level(self) -> bool:
        """Advance to next finer level.

        Transfers learned parameters from current level to next level.

        Returns:
            True if advanced successfully, False if already at finest level
        """
        if self.current_level >= len(self.hierarchy) - 1:
            return False

        # Prolongate parameters to next level
        coarse_model = self.hierarchy[self.current_level]
        fine_model = self.hierarchy[self.current_level + 1]
        prolongate(coarse_model, fine_model)

        self.current_level += 1
        return True

    def is_at_finest(self) -> bool:
        """Check if at finest level.

        Returns:
            True if at finest level
        """
        return self.current_level >= len(self.hierarchy) - 1

    def get_epochs_for_current_level(self) -> int:
        """Get number of epochs for current level.

        Returns:
            Number of epochs to train at current level
        """
        if self.current_level < len(self.config.level_epochs):
            base_epochs = self.config.level_epochs[self.current_level]
        else:
            base_epochs = self.config.level_epochs[-1]

        if self.is_at_finest():
            return base_epochs + self.config.warmup_epochs

        return base_epochs
