"""
Multi-Scale Physics-Informed Neural Networks

This module implements Multi-Scale PINNs for solving PDEs with multiple characteristic
length scales. The architecture combines information from different resolution levels
to capture both coarse and fine-scale physics phenomena.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class MultiScalePINN(nnx.Module):
    """Multi-Scale Physics-Informed Neural Network.

    This architecture processes input at multiple scale levels to capture
    multi-scale physics phenomena. Each scale network captures information
    at different resolutions, and the outputs are combined to form the
    final prediction.

    The architecture is particularly effective for:
    - Fluid dynamics with multiple scales (boundary layers, turbulence)
    - Heat transfer with different thermal scales
    - Electromagnetic phenomena with multiple wavelengths
    - Quantum mechanics with multi-scale wave functions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scales: list[int],
        hidden_dims: list[int],
        *,
        activation: Callable[[Array], Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize Multi-Scale PINN.

        Args:
            input_dim: Input dimensionality (spatial coordinates)
            output_dim: Output dimensionality (solution fields)
            scales: List of scale factors for multi-scale processing
            hidden_dims: Hidden layer dimensions for each scale network
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scales = scales
        self.num_scales = len(scales)
        self.activation = activation

        # Create scale-specific networks
        scale_networks_temp = []
        for _i, _scale in enumerate(scales):
            # Each scale network processes scaled input coordinates
            layers = []

            # Input layer
            layers.append(
                nnx.Linear(
                    in_features=input_dim,
                    out_features=hidden_dims[0],
                    rngs=rngs,
                )
            )

            # Hidden layers
            for j in range(len(hidden_dims) - 1):
                layers.append(
                    nnx.Linear(
                        in_features=hidden_dims[j],
                        out_features=hidden_dims[j + 1],
                        rngs=rngs,
                    )
                )

            # Output layer for this scale
            layers.append(
                nnx.Linear(
                    in_features=hidden_dims[-1],
                    out_features=output_dim,
                    rngs=rngs,
                )
            )

            scale_network = nnx.Sequential(*layers)
            scale_networks_temp.append(scale_network)
            self.scale_networks = nnx.List(scale_networks_temp)

        # Combination weights for multi-scale fusion
        self.scale_weights = nnx.Linear(
            in_features=self.num_scales * output_dim,
            out_features=output_dim,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Apply Multi-Scale PINN to input coordinates.

        Args:
            x: Input coordinates (batch_size, input_dim)

        Returns:
            Multi-scale solution (batch_size, output_dim)
        """
        scale_outputs = []

        for _i, (scale, network) in enumerate(
            zip(self.scales, self.scale_networks, strict=False)
        ):
            # Scale input coordinates for this level
            scaled_x = x * scale

            # Process through scale-specific network
            h = scaled_x
            for _j, layer in enumerate(network.layers[:-1]):
                h = layer(h)
                h = self.activation(h)

            # Output layer (no activation)
            scale_output = network.layers[-1](h)
            scale_outputs.append(scale_output)

        # Concatenate all scale outputs
        combined_output = jnp.concatenate(scale_outputs, axis=-1)

        # Weighted combination of scale outputs
        return self.scale_weights(combined_output)

    def get_scale_outputs(self, x: Array) -> list[Array]:
        """Get outputs from individual scale networks.

        This is useful for analysis and debugging multi-scale behavior.

        Args:
            x: Input coordinates (batch_size, input_dim)

        Returns:
            List of outputs from each scale network
        """
        scale_outputs = []

        for scale, network in zip(self.scales, self.scale_networks, strict=False):
            scaled_x = x * scale

            h = scaled_x
            for layer in network.layers[:-1]:
                h = layer(h)
                h = self.activation(h)

            scale_output = network.layers[-1](h)
            scale_outputs.append(scale_output)

        return scale_outputs

    def compute_derivatives(self, x: Array, order: int = 1) -> dict[str, Array]:
        """Compute derivatives of the multi-scale solution.

        This is essential for physics-informed training where PDE residuals
        require derivative computations.

        Args:
            x: Input coordinates (batch_size, input_dim)
            order: Derivative order (1 for first derivatives, 2 for second)

        Returns:
            Dictionary containing derivative tensors
        """

        def solution_fn(coords):
            return self(coords.reshape(1, -1)).squeeze()

        derivatives = {}

        if order >= 1:
            # First derivatives
            grad_fn = jax.grad(solution_fn)
            if x.ndim == 2:
                # Batch processing
                derivatives["grad"] = jax.vmap(grad_fn)(x)
            else:
                derivatives["grad"] = grad_fn(x)

        if order >= 2:
            # Second derivatives (Laplacian)
            def laplacian_fn(coords):
                grad_fn = jax.grad(solution_fn)
                hessian_fn = jax.jacfwd(grad_fn)
                hessian = hessian_fn(coords)
                return jnp.trace(hessian)  # Laplacian = trace of Hessian

            if x.ndim == 2:
                derivatives["laplacian"] = jax.vmap(laplacian_fn)(x)
            else:
                derivatives["laplacian"] = laplacian_fn(x)

        return derivatives


def create_heat_equation_pinn(
    spatial_dim: int,
    scales: list[int] | None = None,
    hidden_dims: list[int] | None = None,
    *,
    rngs: nnx.Rngs,
) -> MultiScalePINN:
    """Create a Multi-Scale PINN for heat equation problems.

    Args:
        spatial_dim: Spatial dimensionality (1D, 2D, or 3D)
        scales: Scale factors (default: [1, 2, 4] for multi-scale)
        hidden_dims: Hidden layer dimensions (default: [64, 32])
        rngs: Random number generators

    Returns:
        Configured Multi-Scale PINN for heat equation
    """
    if scales is None:
        scales = [1, 2, 4]  # Default multi-scale levels

    if hidden_dims is None:
        hidden_dims = [64, 32]  # Default architecture

    return MultiScalePINN(
        input_dim=spatial_dim + 1,  # spatial coordinates + time
        output_dim=1,  # temperature field
        scales=scales,
        hidden_dims=hidden_dims,
        activation=nnx.tanh,  # Good for PINNs
        rngs=rngs,
    )


class SimplePINN(nnx.Module):
    """Simple MLP-based Physics-Informed Neural Network.

    A standard MLP architecture for solving PDEs. This is the default choice
    for many PINN problems where multi-scale features are not required.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize Simple PINN.

        Args:
            input_dim: Input dimensionality (spatial coordinates, possibly + time)
            output_dim: Output dimensionality (solution fields)
            hidden_dims: Hidden layer dimensions
            activation: Activation function (default: tanh, good for PINNs)
            rngs: Random number generators
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Build layers
        layers = []
        in_features = input_dim
        for h in hidden_dims:
            layers.append(nnx.Linear(in_features, h, rngs=rngs))
            in_features = h
        layers.append(nnx.Linear(in_features, output_dim, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        """Forward pass: x -> u(x).

        Args:
            x: Input coordinates (batch_size, input_dim)

        Returns:
            Solution prediction (batch_size, output_dim)
        """
        h = x
        for layer in list(self.layers)[:-1]:
            h = self.activation(layer(h))
        return list(self.layers)[-1](h)


def create_poisson_pinn(
    spatial_dim: int,
    hidden_dims: list[int] | None = None,
    *,
    rngs: nnx.Rngs,
) -> SimplePINN:
    """Create a Simple PINN for Poisson equation problems.

    The Poisson equation: -∇²u = f(x)

    Args:
        spatial_dim: Spatial dimensionality (1D, 2D, or 3D)
        hidden_dims: Hidden layer dimensions (default: [50, 50, 50])
        rngs: Random number generators

    Returns:
        Configured Simple PINN for Poisson equation
    """
    if hidden_dims is None:
        hidden_dims = [50, 50, 50]  # Default architecture

    return SimplePINN(
        input_dim=spatial_dim,  # spatial coordinates only (steady-state)
        output_dim=1,  # scalar solution field u
        hidden_dims=hidden_dims,
        activation=jnp.tanh,  # Good for PINNs
        rngs=rngs,
    )


def create_navier_stokes_pinn(
    spatial_dim: int,
    scales: list[int] | None = None,
    hidden_dims: list[int] | None = None,
    *,
    rngs: nnx.Rngs,
) -> MultiScalePINN:
    """Create a Multi-Scale PINN for Navier-Stokes equations.

    Args:
        spatial_dim: Spatial dimensionality (2D or 3D)
        scales: Scale factors (default: [1, 2, 4, 8] for turbulence)
        hidden_dims: Hidden layer dimensions (default: [128, 64, 32])
        rngs: Random number generators

    Returns:
        Configured Multi-Scale PINN for Navier-Stokes
    """
    if scales is None:
        scales = [1, 2, 4, 8]  # More scales for turbulent flows

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]  # Larger network for complex flows

    return MultiScalePINN(
        input_dim=spatial_dim + 1,  # spatial coordinates + time
        output_dim=spatial_dim + 1,  # velocity components + pressure
        scales=scales,
        hidden_dims=hidden_dims,
        activation=nnx.tanh,
        rngs=rngs,
    )
