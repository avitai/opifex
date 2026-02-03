"""Gradient-enhanced Self-Training PINN (gST-PINN).

Implements the gST-PINN architecture from:
    "Gradient Enhanced Self-Training Physics-Informed Neural Network"
    (2023 International Conference on Machine Learning and Computer Application)

Key components:
    - Gradient-enhanced loss: adds ||grad(R)||^2 where R is PDE residual
    - Self-training: pseudo-labels from high-confidence (low-residual) predictions

The gradient-enhanced component (gPINN) helps enforce PDE constraints more
strongly by penalizing spatial variation in the residual, which is critical
for problems with shocks, discontinuities, or steep gradients.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


@dataclass(frozen=True)
class GSTConfig:
    """Configuration for gST-PINN.

    Attributes:
        gradient_weight: Weight for the gradient-enhanced term in loss.
            Higher values enforce smoother PDE residuals.
        pseudo_label_threshold: Residual threshold for pseudo-label
            generation. Points with |residual| < threshold are treated
            as reliable predictions for self-training.
        hidden_dims: Default hidden layer dimensions.
    """

    gradient_weight: float = 0.1
    pseudo_label_threshold: float = 0.05
    hidden_dims: tuple[int, ...] = (64, 64, 64)


class GradientEnhancedPINN(nnx.Module):
    """Gradient-enhanced Self-Training Physics-Informed Neural Network.

    Extends SimplePINN with two key mechanisms:

    1. **gPINN loss**: L = MSE(residual) + w * MSE(grad(residual))
       The gradient of the PDE residual w.r.t. spatial coordinates is
       penalized, enforcing that the residual is not only small but also
       spatially smooth.

    2. **Self-training**: High-confidence predictions (where residual is
       below a threshold) generate pseudo-labels for self-supervised
       learning, improving generalization without labeled data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        config: GSTConfig | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize gST-PINN.

        Args:
            input_dim: Input dimensionality (spatial coordinates + time).
            output_dim: Output dimensionality (solution fields).
            hidden_dims: Hidden layer dimensions. Defaults to config.
            activation: Activation function.
            config: gST-PINN configuration.
            rngs: Random number generators.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or GSTConfig()
        self.activation = activation

        dims = list(hidden_dims or self.config.hidden_dims)

        # Build MLP layers
        layers = []
        in_features = input_dim
        for h in dims:
            layers.append(nnx.Linear(in_features, h, rngs=rngs))
            in_features = h
        layers.append(nnx.Linear(in_features, output_dim, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        """Forward pass: x -> u(x).

        Args:
            x: Input coordinates (batch_size, input_dim).

        Returns:
            Solution prediction (batch_size, output_dim).
        """
        h = x
        for layer in list(self.layers)[:-1]:
            h = self.activation(layer(h))
        return list(self.layers)[-1](h)

    def compute_derivatives(self, x: Array, order: int = 1) -> dict[str, Array]:
        """Compute spatial derivatives of the solution.

        Args:
            x: Input coordinates (batch_size, input_dim).
            order: Derivative order (1 for gradient, 2 for Hessian/Laplacian).

        Returns:
            Dictionary with derivative tensors.
        """

        def solution_fn(coords):
            return self(coords[None, :])[0, 0]

        derivatives = {}

        if order >= 1:
            grad_fn = jax.grad(solution_fn)
            derivatives["grad"] = jax.vmap(grad_fn)(x)

        if order >= 2:

            def laplacian_fn(coords):
                hessian = jax.hessian(solution_fn)(coords)
                return jnp.trace(hessian)

            derivatives["laplacian"] = jax.vmap(laplacian_fn)(x)

        return derivatives

    def gradient_enhanced_loss(
        self,
        x: Array,
        pde_residual_fn: Callable[["GradientEnhancedPINN", Array], Array],
    ) -> Array:
        """Compute gPINN loss: MSE(R) + w * MSE(grad_x(R)).

        The gradient-enhanced term penalizes spatial variation in the PDE
        residual, encouraging smoother residual fields.

        Args:
            x: Collocation points (batch_size, input_dim).
            pde_residual_fn: Function(model, x) -> residual (batch_size,).

        Returns:
            Scalar loss value.
        """
        # Standard PDE residual loss
        residual = pde_residual_fn(self, x)
        mse_residual = jnp.mean(residual**2)

        # Gradient-enhanced term: ||d(R)/dx||^2
        def residual_at_point(xi):
            return pde_residual_fn(self, xi[None, :])[0]

        grad_residual = jax.vmap(jax.grad(residual_at_point))(x)
        mse_grad = jnp.mean(jnp.sum(grad_residual**2, axis=-1))

        return mse_residual + self.config.gradient_weight * mse_grad

    def generate_pseudo_labels(
        self,
        x: Array,
        pde_residual_fn: Callable[["GradientEnhancedPINN", Array], Array],
    ) -> tuple[Array, Array]:
        """Generate pseudo-labels from high-confidence predictions.

        Points where the PDE residual is below the threshold are considered
        reliable, and their predictions become pseudo-labels for self-training.

        Args:
            x: Candidate points (batch_size, input_dim).
            pde_residual_fn: Function(model, x) -> residual (batch_size,).

        Returns:
            Tuple of (pseudo_labels, mask):
                - pseudo_labels: Predictions at x (batch_size, output_dim).
                - mask: Boolean mask of reliable points (batch_size,).
        """
        predictions = self(x)
        residual = pde_residual_fn(self, x)
        mask = jnp.abs(residual) < self.config.pseudo_label_threshold
        return predictions, mask

    def self_training_loss(
        self,
        x: Array,
        pde_residual_fn: Callable[["GradientEnhancedPINN", Array], Array],
    ) -> Array:
        """Compute self-training loss using pseudo-labels.

        Uses detached (stop-gradient) predictions as pseudo-ground-truth
        for points where the PDE residual is below threshold.

        Args:
            x: Collocation points (batch_size, input_dim).
            pde_residual_fn: Function(model, x) -> residual (batch_size,).

        Returns:
            Scalar self-training loss.
        """
        pseudo_labels, mask = self.generate_pseudo_labels(x, pde_residual_fn)
        # Detach pseudo-labels (stop gradient)
        pseudo_labels = jax.lax.stop_gradient(pseudo_labels)

        # Current predictions
        current = self(x)

        # MSE only on reliable points
        diff = (current - pseudo_labels) ** 2
        # Mean over output dim, then masked mean over batch
        per_point = jnp.mean(diff, axis=-1)  # (batch,)
        # Use mask as weights; avoid division by zero
        n_reliable = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(per_point * mask) / n_reliable


def create_gst_pinn(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    *,
    config: GSTConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> GradientEnhancedPINN:
    """Create a gST-PINN model.

    Args:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        hidden_dims: Hidden layer dimensions (default: [64, 64, 64]).
        config: gST-PINN configuration.
        activation: Activation function.
        rngs: Random number generators.

    Returns:
        Configured GradientEnhancedPINN instance.
    """
    return GradientEnhancedPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        config=config,
        activation=activation,
        rngs=rngs,
    )
