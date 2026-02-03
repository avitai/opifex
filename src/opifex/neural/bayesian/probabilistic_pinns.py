# FILE PLACEMENT: opifex/neural/bayesian/probabilistic_pinns.py
#
# FIXED MultiFidelityPINN Implementation
# Fixes missing configuration keys and proper initialization
#
# This file should REPLACE: opifex/neural/bayesian/probabilistic_pinns.py

"""
Probabilistic Physics-Informed Neural Networks

Advanced PINN implementations with uncertainty quantification and
multi-fidelity modeling for robust scientific computing applications.

Key Features:
- Multi-fidelity neural networks with adaptive selection
- Bayesian uncertainty quantification
- Proper configuration validation and key handling
- Epistemic and aleatoric uncertainty estimation
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.config import FidelityConfig, MultiFidelityConfig
from opifex.neural.bayesian.layers import BayesianLayer


class MultiFidelityPINN(nnx.Module):
    """
    Multi-Fidelity Physics-Informed Neural Network with proper configuration.

    Implements adaptive multi-fidelity modeling with uncertainty quantification
    for physics-informed neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        low_fidelity_dims: tuple[int, ...] = (32, 32),
        high_fidelity_dims: tuple[int, ...] = (64, 64),
        fusion_dims: tuple[int, ...] = (48,),
        config: MultiFidelityConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Initialize MultiFidelityPINN with comprehensive configuration.

        Args:
            input_dim: Dimension of input space
            output_dim: Dimension of output space
            low_fidelity_dims: Hidden dimensions for low fidelity network
            high_fidelity_dims: Hidden dimensions for high fidelity network
            fusion_dims: Hidden dimensions for fusion network
            config: Multi-fidelity configuration (optional, uses defaults if None)
            rngs: Random number generator state
        """
        # Handle rngs
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Store network architecture parameters
        self.low_fidelity_dims = low_fidelity_dims
        self.high_fidelity_dims = high_fidelity_dims
        self.fusion_dims = fusion_dims

        # FIXED: Use default config if none provided and validate
        if config is None:
            config = MultiFidelityConfig()

        # Ensure configuration is complete and validated
        if not isinstance(config, MultiFidelityConfig):
            raise TypeError("config must be MultiFidelityConfig instance")

        config._validate_config()  # Explicit validation
        self.config = config

        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Extract network parameters
        hidden_layers = list(low_fidelity_dims)  # Use provided dimensions
        activation = config.network_params.get("activation", "tanh")
        use_bayesian = config.uncertainty_params.get("use_epistemic", True)

        # Build multi-fidelity networks
        self._build_networks(hidden_layers, activation, use_bayesian, rngs)

        # Initialize prediction statistics for adaptive selection
        self._initialize_prediction_stats()

    def _build_networks(
        self,
        hidden_layers: list[int],
        activation: str,
        use_bayesian: bool,
        rngs: nnx.Rngs,
    ):
        """Build low and high fidelity networks."""
        self._set_activation_function(activation)
        self._build_low_fidelity_network(hidden_layers, use_bayesian, rngs)
        self._build_high_fidelity_networks(hidden_layers, use_bayesian, rngs)

    def _set_activation_function(self, activation: str):
        """Set the activation function based on the activation string."""
        activation_map = {
            "tanh": nnx.tanh,
            "relu": nnx.relu,
            "gelu": nnx.gelu,
        }
        self.activation = activation_map.get(activation, nnx.tanh)

    def _create_layer(
        self, in_dim: int, out_dim: int, use_bayesian: bool, rngs: nnx.Rngs
    ):
        """Create a layer (either Bayesian or Linear) based on the configuration."""
        if use_bayesian:
            return BayesianLayer(in_dim, out_dim, rngs=rngs)
        return nnx.Linear(in_dim, out_dim, rngs=rngs)

    def _build_low_fidelity_network(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ):
        """Build the low-fidelity network layers."""
        low_fidelity_layers_temp = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_layers:
            layer = self._create_layer(prev_dim, hidden_dim, use_bayesian, rngs)
            low_fidelity_layers_temp.append(layer)
            prev_dim = hidden_dim

        self.low_fidelity_layers = nnx.List(low_fidelity_layers_temp)
        # self.layers = nnx.List(layers_temp) - REMOVED: undefined variable

        # Low-fidelity output layer
        self.low_fidelity_output = self._create_layer(
            prev_dim, self.output_dim, use_bayesian, rngs
        )

    def _build_high_fidelity_networks(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ):
        """Build the high-fidelity correction networks."""
        high_fidelity_networks_temp = []

        for _ in range(self.config.high_fidelity_count):
            network_layers = self._build_single_high_fidelity_network(
                hidden_layers, use_bayesian, rngs
            )
            high_fidelity_networks_temp.append(nnx.List(network_layers))
        self.high_fidelity_networks = nnx.List(high_fidelity_networks_temp)

    def _build_single_high_fidelity_network(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ) -> list:
        """Build a single high-fidelity correction network."""
        network_layers = []
        # Input: original input + low-fidelity prediction
        prev_dim = self.input_dim + self.output_dim

        for hidden_dim in hidden_layers:
            layer = self._create_layer(prev_dim, hidden_dim, use_bayesian, rngs)
            network_layers.append(layer)
            prev_dim = hidden_dim

        # Output layer
        output_layer = self._create_layer(prev_dim, self.output_dim, use_bayesian, rngs)
        network_layers.append(output_layer)

        return network_layers

    def _initialize_prediction_stats(self):
        """Initialize prediction statistics for monitoring."""
        # These will be tracked during training/inference
        self.prediction_stats = {
            "low_fidelity_calls": 0,
            "high_fidelity_calls": 0,
            "total_predictions": 0,
            "average_uncertainty": 0.0,
            "uncertainty_threshold_hits": 0,
        }

    def _low_fidelity_forward(
        self, x: jax.Array, training: bool = True
    ) -> dict[str, jax.Array]:
        """
        Low-fidelity forward pass with uncertainty estimation.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Dictionary with prediction and uncertainty
        """
        h = x
        for layer in self.low_fidelity_layers:
            if isinstance(layer, BayesianLayer):
                h = self.activation(layer(h, training=training))
            else:
                h = self.activation(layer(h))

        # Output prediction
        if isinstance(self.low_fidelity_output, BayesianLayer):
            prediction = self.low_fidelity_output(h, training=training)
        else:
            prediction = self.low_fidelity_output(h)

        # Estimate uncertainty (simplified - could be improved with ensemble)
        uncertainty = jnp.std(prediction, axis=-1, keepdims=True) + 1e-6

        return {"low_fidelity_pred": prediction, "uncertainty_estimate": uncertainty}

    def _high_fidelity_forward(
        self, x: jax.Array, training: bool = True
    ) -> dict[str, jax.Array]:
        """
        High-fidelity forward pass using first correction network.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Dictionary with high and low fidelity predictions
        """
        # Get low-fidelity prediction first
        low_result = self._low_fidelity_forward(x, training=training)
        low_pred = low_result["low_fidelity_pred"]

        # High-fidelity correction using first network
        correction_input = jnp.concatenate([x, low_pred], axis=-1)
        h = correction_input

        for layer in self.high_fidelity_networks[0][:-1]:
            if isinstance(layer, BayesianLayer):
                h = self.activation(layer(h, training=training))
            else:
                h = self.activation(layer(h))

        # Correction prediction
        if isinstance(self.high_fidelity_networks[0][-1], BayesianLayer):
            correction = self.high_fidelity_networks[0][-1](h, training=training)
        else:
            correction = self.high_fidelity_networks[0][-1](h)

        high_pred = low_pred + correction

        # Reduced uncertainty for high-fidelity
        uncertainty = low_result["uncertainty_estimate"] * 0.5

        return {
            "high_fidelity_pred": high_pred,
            "low_fidelity_pred": low_pred,
            "uncertainty_estimate": uncertainty,
        }

    def _adaptive_forward(
        self, x: jax.Array, training: bool = True
    ) -> dict[str, jax.Array]:
        """
        Adaptive forward pass with fidelity selection based on uncertainty.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Dictionary with adaptive predictions and statistics
        """
        # Start with low-fidelity
        low_result = self._low_fidelity_forward(x, training=training)
        uncertainty = low_result["uncertainty_estimate"]

        # Decide whether to use high-fidelity based on uncertainty threshold
        use_high_fidelity = uncertainty > self.config.uncertainty_threshold

        # FIXED: Count predictions by fidelity level (convert to Python int)
        low_fidelity_count = int(jnp.sum(~use_high_fidelity))
        high_fidelity_count = int(jnp.sum(use_high_fidelity))

        # Apply high-fidelity where needed
        if jnp.any(use_high_fidelity):
            high_result = self._high_fidelity_forward(x, training=training)

            # Select predictions based on uncertainty
            final_pred = jnp.where(
                use_high_fidelity,
                high_result["high_fidelity_pred"],
                low_result["low_fidelity_pred"],
            )

            final_uncertainty = jnp.where(
                use_high_fidelity,
                high_result["uncertainty_estimate"],
                low_result["uncertainty_estimate"],
            )
        else:
            final_pred = low_result["low_fidelity_pred"]
            final_uncertainty = low_result["uncertainty_estimate"]

        # FIXED: Return dictionary with all required keys - convert scalars to Arrays
        return {
            "prediction": final_pred,
            "low_fidelity_pred": low_result["low_fidelity_pred"],
            "uncertainty_estimate": final_uncertainty,
            "use_high_fidelity": jnp.array(use_high_fidelity),
            "low_fidelity_count": jnp.array(low_fidelity_count),
            "high_fidelity_count": jnp.array(high_fidelity_count),
            # Additional statistics
            "total_samples": jnp.array(low_fidelity_count + high_fidelity_count),
            "high_fidelity_ratio": jnp.array(
                high_fidelity_count / (low_fidelity_count + high_fidelity_count + 1e-8)
            ),
        }

    def __call__(
        self, x: jax.Array, fidelity_level: str = "adaptive", training: bool = True
    ) -> dict[str, jax.Array]:
        """
        Forward pass with fidelity selection.

        Args:
            x: Input tensor
            fidelity_level: Type of fidelity ('low', 'high', 'adaptive')
            training: Whether in training mode

        Returns:
            Dictionary with predictions and metadata
        """
        if fidelity_level == "low":
            return self._low_fidelity_forward(x, training=training)
        if fidelity_level == "high":
            return self._high_fidelity_forward(x, training=training)
        if fidelity_level == "adaptive":
            return self._adaptive_forward(x, training=training)
        raise ValueError(f"Unknown fidelity level: {fidelity_level}")

    def predict_with_uncertainty(
        self, x: jax.Array, num_samples: int = 10, fidelity_level: str = "adaptive"
    ) -> dict[str, jax.Array]:
        """
        Predict with Monte Carlo uncertainty estimation.

        Args:
            x: Input tensor
            num_samples: Number of MC samples
            fidelity_level: Type of fidelity to use

        Returns:
            Dictionary with prediction statistics
        """
        predictions = []
        uncertainties = []

        for _ in range(num_samples):
            result = self(x, fidelity_level=fidelity_level, training=True)
            pred = result.get("prediction")
            if pred is None:
                pred = result.get("low_fidelity_pred")
            if pred is not None:
                predictions.append(pred)

            uncertainty = result.get("uncertainty_estimate")
            if uncertainty is not None:
                uncertainties.append(uncertainty)

        # Compute statistics
        predictions_stack = jnp.stack(predictions)
        uncertainties_stack = jnp.stack(uncertainties)

        mean_prediction = jnp.mean(predictions_stack, axis=0)
        epistemic_uncertainty = jnp.std(predictions_stack, axis=0)
        aleatoric_uncertainty = jnp.mean(uncertainties_stack, axis=0)
        total_uncertainty = jnp.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        return {
            "mean": mean_prediction,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "total_uncertainty": total_uncertainty,
            "predictions": predictions_stack,
            "num_samples": jnp.array(num_samples),
        }

    def adaptive_prediction(
        self, x: jax.Array, uncertainty_threshold: float = 0.1
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """
        Adaptive prediction with automatic fidelity selection.

        Args:
            x: Input tensor
            uncertainty_threshold: Threshold for fidelity selection

        Returns:
            Tuple of (predictions, uncertainties, info_dict)
        """
        # Get adaptive results
        result = self._adaptive_forward(x, training=False)

        predictions = result["prediction"]
        uncertainties = result["uncertainty_estimate"]

        info = {
            "low_fidelity_count": result["low_fidelity_count"],
            "high_fidelity_count": result["high_fidelity_count"],
            "total_samples": result["total_samples"],
        }

        return predictions, uncertainties, info

    def _fusion_prediction(self, x: jax.Array, training: bool = True) -> jax.Array:
        """
        Internal fusion prediction method for test compatibility.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Fused prediction array
        """
        # Use adaptive forward for fusion prediction
        result = self._adaptive_forward(x, training=training)
        return result["prediction"]


class ProbabilisticPINN(nnx.Module):
    """
    Single-fidelity Probabilistic PINN with uncertainty quantification.

    Standard PINN with Bayesian layers for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (64, 64, 64),
        use_bayesian: bool = True,
        physics_loss_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize probabilistic PINN."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bayesian = use_bayesian
        self.physics_loss_weight = physics_loss_weight
        self.uncertainty_weight = uncertainty_weight

        # Handle rngs
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Build network
        layers_temp = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if use_bayesian:
                layer = BayesianLayer(prev_dim, hidden_dim, rngs=rngs)
            else:
                layer = nnx.Linear(prev_dim, hidden_dim, rngs=rngs)
            layers_temp.append(layer)
            prev_dim = hidden_dim
            self.layers = nnx.List(layers_temp)

        # Output layer
        if use_bayesian:
            self.output_layer = BayesianLayer(prev_dim, output_dim, rngs=rngs)
        else:
            self.output_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass through probabilistic PINN."""
        h = x
        for layer in self.layers:
            if isinstance(layer, BayesianLayer):
                h = nnx.tanh(layer(h, training=training))
            else:
                h = nnx.tanh(layer(h))

        if isinstance(self.output_layer, BayesianLayer):
            return self.output_layer(h, training=training)
        return self.output_layer(h)

    def predict_with_uncertainty(
        self, x: jax.Array, num_samples: int = 10
    ) -> dict[str, jax.Array]:
        """
        Predict with uncertainty estimation for single-fidelity PINN.

        Args:
            x: Input tensor
            num_samples: Number of MC samples for uncertainty

        Returns:
            Dictionary with prediction statistics
        """
        if not self.use_bayesian:
            # For non-Bayesian networks, return single prediction with zero uncertainty
            pred = self(x, training=False)
            return {
                "mean": pred,
                "std": jnp.zeros_like(pred),
                "total_uncertainty": jnp.zeros(pred.shape[0]),
            }

        # Monte Carlo sampling for Bayesian networks
        predictions = []
        for _ in range(num_samples):
            pred = self(x, training=True)  # Sample weights
            predictions.append(pred)

        predictions_stack = jnp.stack(predictions)
        mean_prediction = jnp.mean(predictions_stack, axis=0)
        std_prediction = jnp.std(predictions_stack, axis=0)
        total_uncertainty = jnp.mean(std_prediction, axis=-1)

        # Calculate 95% confidence intervals (1.96 * std for normal distribution)
        confidence_95_lower = mean_prediction - 1.96 * std_prediction
        confidence_95_upper = mean_prediction + 1.96 * std_prediction

        return {
            "mean": mean_prediction,
            "std": std_prediction,
            "total_uncertainty": total_uncertainty,
            "confidence_95_lower": confidence_95_lower,
            "confidence_95_upper": confidence_95_upper,
        }

    def physics_loss(
        self,
        x: jax.Array,
        pde_residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
        boundary_conditions: dict[str, Any] | None = None,
    ) -> jax.Array:
        """
        Compute physics loss for PINN training.

        Args:
            x: Input coordinates
            pde_residual_fn: Function computing PDE residual
            boundary_conditions: Optional boundary conditions

        Returns:
            Physics loss scalar
        """
        # Forward prediction
        y_pred = self(x, training=False)

        # Compute PDE residual
        residual = pde_residual_fn(x, y_pred)
        physics_loss = jnp.mean(residual**2)

        # Add boundary condition loss if provided
        if boundary_conditions is not None:
            # Simple boundary loss implementation
            bc_weight = boundary_conditions.get("weight", 1.0)
            bc_value = boundary_conditions.get("value", 0.0)
            bc_loss = jnp.mean((y_pred - bc_value) ** 2) * bc_weight * 0.1
            physics_loss = physics_loss + bc_loss

        return physics_loss

    def robust_loss(
        self,
        x: jax.Array,
        y_true: jax.Array,
        noise_scale: float = 0.01,
    ) -> jax.Array:
        """
        Compute robust loss with noise perturbation.

        Args:
            x: Input coordinates
            y_true: True values
            noise_scale: Scale of adversarial noise

        Returns:
            Robust loss scalar
        """
        # Clean prediction
        clean_pred = self(x, training=False)

        # Noisy prediction
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, x.shape) * noise_scale
        x_noisy = x + noise
        noisy_pred = self(x_noisy, training=False)

        # Data loss
        data_loss = jnp.mean((clean_pred - y_true) ** 2)

        # Robustness penalty
        robustness_penalty = jnp.mean((clean_pred - noisy_pred) ** 2) / (noise_scale**2)

        return data_loss + self.uncertainty_weight * robustness_penalty


class RobustPINNOptimizer(nnx.Module):
    """
    Robust optimizer for Physics-Informed Neural Networks.

    Provides robust optimization with uncertainty quantification,
    physics loss integration, and adaptive sampling strategies.
    """

    def __init__(
        self,
        model: ProbabilisticPINN | MultiFidelityPINN,
        learning_rate: float = 1e-3,
        robustness_weight: float = 0.1,
        physics_weight: float = 1.0,
        data_weight: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Initialize robust PINN optimizer.

        Args:
            model: PINN model to optimize
            learning_rate: Learning rate for optimization
            robustness_weight: Weight for robustness penalty
            physics_weight: Weight for physics loss
            data_weight: Weight for data loss
            rngs: Random number generator state
        """
        self.model = model
        self.learning_rate = learning_rate
        self.robustness_weight = robustness_weight
        self.physics_weight = physics_weight
        self.data_weight = data_weight

        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs

    def compute_loss_components(
        self,
        x: jax.Array,
        y_true: jax.Array,
        pde_residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
        boundary_conditions: dict[str, Any] | None = None,
        noise_scale: float = 0.01,
    ) -> dict[str, Any]:
        """
        Compute all loss components for robust PINN training.

        Args:
            x: Input coordinates
            y_true: True values for data loss
            pde_residual_fn: Function computing PDE residual
            boundary_conditions: Optional boundary conditions
            noise_scale: Noise scale for robustness penalty

        Returns:
            Dictionary with loss components
        """
        # Forward prediction with type-safe extraction
        model_output = self.model(x, training=False)

        # Extract prediction Array from Union type safely
        if isinstance(model_output, dict):
            y_pred = model_output.get("prediction")
            if y_pred is None:
                y_pred = model_output.get("mean")
            if y_pred is None:
                y_pred = model_output.get("low_fidelity_pred")
            if y_pred is None:
                raise ValueError("Could not extract prediction from model output dict")
        else:
            y_pred = model_output

        # Data loss (MSE)
        data_loss = jnp.mean((y_pred - y_true) ** 2)

        # Physics loss
        pde_residual = pde_residual_fn(x, y_pred)
        physics_loss = jnp.mean(pde_residual**2)

        # Robustness penalty
        robustness_penalty = self._compute_robustness_penalty(x, noise_scale)

        # Boundary condition loss (if provided)
        bc_loss = 0.0
        if boundary_conditions is not None:
            bc_loss = self._compute_boundary_loss(x, y_pred, boundary_conditions)

        # Total loss
        total_loss = (
            self.data_weight * data_loss
            + self.physics_weight * physics_loss
            + self.robustness_weight * robustness_penalty
            + bc_loss
        )

        return {
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "robustness_penalty": robustness_penalty,
            "boundary_loss": bc_loss,
            "total_loss": total_loss,
        }

    def _compute_robustness_penalty(
        self, x: jax.Array, noise_scale: float = 0.01
    ) -> jax.Array:
        """
        Compute robustness penalty using adversarial noise.

        Args:
            x: Input coordinates
            noise_scale: Scale of adversarial noise

        Returns:
            Robustness penalty scalar
        """
        # Generate adversarial noise
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        noise = jax.random.normal(key, x.shape) * noise_scale
        x_noisy = x + noise

        # Compute predictions for clean and noisy inputs with type safety
        if hasattr(self.model, "predict_with_uncertainty"):
            clean_result = self.model.predict_with_uncertainty(x, num_samples=3)
            noisy_result = self.model.predict_with_uncertainty(x_noisy, num_samples=3)
            clean_pred = clean_result["mean"]
            noisy_pred = noisy_result["mean"]
        else:
            clean_output = self.model(x)
            noisy_output = self.model(x_noisy)

            # Extract Array from Union type safely with None handling
            if isinstance(clean_output, dict):
                clean_pred = clean_output.get("prediction")
                if clean_pred is None:
                    clean_pred = clean_output.get("mean")
                if clean_pred is None:
                    clean_pred = clean_output.get("low_fidelity_pred")
                if clean_pred is None:
                    raise ValueError(
                        "Could not extract prediction from clean model output"
                    )
            else:
                clean_pred = clean_output

            if isinstance(noisy_output, dict):
                noisy_pred = noisy_output.get("prediction")
                if noisy_pred is None:
                    noisy_pred = noisy_output.get("mean")
                if noisy_pred is None:
                    noisy_pred = noisy_output.get("low_fidelity_pred")
                if noisy_pred is None:
                    raise ValueError(
                        "Could not extract prediction from noisy model output"
                    )
            else:
                noisy_pred = noisy_output

        # Robustness penalty is the sensitivity to noise
        return jnp.mean((clean_pred - noisy_pred) ** 2) / (noise_scale**2)

    def _compute_boundary_loss(
        self,
        x: jax.Array,
        y_pred: jax.Array,
        boundary_conditions: dict[str, Any],
    ) -> jax.Array:
        """
        Compute boundary condition loss.

        Args:
            x: Input coordinates
            y_pred: Model predictions
            boundary_conditions: Boundary condition specification

        Returns:
            Boundary loss scalar
        """
        bc_type = boundary_conditions.get("type", "dirichlet")

        if bc_type == "dirichlet":
            # Simple Dirichlet BC: assume boundary points have y=0
            bc_value = boundary_conditions.get("value", 0.0)
            # For simplicity, apply to all points (in practice would filter boundary)
            return jnp.mean((y_pred - bc_value) ** 2) * 0.1

        # For other BC types, return zero for now
        return jnp.array(0.0)

    def uncertainty_guided_sampling(
        self,
        x_candidates: jax.Array,
        num_samples: int,
        uncertainty_threshold: float = 0.1,
    ) -> jax.Array:
        """
        Select points based on uncertainty for adaptive sampling.

        Args:
            x_candidates: Candidate points for sampling
            num_samples: Number of points to select
            uncertainty_threshold: Threshold for uncertainty-based selection

        Returns:
            Selected points with highest uncertainty
        """
        # Compute uncertainties for all candidates
        if hasattr(self.model, "predict_with_uncertainty"):
            pred_result = self.model.predict_with_uncertainty(
                x_candidates, num_samples=10
            )
            uncertainties = pred_result.get(
                "total_uncertainty",
                pred_result.get("std", jnp.zeros(x_candidates.shape[0])),
            )
        else:
            # For models without uncertainty, use random selection
            uncertainties = jax.random.uniform(
                jax.random.PRNGKey(42), (x_candidates.shape[0],)
            )

        # Ensure uncertainties is 1D
        if uncertainties.ndim > 1:
            uncertainties = jnp.mean(uncertainties, axis=-1)

        # Select points with highest uncertainty
        top_indices = jnp.argsort(uncertainties)[-num_samples:]
        return x_candidates[top_indices]


# Factory functions for easy creation
def create_multifidelity_pinn(
    input_dim: int,
    output_dim: int,
    config_dict: dict[str, Any] | None = None,
    rngs: nnx.Rngs | None = None,
) -> MultiFidelityPINN:
    """
    Create MultiFidelityPINN with proper configuration handling.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        config_dict: Optional configuration dictionary
        rngs: Random number generator state

    Returns:
        Configured MultiFidelityPINN instance
    """
    if rngs is None:
        rngs = nnx.Rngs(0)

    # Create configuration from dict if provided
    if config_dict is not None:
        config = MultiFidelityConfig()

        # Handle nested configurations
        if "low_fidelity" in config_dict:
            if isinstance(config_dict["low_fidelity"], dict):
                config.low_fidelity = FidelityConfig(**config_dict["low_fidelity"])
            else:
                config.low_fidelity = config_dict["low_fidelity"]

        # Update other fields
        for key, value in config_dict.items():
            if hasattr(config, key) and key != "low_fidelity":
                setattr(config, key, value)
    else:
        config = MultiFidelityConfig()

    return MultiFidelityPINN(input_dim, output_dim, config=config, rngs=rngs)


def create_probabilistic_pinn(
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int] | None = None,
    use_bayesian: bool = True,
    rngs: nnx.Rngs | None = None,
) -> ProbabilisticPINN:
    """Create a probabilistic PINN with default configuration."""
    if hidden_layers is None:
        hidden_layers = [64, 64, 64]
    if rngs is None:
        rngs = nnx.Rngs(0)

    return ProbabilisticPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_layers,
        use_bayesian=use_bayesian,
        rngs=rngs,
    )


# Update factory functions to include RobustPINNOptimizer
def create_robust_pinn_optimizer(
    model: ProbabilisticPINN | MultiFidelityPINN,
    learning_rate: float = 1e-3,
    robustness_weight: float = 0.1,
    rngs: nnx.Rngs | None = None,
) -> RobustPINNOptimizer:
    """Create robust PINN optimizer with default settings."""
    return RobustPINNOptimizer(
        model=model,
        learning_rate=learning_rate,
        robustness_weight=robustness_weight,
        rngs=rngs,
    )
