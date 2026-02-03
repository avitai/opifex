"""Calibration tools for uncertainty quantification and model calibration.

This module provides tools for assessing and improving the calibration of
uncertainty estimates from probabilistic models, including temperature scaling
and reliability diagram computation.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class TemperatureScaling(nnx.Module):
    """Temperature scaling for uncertainty calibration.

    Applies learnable temperature scaling to improve calibration of
    probabilistic predictions while respecting physics constraints.
    """

    def __init__(
        self,
        physics_constraints: Sequence[str] = (),
        adaptive: bool = False,
        learning_rate: float = 0.01,
        constraint_strength: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize temperature scaling.

        Args:
            physics_constraints: List of physics constraints to enforce
            adaptive: Whether to use adaptive temperature learning
            learning_rate: Learning rate for temperature optimization
            constraint_strength: Strength of physics constraint enforcement
            rngs: Random number generators
        """
        super().__init__()

        self.physics_constraints = physics_constraints
        self.adaptive = adaptive
        self.learning_rate = learning_rate
        self.constraint_strength = constraint_strength

        # Initialize temperature parameter
        self.temperature = nnx.Param(nnx.initializers.constant(1.0)(rngs.params(), ()))

        # Initialize adaptive parameters if needed
        if adaptive:
            self.momentum = nnx.Param(nnx.initializers.constant(0.9)(rngs.params(), ()))
            self.velocity = nnx.Variable(jnp.array(0.0))

        # Initialize physics constraint tracking
        self.constraint_penalty_history: list[float] = []

    def __call__(
        self, predictions: jax.Array, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Apply temperature scaling to predictions.

        Args:
            predictions: Model predictions
            inputs: Input data for context

        Returns:
            Tuple of (calibrated_predictions, aleatoric_uncertainty)
        """
        # Apply temperature scaling
        calibrated_predictions = predictions / self.temperature[...]

        # Estimate aleatoric uncertainty (data-dependent)
        # Enhanced approach: use input complexity as uncertainty proxy
        input_magnitude = jnp.linalg.norm(inputs, axis=-1, keepdims=True)
        input_variance = jnp.var(inputs, axis=-1, keepdims=True)

        # Combine magnitude and variance for better uncertainty estimation
        aleatoric_uncertainty = (
            0.1 * (input_magnitude + input_variance) / (1.0 + input_magnitude)
        )

        return calibrated_predictions, aleatoric_uncertainty

    def apply_physics_aware_calibration(
        self, predictions: jax.Array, inputs: jax.Array
    ) -> tuple[jax.Array, float]:
        """Apply physics-aware temperature scaling with constraint enforcement.

        Args:
            predictions: Model predictions to calibrate
            inputs: Input data for constraint evaluation

        Returns:
            Tuple of (calibrated_predictions, physics_constraint_penalty)
        """
        # Apply base temperature scaling
        calibrated_predictions = predictions / self.temperature[...]

        # Apply physics constraint enforcement
        calibrated_predictions, constraint_penalty = self._enforce_physics_constraints(
            calibrated_predictions, inputs
        )

        # Track constraint penalty for adaptive learning
        self.constraint_penalty_history.append(float(constraint_penalty))
        if len(self.constraint_penalty_history) > 100:  # Keep history bounded
            self.constraint_penalty_history.pop(0)

        return calibrated_predictions, float(constraint_penalty)

    def _enforce_physics_constraints(
        self, predictions: jax.Array, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Enforce physics constraints on predictions.

        Args:
            predictions: Model predictions
            inputs: Input data for context

        Returns:
            Tuple of (constrained_predictions, constraint_penalty)
        """
        constrained_predictions = predictions
        total_penalty = jnp.array(0.0)

        for constraint in self.physics_constraints:
            constrained_predictions, penalty = self._apply_single_constraint(
                constrained_predictions, inputs, constraint
            )
            total_penalty += penalty * self.constraint_strength

        return constrained_predictions, total_penalty

    def _apply_single_constraint(
        self, predictions: jax.Array, inputs: jax.Array, constraint: str
    ) -> tuple[jax.Array, jax.Array]:
        """Apply a single physics constraint.

        Args:
            predictions: Model predictions
            inputs: Input data
            constraint: Type of constraint to apply

        Returns:
            Tuple of (constrained_predictions, constraint_penalty)
        """
        if constraint == "energy_conservation":
            # Energy should be non-negative
            violations = jnp.maximum(0.0, -predictions)
            penalty = jnp.mean(violations**2)
            constrained_predictions = jnp.maximum(predictions, 0.0)

        elif constraint == "mass_conservation":
            # Mass should be conserved (sum close to initial)
            total_mass = jnp.sum(predictions, axis=-1, keepdims=True)
            target_mass = jnp.sum(jnp.abs(inputs), axis=-1, keepdims=True)
            mass_violation = jnp.abs(total_mass - target_mass)
            penalty = jnp.mean(mass_violation**2)
            # Scale predictions to conserve mass
            constrained_predictions = predictions * (target_mass / (total_mass + 1e-8))

        elif constraint == "positivity":
            # Values should be positive
            violations = jnp.maximum(0.0, -predictions)
            penalty = jnp.mean(violations**2)
            constrained_predictions = jnp.maximum(predictions, 1e-8)

        elif constraint == "boundedness":
            # Values should be bounded (clamp to reasonable range)
            lower_bound, upper_bound = -10.0, 10.0
            violations = jnp.maximum(0.0, lower_bound - predictions) + jnp.maximum(
                0.0, predictions - upper_bound
            )
            penalty = jnp.mean(violations**2)
            constrained_predictions = jnp.clip(predictions, lower_bound, upper_bound)

        else:
            # Unknown constraint - no modification
            constrained_predictions = predictions
            penalty = jnp.array(0.0)

        return constrained_predictions, penalty

    def optimize_temperature(self, logits: jax.Array, labels: jax.Array) -> float:
        """Optimize temperature parameter for calibration.

        Args:
            logits: Model logits for validation data
            labels: True labels for validation data

        Returns:
            Optimized temperature value
        """

        def calibration_loss(temp):
            """Negative log-likelihood loss for temperature optimization."""
            scaled_logits = logits / temp

            # Handle both binary and multi-class cases
            if len(scaled_logits.shape) == 1 or scaled_logits.shape[-1] == 1:
                # Binary classification case
                probs = jax.nn.sigmoid(scaled_logits.squeeze())
                probs = jnp.clip(probs, 1e-8, 1.0 - 1e-8)
                return -jnp.mean(
                    labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs)
                )
            # Multi-class classification case
            log_probs = jax.nn.log_softmax(scaled_logits, axis=-1)
            return -jnp.mean(log_probs[jnp.arange(len(labels)), labels])

        # Enhanced optimization with momentum for adaptive mode
        current_temp = self.temperature[...]

        if self.adaptive:
            # Use momentum-based optimization
            for _ in range(200):  # More iterations for better convergence
                grad = jax.grad(calibration_loss)(current_temp)

                # Update velocity with momentum
                self.velocity[...] = self.momentum[...] * self.velocity[...] + grad

                # Update temperature
                current_temp = current_temp - self.learning_rate * self.velocity[...]
                current_temp = jnp.maximum(current_temp, 0.01)  # Ensure positive
        else:
            # Simple gradient-based optimization
            for _ in range(100):  # Fixed number of optimization steps
                grad = jax.grad(calibration_loss)(current_temp)
                current_temp = current_temp - self.learning_rate * grad
                current_temp = jnp.maximum(current_temp, 0.01)  # Ensure positive

        return float(current_temp)

    def optimize_temperature_with_physics_constraints(
        self, predictions: jax.Array, targets: jax.Array, inputs: jax.Array
    ) -> float:
        """Optimize temperature parameter with physics constraint awareness.

        Args:
            predictions: Model predictions
            targets: Target values
            inputs: Input data for constraint evaluation

        Returns:
            Optimized temperature value
        """

        def physics_aware_loss(temp):
            """Loss function that includes physics constraints."""
            # Scale predictions by temperature
            scaled_predictions = predictions / temp

            # Apply physics constraints
            constrained_predictions, constraint_penalty = (
                self._enforce_physics_constraints(scaled_predictions, inputs)
            )

            # Base calibration loss (MSE)
            calibration_loss = jnp.mean((constrained_predictions - targets) ** 2)

            # Combined loss with physics penalty
            return (
                calibration_loss + self.constraint_strength * constraint_penalty,
                constraint_penalty,
            )

        # Optimize temperature
        current_temp = self.temperature[...]

        if self.adaptive:
            # Use momentum-based optimization with physics constraints
            for _ in range(150):  # More iterations for physics-aware optimization
                _loss_value, constraint_penalty = physics_aware_loss(current_temp)
                grad = jax.grad(lambda t: physics_aware_loss(t)[0])(current_temp)

                # Store constraint penalty in history
                self.constraint_penalty_history.append(float(constraint_penalty))

                # Update velocity with momentum
                self.velocity[...] = self.momentum[...] * self.velocity[...] + grad

                # Update temperature
                current_temp = current_temp - self.learning_rate * self.velocity[...]
                current_temp = jnp.maximum(current_temp, 0.01)  # Ensure positive
        else:
            # Simple gradient-based optimization with physics constraints
            for _ in range(100):
                _loss_value, constraint_penalty = physics_aware_loss(current_temp)
                grad = jax.grad(lambda t: physics_aware_loss(t)[0])(current_temp)

                # Store constraint penalty in history
                self.constraint_penalty_history.append(float(constraint_penalty))

                current_temp = current_temp - self.learning_rate * grad
                current_temp = jnp.maximum(current_temp, 0.01)  # Ensure positive

        # Keep history bounded
        if len(self.constraint_penalty_history) > 100:
            self.constraint_penalty_history = self.constraint_penalty_history[-100:]

        return float(current_temp)

    def adaptive_temperature_scaling(
        self, predictions: jax.Array, uncertainties: jax.Array, true_values: jax.Array
    ) -> jax.Array:
        """Apply adaptive temperature scaling based on uncertainty quality.

        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            true_values: Ground truth values

        Returns:
            Adaptively calibrated temperatures
        """
        # Compute prediction errors
        errors = jnp.abs(predictions - true_values)

        # Compute calibration quality (how well uncertainty predicts error)
        # Higher correlation = better calibration = lower temperature adjustment
        correlation = jnp.corrcoef(uncertainties.flatten(), errors.flatten())[0, 1]

        # Adaptive temperature based on calibration quality
        base_temp = self.temperature[...]
        adaptation_factor = 1.0 + 0.5 * (1.0 - jnp.abs(correlation))

        return base_temp * adaptation_factor


class PlattScaling(nnx.Module):
    """Platt scaling for probabilistic calibration.

    Applies a sigmoid function to logits to improve calibration of
    binary classification problems.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize Platt scaling parameters.

        Args:
            rngs: Random number generators
        """
        super().__init__()

        # Learnable scaling parameters for sigmoid: P(y=1|f) = 1/(1+exp(A*f+B))
        self.a = nnx.Param(nnx.initializers.constant(-1.0)(rngs.params(), ()))
        self.b = nnx.Param(nnx.initializers.constant(0.0)(rngs.params(), ()))

    def __call__(self, logits: jax.Array) -> jax.Array:
        """Apply Platt scaling to logits.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        return jax.nn.sigmoid(self.a[...] * logits + self.b[...])

    def fit(
        self, logits: jax.Array, labels: jax.Array, max_iterations: int = 100
    ) -> None:
        """Fit Platt scaling parameters using maximum likelihood.

        Args:
            logits: Training logits
            labels: Binary labels (0 or 1)
            max_iterations: Maximum number of optimization iterations
        """

        def loss_fn(a_param, b_param):
            """Binary cross-entropy loss for Platt scaling."""
            probs = jax.nn.sigmoid(a_param * logits + b_param)
            # Clip probabilities to avoid log(0)
            probs = jnp.clip(probs, 1e-8, 1.0 - 1e-8)
            return -jnp.mean(
                labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs)
            )

        # Optimize A and B parameters
        current_A, current_B = self.a[...], self.b[...]
        learning_rate = 0.01

        for _ in range(max_iterations):
            grads = jax.grad(loss_fn, argnums=(0, 1))(current_A, current_B)
            current_A -= learning_rate * grads[0]
            current_B -= learning_rate * grads[1]

        # Update parameters
        self.a = nnx.Param(current_A)
        self.b = nnx.Param(current_B)


class IsotonicRegression(nnx.Module):
    """Isotonic regression for calibration.

    Non-parametric calibration method that learns a monotonic mapping
    from confidence scores to calibrated probabilities.
    """

    def __init__(self, n_bins: int = 100, *, rngs: nnx.Rngs):
        """Initialize isotonic regression.

        Args:
            n_bins: Number of bins for isotonic regression
            rngs: Random number generators
        """
        super().__init__()
        self.n_bins = n_bins

        # Calibration mapping (confidence -> probability)
        self.calibration_map = nnx.Variable(jnp.linspace(0.0, 1.0, n_bins))
        self.bin_edges = nnx.Variable(jnp.linspace(0.0, 1.0, n_bins + 1))

    def __call__(self, confidences: jax.Array) -> jax.Array:
        """Apply isotonic calibration to confidence scores.

        Args:
            confidences: Input confidence scores

        Returns:
            Calibrated probabilities
        """
        # Find bin indices for each confidence
        bin_indices = jnp.searchsorted(self.bin_edges[...][:-1], confidences)
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        # Apply calibration mapping
        return self.calibration_map[...][bin_indices]

    def fit(self, confidences: jax.Array, labels: jax.Array) -> None:
        """Fit isotonic regression using pool adjacent violators algorithm.

        Args:
            confidences: Training confidence scores
            labels: Binary labels (0 or 1)
        """
        # Sort by confidence
        sorted_indices = jnp.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Bin the data
        bin_indices = (
            jnp.searchsorted(self.bin_edges[...][:-1], sorted_confidences, side="right")
            - 1
        )
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        # Compute bin averages
        calibration_values = []
        for i in range(self.n_bins):
            mask = bin_indices == i
            if jnp.sum(mask) > 0:
                bin_accuracy = jnp.mean(sorted_labels[mask])
            else:
                # Use linear interpolation for empty bins
                bin_accuracy = i / (self.n_bins - 1)
            calibration_values.append(bin_accuracy)

        calibration_array = jnp.array(calibration_values)

        # Apply pool adjacent violators to ensure monotonicity
        calibrated_values = self._pool_adjacent_violators(calibration_array)

        # Update calibration map
        self.calibration_map[...] = calibrated_values

    def _pool_adjacent_violators(self, y: jax.Array) -> jax.Array:
        """Pool adjacent violators algorithm for isotonic regression.

        Args:
            y: Input values to make monotonic

        Returns:
            Monotonic version of input values
        """
        # Simple implementation of PAV algorithm
        n = len(y)
        result = jnp.copy(y)

        # Find violations and pool adjacent values
        for i in range(n - 1):
            if result[i] > result[i + 1]:
                # Pool values to maintain monotonicity
                pooled_value = (result[i] + result[i + 1]) / 2
                result = result.at[i].set(pooled_value)
                result = result.at[i + 1].set(pooled_value)

        return result


class ConformalPrediction(nnx.Module):
    """Conformal prediction for calibrated uncertainty intervals.

    Provides prediction intervals with finite-sample coverage guarantees
    based on conformal prediction theory.
    """

    def __init__(self, alpha: float = 0.1, *, rngs: nnx.Rngs):
        """Initialize conformal prediction.

        Args:
            alpha: Miscoverage level (1-alpha is the target coverage)
            rngs: Random number generators
        """
        super().__init__()
        self.alpha = alpha

        # Conformal scores from calibration set
        self.conformal_scores = nnx.Variable(jnp.array([]))
        self.quantile = nnx.Variable(jnp.array(0.0))

    def calibrate(self, predictions: jax.Array, true_values: jax.Array) -> None:
        """Calibrate conformal prediction using calibration set.

        Args:
            predictions: Model predictions on calibration set
            true_values: True values for calibration set
        """
        # Compute nonconformity scores (absolute residuals)
        scores = jnp.abs(predictions - true_values)

        # Compute quantile for prediction intervals
        n = len(scores)
        quantile_level = jnp.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_value = jnp.quantile(scores, quantile_level)

        # Store calibration results
        self.conformal_scores[...] = scores
        self.quantile[...] = quantile_value

    def predict_intervals(self, predictions: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute conformal prediction intervals.

        Args:
            predictions: Model predictions for test set

        Returns:
            Tuple of (lower_bounds, upper_bounds) for prediction intervals
        """
        interval_width = self.quantile[...]
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width

        return lower_bounds, upper_bounds

    def compute_coverage(
        self, lower_bounds: jax.Array, upper_bounds: jax.Array, true_values: jax.Array
    ) -> float:
        """Compute empirical coverage of prediction intervals.

        Args:
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            true_values: True values

        Returns:
            Empirical coverage rate
        """
        in_interval = (true_values >= lower_bounds) & (true_values <= upper_bounds)
        return float(jnp.mean(in_interval))


class CalibrationTools(nnx.Module):
    """Enhanced tools for uncertainty calibration assessment and improvement."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize calibration tools.

        Args:
            rngs: Random number generators
        """
        super().__init__()

    def assess_calibration(
        self,
        predictions: jax.Array,
        uncertainties: jax.Array,
        true_values: jax.Array,
        num_bins: int = 10,
    ) -> dict[str, float | dict[str, jax.Array]]:
        """Assess calibration quality of uncertainty estimates.

        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            true_values: Ground truth values
            num_bins: Number of bins for reliability diagram

        Returns:
            Dictionary with calibration metrics
        """
        # Compute confidence values (inverse of uncertainty)
        confidences = 1.0 / (1.0 + uncertainties)

        # Compute accuracies (whether predictions are close to truth)
        errors = jnp.abs(predictions - true_values)
        accuracies = errors < jnp.median(errors)  # Binary accuracy threshold

        # Compute reliability diagram
        reliability_data = self.compute_reliability_diagram(
            confidences, accuracies, num_bins
        )

        # Expected Calibration Error (ECE)
        bin_weights = reliability_data["bin_counts"] / jnp.sum(
            reliability_data["bin_counts"]
        )
        calibration_errors = jnp.abs(
            reliability_data["bin_confidences"] - reliability_data["bin_accuracies"]
        )
        ece = jnp.sum(bin_weights * calibration_errors)

        # Maximum Calibration Error (MCE)
        mce = jnp.max(calibration_errors)

        # Brier score
        brier_score = jnp.mean((confidences - jnp.asarray(accuracies)) ** 2)

        # Average Calibration Error for regression tasks
        ace = self._compute_average_calibration_error(uncertainties, errors)

        return {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "brier_score": float(brier_score),
            "average_calibration_error": float(ace),
            "reliability_diagram_data": reliability_data,
        }

    def _compute_average_calibration_error(
        self, uncertainties: jax.Array, errors: jax.Array
    ) -> float:
        """Compute average calibration error for regression tasks.

        Args:
            uncertainties: Predicted uncertainties
            errors: Actual prediction errors

        Returns:
            Average calibration error
        """
        # Sort by uncertainty
        sorted_indices = jnp.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = errors[sorted_indices]

        # Compute expected vs actual error in uncertainty bins
        n_samples = len(uncertainties)
        bin_size = n_samples // 10  # 10 bins

        calibration_error = 0.0
        for i in range(10):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < 9 else n_samples

            if end_idx > start_idx:
                bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
                bin_errors = sorted_errors[start_idx:end_idx]

                expected_error = jnp.mean(bin_uncertainties)
                actual_error = jnp.mean(bin_errors)

                calibration_error += (
                    jnp.abs(expected_error - actual_error)
                    * (end_idx - start_idx)
                    / n_samples
                )

        return float(calibration_error)

    def compute_reliability_diagram(
        self,
        confidences: jax.Array,
        accuracies: jax.Array,
        num_bins: int = 10,
    ) -> dict[str, jax.Array]:
        """Compute reliability diagram data.

        Args:
            confidences: Predicted confidence values
            accuracies: Binary accuracy indicators
            num_bins: Number of bins for the diagram

        Returns:
            Dictionary with binned confidence and accuracy data
        """
        # Create bin boundaries
        bin_boundaries = jnp.linspace(0.0, 1.0, num_bins + 1)

        # Initialize bin statistics
        bin_confidences = jnp.zeros(num_bins)
        bin_accuracies = jnp.zeros(num_bins)
        bin_counts = jnp.zeros(num_bins)

        # Compute bin statistics
        for i in range(num_bins):
            # Find samples in this bin
            in_bin = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1]
            )

            # Handle last bin edge case
            if i == num_bins - 1:
                in_bin = in_bin | (confidences == bin_boundaries[i + 1])

            bin_count = jnp.sum(in_bin)

            if bin_count > 0:
                bin_confidences = bin_confidences.at[i].set(
                    jnp.mean(confidences[in_bin])
                )
                bin_accuracies = bin_accuracies.at[i].set(jnp.mean(accuracies[in_bin]))
                bin_counts = bin_counts.at[i].set(bin_count)

        # Compute bin centers from boundaries
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        return {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "bin_boundaries": bin_boundaries,
            "bin_centers": bin_centers,
        }

    def platt_scaling(
        self,
        logits: jax.Array,
        labels: jax.Array,
        validation_logits: jax.Array,
    ) -> tuple[float, float]:
        """Apply Platt scaling for probability calibration.

        Args:
            logits: Training logits for fitting scaling parameters
            labels: Training labels
            validation_logits: Validation logits to calibrate

        Returns:
            Tuple of (slope, intercept) scaling parameters
        """
        # Simplified Platt scaling implementation
        # In practice, would use scipy.optimize for better fitting

        # Note: In full implementation, convert labels to +1/-1 for binary
        # binary_labels = 2 * labels - 1  # Unused in simplified implementation

        # Initial parameters
        slope = jnp.array(1.0)
        intercept = jnp.array(0.0)

        # Simple gradient-based optimization
        learning_rate = 0.01
        for _ in range(100):
            # Apply current scaling
            scaled_logits = slope * logits + intercept
            probs = jax.nn.sigmoid(scaled_logits)

            # Compute gradients of cross-entropy loss
            residuals = probs - jnp.asarray(labels > 0.5)

            slope_grad = jnp.mean(residuals * logits)
            intercept_grad = jnp.mean(residuals)

            # Update parameters
            slope -= learning_rate * slope_grad
            intercept -= learning_rate * intercept_grad

        return float(slope), float(intercept)

    def isotonic_regression_calibration(
        self,
        confidences: jax.Array,
        accuracies: jax.Array,
    ) -> jax.Array:
        """Apply isotonic regression for calibration.

        Args:
            confidences: Predicted confidence values
            accuracies: Binary accuracy indicators

        Returns:
            Calibrated confidence values
        """
        # Simplified isotonic regression - in practice would use sklearn
        # Sort by confidence
        sorted_indices = jnp.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_accuracies = accuracies[sorted_indices]

        # Apply smoothing to create monotonic calibration mapping
        window_size = max(1, len(confidences) // 20)  # 5% window
        calibrated = jnp.zeros_like(sorted_confidences)

        for i in range(len(sorted_confidences)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(sorted_confidences), i + window_size // 2 + 1)
            calibrated = calibrated.at[i].set(
                jnp.mean(sorted_accuracies[start_idx:end_idx])
            )

        # Map back to original order
        inverse_indices = jnp.argsort(sorted_indices)
        return calibrated[inverse_indices]
