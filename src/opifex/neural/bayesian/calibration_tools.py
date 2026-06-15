"""Calibration tools for uncertainty quantification and model calibration.

This module provides tools for assessing and improving the calibration of
uncertainty estimates from probabilistic models, including temperature scaling
and reliability diagram computation.
"""

from collections.abc import Callable, Sequence
from typing import TypeVar

import jax
import jax.numpy as jnp
from flax import nnx


_Params = TypeVar("_Params")


def _gradient_descent(
    loss_fn: Callable[[_Params], jax.Array],
    init_params: _Params,
    *,
    n_steps: int,
    lr: float,
    project: Callable[[_Params], _Params] | None = None,
    update: Callable[[_Params, _Params], _Params] | None = None,
    on_step: Callable[[_Params], None] | None = None,
) -> _Params:
    """Run a fixed-iteration gradient-descent loop over an arbitrary pytree.

    Centralises the ``for _ in range(n_steps): g = grad(loss)(p); p = step(p, g);
    p = project(p)`` skeleton that the temperature-scaling and Platt calibrators
    previously re-implemented four times. The step rule, projection and a
    per-iteration observer are injected so each call site keeps its original
    behaviour exactly.

    Args:
        loss_fn: Scalar loss as a function of the parameter pytree. Gradients
            are taken with :func:`jax.grad`, so ``init_params`` may be a scalar
            array or a tuple/pytree of arrays.
        init_params: Initial parameter pytree.
        n_steps: Number of gradient-descent iterations (fixed, not adaptive).
        lr: Learning rate used by the default plain-SGD update. Ignored when a
            custom ``update`` is supplied.
        project: Optional projection applied to the parameters after every
            update (e.g. a positivity floor). ``None`` leaves them unconstrained.
        update: Optional custom update mapping ``(params, grads) -> params``
            (e.g. momentum). Defaults to plain SGD ``p - lr * g`` applied
            leaf-wise via :func:`jax.tree.map`.
        on_step: Optional side-effecting observer invoked with the *current*
            parameters at the start of each iteration, before the gradient
            step. Used to record per-step diagnostics.

    Returns:
        The parameter pytree after ``n_steps`` iterations.
    """
    params = init_params
    step = (
        update
        if update is not None
        else (lambda current, grads: jax.tree.map(lambda p, g: p - lr * g, current, grads))
    )

    for _ in range(n_steps):
        if on_step is not None:
            on_step(params)
        grads = jax.grad(loss_fn)(params)
        params = step(params, grads)
        if project is not None:
            params = project(params)

    return params


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
    ) -> None:
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

    def __call__(self, predictions: jax.Array, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
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
        aleatoric_uncertainty = 0.1 * (input_magnitude + input_variance) / (1.0 + input_magnitude)

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
                return -jnp.mean(labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs))
            # Multi-class classification case
            log_probs = jax.nn.log_softmax(scaled_logits, axis=-1)
            return -jnp.mean(log_probs[jnp.arange(len(labels)), labels])

        # Enhanced optimization with momentum for adaptive mode
        positivity_floor = lambda temp: jnp.maximum(temp, 0.01)  # Ensure positive

        if self.adaptive:
            # Momentum-based optimization: 200 steps for better convergence.
            current_temp = _gradient_descent(
                calibration_loss,
                self.temperature[...],
                n_steps=200,
                lr=self.learning_rate,
                update=self._momentum_update,
                project=positivity_floor,
            )
        else:
            # Plain SGD: a fixed number of optimization steps.
            current_temp = _gradient_descent(
                calibration_loss,
                self.temperature[...],
                n_steps=100,
                lr=self.learning_rate,
                project=positivity_floor,
            )

        return float(current_temp)

    def _momentum_update(self, current_temp: jax.Array, grad: jax.Array) -> jax.Array:
        """Momentum gradient step that mutates the persistent velocity state.

        Reproduces the historical adaptive update exactly: the velocity is
        advanced as ``momentum * velocity + grad`` before the temperature is
        moved by ``-learning_rate * velocity``.

        Args:
            current_temp: Current temperature scalar.
            grad: Loss gradient at ``current_temp``.

        Returns:
            Updated temperature scalar (before projection).
        """
        self.velocity[...] = self.momentum[...] * self.velocity[...] + grad
        return current_temp - self.learning_rate * self.velocity[...]

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
            constrained_predictions, constraint_penalty = self._enforce_physics_constraints(
                scaled_predictions, inputs
            )

            # Base calibration loss (MSE)
            calibration_loss = jnp.mean((constrained_predictions - targets) ** 2)

            # Combined loss with physics penalty
            return (
                calibration_loss + self.constraint_strength * constraint_penalty,
                constraint_penalty,
            )

        scalar_loss = lambda temp: physics_aware_loss(temp)[0]
        positivity_floor = lambda temp: jnp.maximum(temp, 0.01)  # Ensure positive

        def record_penalty(temp: jax.Array) -> None:
            """Append the current-step constraint penalty to the history."""
            self.constraint_penalty_history.append(float(physics_aware_loss(temp)[1]))

        # Momentum-based when adaptive (150 steps), else plain SGD (100 steps).
        update = self._momentum_update if self.adaptive else None
        n_steps = 150 if self.adaptive else 100
        current_temp = _gradient_descent(
            scalar_loss,
            self.temperature[...],
            n_steps=n_steps,
            lr=self.learning_rate,
            update=update,
            project=positivity_floor,
            on_step=record_penalty,
        )

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

    def __init__(self, *, rngs: nnx.Rngs) -> None:
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

    def fit(self, logits: jax.Array, labels: jax.Array, max_iterations: int = 100) -> None:
        """Fit Platt scaling parameters using maximum likelihood.

        Args:
            logits: Training logits
            labels: Binary labels (0 or 1)
            max_iterations: Maximum number of optimization iterations
        """

        def loss_fn(params: tuple[jax.Array, jax.Array]) -> jax.Array:
            """Binary cross-entropy loss for Platt scaling over ``(a, b)``."""
            a_param, b_param = params
            probs = jax.nn.sigmoid(a_param * logits + b_param)
            # Clip probabilities to avoid log(0)
            probs = jnp.clip(probs, 1e-8, 1.0 - 1e-8)
            return -jnp.mean(labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs))

        # Optimize the (A, B) parameter pair with plain SGD.
        current_a, current_b = _gradient_descent(
            loss_fn,
            (self.a[...], self.b[...]),
            n_steps=max_iterations,
            lr=0.01,
        )

        # Update parameters
        self.a = nnx.Param(current_a)
        self.b = nnx.Param(current_b)


class IsotonicRegression(nnx.Module):
    """Isotonic regression for calibration.

    Non-parametric calibration method that learns a monotonic mapping
    from confidence scores to calibrated probabilities.
    """

    def __init__(self, n_bins: int = 100, *, rngs: nnx.Rngs) -> None:
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
            jnp.searchsorted(self.bin_edges[...][:-1], sorted_confidences, side="right") - 1
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
        """Pool adjacent violators algorithm (PAVA) for isotonic regression.

        Computes the non-decreasing sequence closest to ``y`` (in the
        equal-weight least-squares sense) by repeatedly merging adjacent
        out-of-order blocks and replacing each block by its mean until no
        violations remain. This is the standard pool-adjacent-violators
        algorithm; see scikit-learn's ``sklearn.isotonic`` /
        ``_inplace_contiguous_isotonic_regression`` and Best & Chakravarti
        (1990), "Active set algorithms for isotonic regression".

        The previous single forward pass averaged each adjacent pair at
        most once and therefore left order violations in place (e.g.
        ``[3, 2, 1] -> [2.5, 1.75, 1.75]``, still decreasing). Iterating to
        convergence guarantees the output is non-decreasing.

        Args:
            y: Input values to make monotonic.

        Returns:
            Monotonic (non-decreasing) version of input values.
        """
        # The input is a small, eagerly-evaluated per-bin array (length
        # ``n_bins``), so a host-side block-merge loop is appropriate and
        # keeps the routine exact and JAX-transform-free.
        values = [float(v) for v in y]
        # Each block tracks (sum, weight) so its mean is sum / weight.
        block_sums: list[float] = []
        block_weights: list[float] = []

        for value in values:
            block_sums.append(value)
            block_weights.append(1.0)
            # Merge while the last block violates monotonicity against the
            # block before it (mean of previous block > mean of last block).
            while (
                len(block_sums) > 1
                and block_sums[-2] / block_weights[-2] > block_sums[-1] / block_weights[-1]
            ):
                merged_sum = block_sums.pop() + block_sums.pop()
                merged_weight = block_weights.pop() + block_weights.pop()
                block_sums.append(merged_sum)
                block_weights.append(merged_weight)

        # Expand block means back to a per-element array.
        result: list[float] = []
        for block_sum, block_weight in zip(block_sums, block_weights, strict=True):
            block_mean = block_sum / block_weight
            result.extend([block_mean] * int(block_weight))

        return jnp.asarray(result, dtype=y.dtype)


class CalibrationTools(nnx.Module):
    """Enhanced tools for uncertainty calibration assessment and improvement."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Initialize calibration tools.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        # Retained so the Platt / isotonic helpers can instantiate the
        # canonical calibrator modules they delegate to.
        self.rngs = rngs

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
        reliability_data = self.compute_reliability_diagram(confidences, accuracies, num_bins)

        # Expected Calibration Error (ECE)
        bin_weights = reliability_data["bin_counts"] / jnp.sum(reliability_data["bin_counts"])
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
                    jnp.abs(expected_error - actual_error) * (end_idx - start_idx) / n_samples
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
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

            # Handle last bin edge case
            if i == num_bins - 1:
                in_bin = in_bin | (confidences == bin_boundaries[i + 1])

            bin_count = jnp.sum(in_bin)

            if bin_count > 0:
                bin_confidences = bin_confidences.at[i].set(jnp.mean(confidences[in_bin]))
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
        """Fit Platt scaling and return its ``(slope, intercept)`` parameters.

        Delegates to :class:`PlattScaling` (the single source of truth for
        Platt calibration) so there is exactly one fitting implementation.
        The returned ``slope`` / ``intercept`` are the fitted sigmoid
        parameters ``a`` / ``b`` from ``P(y=1|f) = sigmoid(a * f + b)``.

        Args:
            logits: Training logits for fitting scaling parameters.
            labels: Training labels.
            validation_logits: Validation logits (accepted for API
                compatibility; the fitted parameters are independent of
                them).

        Returns:
            Tuple of ``(slope, intercept)`` scaling parameters.
        """
        del validation_logits  # Parameters depend only on the training fit.
        scaler = PlattScaling(rngs=self.rngs)
        scaler.fit(logits, labels)
        return float(scaler.a[...]), float(scaler.b[...])

    def isotonic_regression_calibration(
        self,
        confidences: jax.Array,
        accuracies: jax.Array,
    ) -> jax.Array:
        """Fit isotonic regression and return calibrated confidences.

        Delegates to :class:`IsotonicRegression` (the single source of
        truth, which uses a convergent pool-adjacent-violators fit) so
        there is exactly one isotonic implementation.

        Args:
            confidences: Predicted confidence values.
            accuracies: Binary accuracy indicators.

        Returns:
            Calibrated confidence values, aligned with ``confidences``.
        """
        regressor = IsotonicRegression(rngs=self.rngs)
        regressor.fit(confidences, accuracies)
        return regressor(confidences)
