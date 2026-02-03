"""Training metrics and state management for Opifex framework.

This module provides comprehensive metrics tracking for scientific machine learning,
including physics-aware metrics, quantum chemistry metrics, and advanced diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    import optax
    from jaxtyping import Array, Float


# Constants for unit conversions
HARTREE_TO_KCAL_MOL = 627.50960803  # Conversion factor from Hartree to kcal/mol


class TrainingMetrics:
    """Training metrics tracking."""

    def __init__(self):
        """Initialize metrics tracking."""
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss: float | None = None
        self.learning_rates: list[float] = []

        # Physics-informed loss metrics
        self.physics_losses: list[float] = []
        self.boundary_losses: list[float] = []

        # Quantum-specific metrics
        self.chemical_accuracies: list[float] = []
        self.scf_converged: list[bool] = []
        self.scf_iterations: list[int] = []
        self.constraint_violations: list[float] = []

    def update_train_loss(self, loss: float) -> None:
        """Update training loss."""
        self.train_losses.append(loss)

    def update_val_loss(self, loss: float) -> None:
        """Update validation loss."""
        self.val_losses.append(loss)
        if self.best_val_loss is None or loss < self.best_val_loss:
            self.best_val_loss = loss

    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate."""
        self.learning_rates.append(lr)

    def update_chemical_accuracy(self, accuracy: float) -> None:
        """Update chemical accuracy (kcal/mol)."""
        self.chemical_accuracies.append(accuracy)

    def update_scf_convergence(self, converged: bool, iterations: int) -> None:
        """Update SCF convergence status."""
        self.scf_converged.append(converged)
        self.scf_iterations.append(iterations)

    def update_constraint_violation(self, violation: float) -> None:
        """Update constraint violation."""
        self.constraint_violations.append(violation)


@dataclass
class TrainingState:
    """Enhanced training state management with comprehensive physics-aware metrics.

    Modernized for Flax NNX compliance while maintaining optax flexibility.
    """

    # Core training state - maintaining optax compatibility
    model: nnx.Module
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    step: int = 0
    epoch: int = 0
    rngs: nnx.Rngs = field(default_factory=lambda: nnx.Rngs(0))

    # Enhanced metrics tracking
    best_loss: float = float("inf")
    best_val_loss: float = float("inf")
    convergence_threshold: float = 1e-6
    plateau_count: int = 0

    # Physics-aware metrics
    physics_metrics: dict[str, list[float]] = field(default_factory=dict)
    conservation_violations: dict[str, list[float]] = field(default_factory=dict)
    chemical_accuracy_history: list[float] = field(default_factory=list)
    scf_convergence_history: list[tuple[bool, int]] = field(default_factory=list)

    # Training diagnostics
    gradient_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    wall_time_history: list[float] = field(default_factory=list)

    # Recovery and checkpointing
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    recovery_state: dict[str, Any] = field(default_factory=dict)

    def update_physics_metric(self, metric_name: str, value: float) -> None:
        """Update a physics-specific metric."""
        if metric_name not in self.physics_metrics:
            self.physics_metrics[metric_name] = []
        self.physics_metrics[metric_name].append(value)

    def update_conservation_violation(self, violation_type: str, value: float) -> None:
        """Update conservation law violation tracking."""
        if violation_type not in self.conservation_violations:
            self.conservation_violations[violation_type] = []
        self.conservation_violations[violation_type].append(value)

    def update_chemical_accuracy(self, accuracy: float) -> None:
        """Update chemical accuracy tracking."""
        self.chemical_accuracy_history.append(accuracy)

    def update_scf_convergence(self, converged: bool, iterations: int) -> None:
        """Update SCF convergence tracking."""
        self.scf_convergence_history.append((converged, iterations))

    def update_gradient_norm(self, norm: float) -> None:
        """Update gradient norm tracking."""
        self.gradient_norms.append(norm)

    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate tracking."""
        self.learning_rates.append(lr)

    def update_wall_time(self, time: float) -> None:
        """Update wall time tracking."""
        self.wall_time_history.append(time)

    def is_converged(self, current_loss: float) -> bool:
        """Check if training has converged."""
        # First check loss-based convergence
        loss_converged = current_loss < self.convergence_threshold

        # Check chemical accuracy convergence if available
        chemical_accuracy_converged = False
        if len(self.chemical_accuracy_history) > 0:
            chemical_accuracy_converged = (
                self.chemical_accuracy_history[-1] < 1e-3
            )  # Chemical accuracy target

        # Require both conditions for convergence (if chemical accuracy is available)
        if len(self.chemical_accuracy_history) > 0:
            return loss_converged and chemical_accuracy_converged
        return loss_converged

    def get_physics_summary(self) -> dict[str, float]:
        """Get summary of physics-related metrics."""
        summary = {}

        # Latest physics metrics
        for metric_name, values in self.physics_metrics.items():
            if values:
                summary[f"latest_{metric_name}"] = values[-1]
                summary[f"avg_{metric_name}"] = sum(values) / len(values)

        # Conservation violation summary
        for violation_type, values in self.conservation_violations.items():
            if values:
                summary[f"max_{violation_type}_violation"] = max(values)
                summary[f"avg_{violation_type}_violation"] = sum(values) / len(values)

        # Chemical accuracy
        if self.chemical_accuracy_history:
            summary["latest_chemical_accuracy"] = self.chemical_accuracy_history[-1]
            summary["best_chemical_accuracy"] = min(self.chemical_accuracy_history)

        # SCF convergence
        if self.scf_convergence_history:
            converged_count = sum(
                1 for converged, _ in self.scf_convergence_history if converged
            )
            summary["scf_convergence_rate"] = converged_count / len(
                self.scf_convergence_history
            )
            avg_iterations = sum(
                iterations for _, iterations in self.scf_convergence_history
            ) / len(self.scf_convergence_history)
            summary["avg_scf_iterations"] = avg_iterations

        return summary


class AdvancedMetricsCollector:
    """Advanced metrics collection for scientific computing training."""

    def __init__(self):
        """Initialize the advanced metrics collector."""
        self.training_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.metrics_history: dict[str, list[float]] = {}

    def start_training(self) -> None:
        """Mark the start of training."""
        self.training_start_time = time.time()

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()

    def collect_physics_metrics(
        self,
        model: nnx.Module,
        x: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
        physics_loss: Any = None,
    ) -> dict[str, float]:
        """Collect physics-specific metrics.

        Args:
            model: The neural network model
            x: Input batch
            y_true: Target output batch
            physics_loss: Physics loss instance

        Returns:
            Dictionary of physics metrics
        """
        from opifex.core.training.utils_legacy import safe_model_call

        metrics = {}

        # Basic prediction accuracy
        y_pred = safe_model_call(model, x)
        mse_loss = jnp.mean((y_pred - y_true) ** 2)
        mae_loss = jnp.mean(jnp.abs(y_pred - y_true))

        metrics["mse_loss"] = float(mse_loss)
        metrics["mae_loss"] = float(mae_loss)
        metrics["max_error"] = float(jnp.max(jnp.abs(y_pred - y_true)))

        # Physics-specific metrics if available
        if physics_loss is not None:
            if hasattr(physics_loss, "compute_residuals"):
                residual = physics_loss.compute_residuals(model, x)
                metrics["physics_residual"] = float(residual)

            if hasattr(physics_loss, "check_conservation_violations"):
                violations = physics_loss.check_conservation_violations(model, x)
                for violation_type, value in violations.items():
                    metrics[f"{violation_type}_violation"] = float(value)

        # Chemical accuracy if quantum chemistry problem
        if y_true.shape[-1] == 1:  # Energy prediction
            energy_error_hartree = jnp.abs(y_pred - y_true)
            energy_error_kcal_mol = energy_error_hartree * HARTREE_TO_KCAL_MOL
            metrics["chemical_accuracy"] = float(jnp.mean(energy_error_kcal_mol))

        return metrics

    def collect_training_diagnostics(
        self, model: nnx.Module, grads: Any, learning_rate: float
    ) -> dict[str, float]:
        """Collect training diagnostics.

        Args:
            model: The neural network model
            grads: Model gradients
            learning_rate: Current learning rate

        Returns:
            Dictionary of training diagnostics
        """
        metrics = {}

        # Gradient diagnostics (handle complex gradients properly)
        grad_norm = jnp.sqrt(
            sum(
                jnp.sum(jnp.real(g * jnp.conj(g)))
                for g in jax.tree_util.tree_leaves(grads)
            )
        )
        metrics["gradient_norm"] = float(grad_norm)

        # Learning rate
        metrics["learning_rate"] = float(learning_rate)

        # Parameter norms (handle complex parameters properly)
        params = nnx.state(model, nnx.Param)
        param_norm = jnp.sqrt(
            sum(
                jnp.sum(jnp.real(p * jnp.conj(p)))
                for p in jax.tree_util.tree_leaves(nnx.to_tree(params))
            )
        )
        metrics["parameter_norm"] = float(param_norm)

        # Timing information
        if self.epoch_start_time is not None:
            metrics["epoch_time"] = time.time() - self.epoch_start_time

        if self.training_start_time is not None:
            metrics["total_training_time"] = time.time() - self.training_start_time

        return metrics

    def collect_convergence_metrics(
        self, training_state: TrainingState
    ) -> dict[str, float]:
        """Collect convergence-related metrics.

        Args:
            training_state: Current training state

        Returns:
            Dictionary of convergence metrics
        """
        metrics = {}

        # Convergence status
        current_loss = training_state.best_loss
        metrics["is_converged"] = float(training_state.is_converged(current_loss))
        metrics["best_loss"] = float(training_state.best_loss)
        metrics["best_val_loss"] = float(training_state.best_val_loss)
        metrics["plateau_count"] = float(training_state.plateau_count)

        # Chemical accuracy convergence
        if training_state.chemical_accuracy_history:
            best_accuracy = min(training_state.chemical_accuracy_history)
            current_accuracy = training_state.chemical_accuracy_history[-1]
            metrics["best_chemical_accuracy"] = float(best_accuracy)
            metrics["current_chemical_accuracy"] = float(current_accuracy)
            metrics["chemical_accuracy_converged"] = float(current_accuracy < 1e-3)

        # SCF convergence statistics
        if training_state.scf_convergence_history:
            converged_count = sum(
                1
                for converged, _ in training_state.scf_convergence_history
                if converged
            )
            total_count = len(training_state.scf_convergence_history)
            metrics["scf_convergence_rate"] = float(converged_count / total_count)

            avg_iterations = (
                sum(
                    iterations
                    for _, iterations in training_state.scf_convergence_history
                )
                / total_count
            )
            metrics["avg_scf_iterations"] = float(avg_iterations)

        return metrics

    def update_metrics_history(self, new_metrics: dict[str, float]) -> None:
        """Update the metrics history with new values.

        Args:
            new_metrics: Dictionary of new metric values
        """
        for metric_name, value in new_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)

    def get_metrics_summary(self, window_size: int = 10) -> dict[str, dict[str, float]]:
        """Get a summary of metrics over a recent window.

        Args:
            window_size: Number of recent measurements to consider

        Returns:
            Dictionary of metric summaries
        """
        summary = {}

        for metric_name, values in self.metrics_history.items():
            if not values:
                continue

            recent_values = values[-window_size:]
            summary[metric_name] = {
                "current": recent_values[-1],
                "mean": sum(recent_values) / len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "trend": (recent_values[-1] - recent_values[0]) / len(recent_values)
                if len(recent_values) > 1
                else 0.0,
            }

        return summary

    def collect_quantum_metrics(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        quantum_config: dict[str, Any],
    ) -> dict[str, float]:
        """Collect quantum training metrics.

        Args:
            x: Input batch
            y_pred: Model predictions
            quantum_config: Quantum training configuration

        Returns:
            Dictionary of quantum metrics
        """
        metrics = {}

        # Early return if quantum training is disabled
        if not quantum_config.get("quantum_training", False):
            return metrics

        # DFT energy metrics
        if quantum_config.get("dft_functional"):
            dft_energy = self._compute_dft_energy(y_pred)
            metrics["dft_energy"] = float(dft_energy)

            # Exchange-correlation energy (approximation)
            exchange_correlation_weight = quantum_config.get(
                "exchange_correlation_weight", 0.3
            )
            metrics["exchange_correlation_energy"] = float(
                dft_energy * exchange_correlation_weight
            )

        # Quantum state tracking
        if quantum_config.get("track_quantum_states", False):
            quantum_state = self._compute_quantum_state(y_pred)
            metrics["quantum_state"] = float(quantum_state)

        return metrics

    def collect_conservation_metrics(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
        conservation_config: dict[str, Any],
    ) -> dict[str, float]:
        """Collect conservation law metrics.

        Args:
            x: Input batch
            y_pred: Model predictions
            y_true: Target outputs
            conservation_config: Conservation law configuration

        Returns:
            Dictionary of conservation metrics
        """
        metrics = {}
        conservation_laws = conservation_config.get("conservation_laws", [])

        # Conservation handlers mapping
        conservation_handlers = {
            "energy": self._compute_energy_conservation,
            "momentum": self._compute_momentum_conservation,
            "particle_number": self._compute_particle_conservation,
            "symmetry": self._compute_symmetry_conservation,
        }

        # Compute metrics for each configured conservation law
        for law in conservation_laws:
            if law in conservation_handlers:
                violation = conservation_handlers[law](x, y_pred, y_true)
                metrics[f"{law}_conservation"] = float(violation)

        return metrics

    def _compute_dft_energy(self, y_pred: Float[Array, "batch ..."]) -> jax.Array:
        """Compute DFT energy approximation.

        Args:
            y_pred: Model predictions

        Returns:
            DFT energy value
        """
        # Simple DFT energy approximation using kinetic energy functional
        return jnp.sum(y_pred**2) * 0.5

    def _compute_quantum_state(self, y_pred: Float[Array, "batch ..."]) -> jax.Array:
        """Compute quantum state measure for tracking.

        Args:
            y_pred: Model predictions

        Returns:
            Quantum state measure
        """
        # Simple quantum state measure using L1 norm (total probability)
        return jnp.sum(jnp.abs(y_pred))

    def _compute_energy_conservation(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> jax.Array:
        """Compute energy conservation violation (delegates to physics_constraints).

        Args:
            x: Input batch
            y_pred: Model predictions
            y_true: Target outputs

        Returns:
            Energy conservation violation measure
        """
        from opifex.core.physics.conservation import energy_violation

        return energy_violation(y_pred, y_true, tolerance=1e-6, monitoring_enabled=True)

    def _compute_momentum_conservation(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> jax.Array:
        """Compute momentum conservation violation (delegates to physics_constraints).

        NOTE: This now uses component-wise momentum conservation (physically correct!).

        Args:
            x: Input batch
            y_pred: Model predictions
            y_true: Target outputs

        Returns:
            Momentum conservation violation measure
        """
        from opifex.core.physics.conservation import momentum_violation

        return momentum_violation(y_pred, y_true, tolerance=1e-5)

    def _compute_particle_conservation(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> jax.Array:
        """Compute particle number conservation violation.

        Delegates to physics_constraints module.

        Args:
            x: Input batch
            y_pred: Model predictions
            y_true: Target outputs

        Returns:
            Particle conservation violation measure
        """
        from opifex.core.physics.conservation import particle_number_violation

        # Use a default target of 10 particles (can be made configurable later)
        return particle_number_violation(
            y_pred, target_particle_number=10.0, tolerance=1e-4
        )

    def _compute_symmetry_conservation(
        self,
        x: Float[Array, "batch ..."],
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> jax.Array:
        """Compute symmetry conservation violation (delegates to physics_constraints).

        Args:
            x: Input batch
            y_pred: Model predictions
            y_true: Target outputs

        Returns:
            Symmetry conservation violation measure
        """
        from opifex.core.physics.conservation import symmetry_violation

        return symmetry_violation(y_pred, tolerance=1e-6)
