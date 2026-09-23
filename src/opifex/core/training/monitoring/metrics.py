"""Training metrics and state management for Opifex framework.

This module provides full metrics tracking for scientific machine learning,
including physics-aware metrics, quantum chemistry metrics, and advanced diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
from calibrax.metrics import MetricTracker
from calibrax.metrics.functional.regression import mae, max_error, mse
from flax import nnx
from jaxtyping import Array, ArrayLike, Float  # noqa: TC002

from opifex.core.physics.losses import PhysicsResidualReporter


# Constants for unit conversions
HARTREE_TO_KCAL_MOL = 627.50960803  # Conversion factor from Hartree to kcal/mol


def _recorded(value: ArrayLike, target: ArrayLike = 0.0) -> float:
    """Return an already-computed scalar unchanged.

    :class:`calibrax.metrics.MetricTracker` computes its metric from a prediction and a
    target pair. A training loop has already reduced its losses to scalars by the time it
    records them, so this is the metric that records one.

    Converting to a Python float rather than an array is deliberate: a float32 array
    would round every recorded value to single precision, so a learning rate of ``1e-3``
    would read back as ``0.0010000000474974513``.

    Args:
        value: The scalar to record.
        target: Unused; the tracker passes a target to every metric.

    Returns:
        ``value`` as a Python float.
    """
    del target
    return float(value)  # pyright: ignore[reportArgumentType]


class TrainingMetrics:
    """Training metrics tracking.

    Each series is a :class:`calibrax.metrics.MetricTracker`, which owns metric history
    and best-value detection for the ecosystem; this class adds the series a scientific
    training loop needs and the vocabulary to record them. Histories are read-only
    tuples, so a value enters one only through its ``update_`` method.

    SCF convergence is kept as a plain log rather than a tracker: a tracker records
    floats, and a converged flag and an iteration count are not metrics to minimise.
    """

    _FLOAT_SERIES = (
        "train_losses",
        "val_losses",
        "learning_rates",
        "physics_losses",
        "boundary_losses",
        "chemical_accuracies",
        "constraint_violations",
    )

    def __init__(self) -> None:
        """Start every series empty."""
        self._series = {name: MetricTracker(_recorded) for name in self._FLOAT_SERIES}
        self._scf_converged: list[bool] = []
        self._scf_iterations: list[int] = []

    @property
    def train_losses(self) -> tuple[float, ...]:
        """Training loss per recorded step."""
        return self._series["train_losses"].history

    @property
    def val_losses(self) -> tuple[float, ...]:
        """Validation loss per recorded evaluation."""
        return self._series["val_losses"].history

    @property
    def learning_rates(self) -> tuple[float, ...]:
        """Learning rate per recorded step."""
        return self._series["learning_rates"].history

    @property
    def physics_losses(self) -> tuple[float, ...]:
        """Physics-residual loss per recorded step."""
        return self._series["physics_losses"].history

    @property
    def boundary_losses(self) -> tuple[float, ...]:
        """Boundary-condition loss per recorded step."""
        return self._series["boundary_losses"].history

    @property
    def chemical_accuracies(self) -> tuple[float, ...]:
        """Chemical accuracy (kcal/mol) per recorded evaluation."""
        return self._series["chemical_accuracies"].history

    @property
    def constraint_violations(self) -> tuple[float, ...]:
        """Constraint violation magnitude per recorded step."""
        return self._series["constraint_violations"].history

    @property
    def scf_converged(self) -> tuple[bool, ...]:
        """Whether each recorded SCF solve converged."""
        return tuple(self._scf_converged)

    @property
    def scf_iterations(self) -> tuple[int, ...]:
        """Iterations each recorded SCF solve took."""
        return tuple(self._scf_iterations)

    @property
    def best_val_loss(self) -> float | None:
        """Lowest validation loss recorded, or ``None`` before the first."""
        tracker = self._series["val_losses"]
        return tracker.best() if tracker.history else None

    @property
    def best_val_epoch(self) -> int | None:
        """Index of the evaluation that produced :attr:`best_val_loss`."""
        tracker = self._series["val_losses"]
        return tracker.best_epoch if tracker.history else None

    def update_train_loss(self, loss: float) -> None:
        """Record a training loss.

        Args:
            loss: The loss value.
        """
        self._series["train_losses"].increment(loss, 0.0)

    def update_val_loss(self, loss: float) -> None:
        """Record a validation loss.

        Args:
            loss: The loss value.
        """
        self._series["val_losses"].increment(loss, 0.0)

    def update_learning_rate(self, learning_rate: float) -> None:
        """Record a learning rate.

        Args:
            learning_rate: The current learning rate.
        """
        self._series["learning_rates"].increment(learning_rate, 0.0)

    def update_physics_loss(self, loss: float) -> None:
        """Record a physics-residual loss.

        Args:
            loss: The loss value.
        """
        self._series["physics_losses"].increment(loss, 0.0)

    def update_boundary_loss(self, loss: float) -> None:
        """Record a boundary-condition loss.

        Args:
            loss: The loss value.
        """
        self._series["boundary_losses"].increment(loss, 0.0)

    def update_chemical_accuracy(self, accuracy: float) -> None:
        """Record a chemical accuracy in kcal/mol.

        Args:
            accuracy: The mean absolute energy error in kcal/mol.
        """
        self._series["chemical_accuracies"].increment(accuracy, 0.0)

    def update_constraint_violation(self, violation: float) -> None:
        """Record a constraint-violation magnitude.

        Args:
            violation: The violation magnitude.
        """
        self._series["constraint_violations"].increment(violation, 0.0)

    def update_scf_convergence(self, converged: bool, iterations: int) -> None:
        """Record the outcome of an SCF solve.

        Args:
            converged: Whether the solve reached its tolerance.
            iterations: Iterations the solve took.
        """
        self._scf_converged.append(converged)
        self._scf_iterations.append(iterations)

    def reset(self) -> None:
        """Clear every recorded series."""
        for tracker in self._series.values():
            tracker.reset()
        self._scf_converged.clear()
        self._scf_iterations.clear()


@dataclass(kw_only=True)
class TrainingState:
    """Enhanced training state management with full physics-aware metrics.

    The optimiser is an ``nnx.Optimizer`` (Flax NNX), which manages the optax state
    internally — there is no separate ``opt_state`` to thread.
    """

    # Core training state
    model: nnx.Module
    optimizer: nnx.Optimizer
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

    def with_updates(self, **changes: Any) -> TrainingState:
        """Return a new state with ``changes`` applied (immutable update).

        Wraps :func:`dataclasses.replace` so scalar progress fields
        (``step``, ``epoch``, ``best_loss``, ...) can be advanced without
        mutating the existing instance. Unchanged fields, including the
        ``model`` and ``optimizer`` references, are shared with the
        original. Prefer this over in-place attribute assignment.
        """
        return replace(self, **changes)

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
            converged_count = sum(1 for converged, _ in self.scf_convergence_history if converged)
            summary["scf_convergence_rate"] = converged_count / len(self.scf_convergence_history)
            avg_iterations = sum(
                iterations for _, iterations in self.scf_convergence_history
            ) / len(self.scf_convergence_history)
            summary["avg_scf_iterations"] = avg_iterations

        return summary


class MetricsCollector:
    """Collects the metrics a scientific training loop reports.

    Computes prediction accuracy, physics residuals, gradient and parameter norms,
    chemical accuracy, conservation violations and quantum diagnostics from a model and
    a batch, and keeps a history of whatever is recorded.

    Regression figures come from :mod:`calibrax.metrics.functional.regression` and the
    history from :class:`calibrax.metrics.MetricTracker`, which own those across the
    ecosystem. What stays here is what calibrax has no equivalent for: the nnx gradient
    and parameter norms, the Hartree-to-kcal/mol accuracy, and the physics and
    conservation diagnostics.
    """

    def __init__(self) -> None:
        """Start with no timing marks and no recorded history."""
        self.training_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self._history: dict[str, MetricTracker] = {}

    @property
    def metrics_history(self) -> dict[str, tuple[float, ...]]:
        """Every recorded series, keyed by metric name."""
        return {name: tracker.history for name, tracker in self._history.items()}

    def start_training(self) -> None:
        """Mark the start of training."""
        self.training_start_time = time.perf_counter()

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.perf_counter()

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
        metrics = {}

        # Basic prediction accuracy. Train/eval mode is set by the caller. The three
        # regression figures come from calibrax, which owns metrics for the ecosystem;
        # recomputing them here would be a second definition to keep in step.
        y_pred = model(x)  # pyright: ignore[reportCallIssue]
        metrics["mse_loss"] = float(mse(y_pred, y_true))
        metrics["mae_loss"] = float(mae(y_pred, y_true))
        metrics["max_error"] = float(max_error(y_pred, y_true))

        # Physics residual diagnostics when the loss reports them. Uses the
        # PhysicsResidualReporter protocol with its real
        # ``(predictions, targets, inputs) -> dict`` signature (reusing the
        # forward pass above) rather than guessing the interface.
        if isinstance(physics_loss, PhysicsResidualReporter):
            residuals = physics_loss.compute_residuals(y_pred, y_true, x)
            for name, value in residuals.items():
                metrics[name] = float(value)

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
            sum(jnp.sum(jnp.real(g * jnp.conj(g))) for g in jax.tree_util.tree_leaves(grads))
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
            metrics["epoch_time"] = time.perf_counter() - self.epoch_start_time

        if self.training_start_time is not None:
            metrics["total_training_time"] = time.perf_counter() - self.training_start_time

        return metrics

    def collect_convergence_metrics(self, training_state: TrainingState) -> dict[str, float]:
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
                1 for converged, _ in training_state.scf_convergence_history if converged
            )
            total_count = len(training_state.scf_convergence_history)
            metrics["scf_convergence_rate"] = float(converged_count / total_count)

            avg_iterations = (
                sum(iterations for _, iterations in training_state.scf_convergence_history)
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
            tracker = self._history.setdefault(metric_name, MetricTracker(_recorded))
            tracker.increment(value, 0.0)

    def get_metrics_summary(self, window_size: int = 10) -> dict[str, dict[str, float]]:
        """Get a summary of metrics over a recent window.

        Args:
            window_size: Number of recent measurements to consider

        Returns:
            Dictionary of metric summaries
        """
        summary = {}

        for metric_name, tracker in self._history.items():
            values = tracker.history
            if not values:
                continue

            recent_values = list(values[-window_size:])
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
        x: Float[Array, "batch ..."],  # noqa: ARG002 - metric-collector interface takes inputs
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
            exchange_correlation_weight = quantum_config.get("exchange_correlation_weight", 0.3)
            metrics["exchange_correlation_energy"] = float(dft_energy * exchange_correlation_weight)

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
        x: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes inputs
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
        x: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes inputs
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
        x: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes inputs
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes targets
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
        return particle_number_violation(y_pred, target_particle_number=10.0, tolerance=1e-4)

    def _compute_symmetry_conservation(
        self,
        x: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes inputs
        y_pred: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],  # noqa: ARG002 - conservation-metric interface takes targets
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
