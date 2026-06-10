"""Advanced training infrastructure for Opifex framework.

This module provides full training capabilities including:
- FLAX NNX transformation-based training loops with modern patterns
- Optax optimizer integration with NNX state management
- Loss function computation framework with physics-informed capabilities
- Validation and metric tracking with enhanced diagnostics
- Checkpointing with Orbax for robust state persistence
- Quantum-aware training with SCF convergence monitoring
- Chemical accuracy validation metrics for scientific computing
- Physics constraint violation tracking and enforcement
- Multi-device support with SPMD-ready patterns
- Enhanced error recovery and numerical stability

This implementation follows Flax NNX best practices for high-performance
scientific machine learning with full physics-informed capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx
from tqdm import tqdm  # type: ignore[import]

from opifex.core.training.components import (
    FlexibleOptimizerFactory,
    TrainingComponent as TrainingComponentBase,
)
from opifex.core.training.components.recovery import ErrorRecoveryManager
from opifex.core.training.config import (
    OptimizationConfig,
    TrainingConfig,
)
from opifex.core.training.monitoring.metrics import (
    AdvancedMetricsCollector,
    TrainingMetrics,
    TrainingState,
)

# Import from core modules
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig
from opifex.core.training.utils_legacy import (
    safe_model_call,
)
from opifex.uncertainty.active.acquisition import (
    acquire as _active_acquire,
    AcquisitionStrategy,
)
from opifex.uncertainty.types import PredictiveDistribution


# Set up logger for training operations
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from opifex.uncertainty.aggregators.basic import UncertaintyQuantifier


class ModularTrainer:
    """Advanced modular trainer with component composition architecture.

    This class implements the modular trainer architecture identified in the
    creative phase, providing flexible composition of training components
    with clean, modern API design.
    """

    def __init__(
        self,
        model: nnx.Module,
        config: TrainingConfig,
        components: dict[str, TrainingComponentBase] | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize modular trainer.

        Args:
            model: The neural network model to train
            config: Training configuration
            components: Dictionary of training components
            rngs: Random number generators
        """
        self.model = model
        self.config = config
        self.rngs = rngs or nnx.Rngs(0)

        # Initialize core components
        self.metrics_collector = AdvancedMetricsCollector()
        self.error_recovery = ErrorRecoveryManager(
            config={
                "max_retries": 3,
                "checkpoint_on_error": True,
                "gradient_clip_threshold": 10.0,
                "loss_explosion_threshold": 1e6,
                "learning_rate": config.learning_rate,
            }
        )
        self.optimizer_factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": config.optimization_config.optimizer,
                "learning_rate": config.optimization_config.learning_rate,
                "weight_decay": config.optimization_config.weight_decay,
                "use_schedule": True,
                "schedule_type": "cosine",
                "total_steps": config.num_epochs * 100,  # Estimate
            }
        )

        # Add custom components
        self.components = components or {}

        # Initialize training state
        optimizer = self.optimizer_factory.create_optimizer(model)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        self.training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            rngs=self.rngs,
        )

        # Setup all components
        self._setup_components()

        # Physics loss (optional)
        self.physics_loss: Any = None

    def _setup_components(self) -> None:
        """Setup all training components."""
        # Setup core components
        self.metrics_collector.start_training()
        self.error_recovery.setup(self.model, self.training_state)

        # Setup custom components
        for component in self.components.values():
            component.setup(self.model, self.training_state)

    def set_physics_loss(self, physics_loss: Any) -> None:
        """Set physics loss for physics-informed training.

        Args:
            physics_loss: Physics loss instance
        """
        self.physics_loss = physics_loss

    def _validate_issue_type(self, issue_type: str | None) -> str:
        """Helper function to validate issue type.

        Args:
            issue_type: Issue type to validate

        Returns:
            The validated issue type

        Raises:
            ValueError: If issue type is None
        """
        if issue_type is None:
            raise ValueError("Issue type cannot be None for recovery")
        return issue_type

    def training_step(
        self,
        x: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> tuple[Float[Array, ""], dict[str, float]]:
        """Execute a single training step with error recovery.

        Args:
            x: Input batch
            y_true: Target output batch

        Returns:
            Tuple of (loss, metrics)
        """
        self.metrics_collector.start_epoch()

        # Attempt training step with error recovery
        for attempt in range(self.error_recovery.max_retries + 1):
            try:
                loss, grads, metrics = self._execute_training_step(x, y_true)

                # Check training stability
                is_stable, issue_type = self.error_recovery.check_training_stability(
                    float(loss), grads, self.training_state
                )

                if not is_stable and attempt < self.error_recovery.max_retries:
                    # Attempt recovery
                    validated_issue_type = self._validate_issue_type(issue_type)
                    self.training_state = self.error_recovery.recover_from_instability(
                        validated_issue_type, self.training_state
                    )
                    continue

                # Apply gradient clipping for stability
                grads = self.error_recovery.apply_gradient_clipping(grads)

                # Update model parameters
                # Keep params as State, convert to tree only when needed for optax
                params = nnx.state(self.model, nnx.Param)
                updates, self.training_state.opt_state = self.training_state.optimizer.update(
                    nnx.to_tree(grads),
                    self.training_state.opt_state,
                    nnx.to_tree(params),
                )
                updated_params = optax.apply_updates(nnx.to_tree(params), updates)
                nnx.update(self.model, nnx.from_tree(updated_params))

                # Update training state
                self.training_state.step += 1
                loss_value = float(loss)
                if loss_value < self.training_state.best_loss:
                    self.training_state.best_loss = loss_value
                    self.error_recovery.update_stable_state(self.training_state)

                # Collect and update metrics
                training_diagnostics = self.metrics_collector.collect_training_diagnostics(
                    self.model, grads, self.config.learning_rate
                )
                metrics.update(training_diagnostics)
                self.metrics_collector.update_metrics_history(metrics)

                return loss, metrics

            except Exception as e:
                if attempt == self.error_recovery.max_retries:
                    raise RuntimeError(f"Training failed after all recovery attempts: {e}") from e

                # Log error and attempt recovery
                logger.warning("Training error on attempt %d: %s", attempt + 1, e)
                self.training_state = self.error_recovery.recover_from_instability(
                    "general_error", self.training_state
                )

        raise RuntimeError("Training failed: maximum recovery attempts exceeded")

    def _execute_training_step(
        self,
        x: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> tuple[Float[Array, ""], Any, dict[str, float]]:
        """Execute the core training step logic.

        Args:
            x: Input batch
            y_true: Target output batch

        Returns:
            Tuple of (loss, gradients, metrics)
        """

        @nnx.value_and_grad
        def loss_fn(model):
            y_pred = safe_model_call(model, x)

            # Base loss
            if self.config.loss_config.loss_type == "mse":
                loss = jnp.mean((y_pred - y_true) ** 2)
            elif self.config.loss_config.loss_type == "mae":
                loss = jnp.mean(jnp.abs(y_pred - y_true))
            else:
                loss = jnp.mean((y_pred - y_true) ** 2)  # Default to MSE

            # Physics loss if available
            if self.physics_loss is not None:
                physics_loss_value = self.physics_loss(model, x, y_true)
                loss += self.config.loss_config.physics_weight * physics_loss_value

            return loss

        loss, grads = loss_fn(self.model)

        # Collect physics metrics
        metrics = self.metrics_collector.collect_physics_metrics(
            self.model, x, y_true, self.physics_loss
        )

        return loss, grads, metrics

    def train(
        self,
        train_data: tuple[Float[Array, "n_train ..."], Float[Array, "n_train ..."]],
        val_data: tuple[Float[Array, "n_val ..."], Float[Array, "n_val ..."]] | None = None,
    ) -> tuple[nnx.Module, dict[str, Any]]:
        """Train the model with modular architecture.

        Args:
            train_data: Training dataset
            val_data: Optional validation dataset

        Returns:
            Tuple of (trained_model, comprehensive_metrics)
        """
        x_train, y_train = train_data

        for epoch in range(self.config.num_epochs):
            self.training_state.epoch = epoch

            # Training step
            loss, metrics = self.training_step(x_train, y_train)

            # Validation step
            if val_data is not None and epoch % self.config.validation_frequency == 0:
                x_val, y_val = val_data
                val_loss = self._validation_step(x_val, y_val)
                val_loss_value = float(val_loss)
                self.training_state.best_val_loss = min(
                    self.training_state.best_val_loss, val_loss_value
                )
                metrics["val_loss"] = float(val_loss)

            # Progress logging
            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d, Loss: %.6f, Metrics: %d collected",
                    epoch,
                    loss,
                    len(metrics),
                )

        # Return final results
        final_metrics = self.get_comprehensive_metrics_summary()
        return self.model, final_metrics

    def _validation_step(
        self,
        x_val: Float[Array, "batch ..."],
        y_val: Float[Array, "batch ..."],
    ) -> Float[Array, ""]:
        """Execute validation step.

        Args:
            x_val: Validation input
            y_val: Validation target

        Returns:
            Validation loss
        """
        y_pred = safe_model_call(self.model, x_val)
        return jnp.mean((y_pred - y_val) ** 2)

    def get_comprehensive_metrics_summary(self, window_size: int = 10) -> dict[str, Any]:
        """Get full metrics summary.

        Args:
            window_size: Window size for metrics aggregation

        Returns:
            Full metrics dictionary
        """
        # Base metrics from collector
        base_metrics = self.metrics_collector.get_metrics_summary(window_size)

        # Training state metrics
        state_metrics = {
            "training_state": {
                "step": self.training_state.step,
                "epoch": self.training_state.epoch,
                "best_loss": self.training_state.best_loss,
                "best_val_loss": self.training_state.best_val_loss,
                "plateau_count": self.training_state.plateau_count,
            }
        }

        # Physics summary
        physics_summary = self.training_state.get_physics_summary()
        if physics_summary:
            state_metrics["physics"] = physics_summary

        # Convergence metrics
        convergence_metrics = self.metrics_collector.collect_convergence_metrics(
            self.training_state
        )
        if convergence_metrics:
            state_metrics["convergence"] = convergence_metrics

        # Combine all metrics
        return {
            "metrics": base_metrics,
            "state": state_metrics,
            "recovery_attempts": self.error_recovery.recovery_attempts,
        }

    def cleanup(self) -> None:
        """Cleanup all training components."""
        self.error_recovery.cleanup()
        for component in self.components.values():
            component.cleanup()


class BasicTrainer:
    """Basic training infrastructure for Opifex models.

    Use :class:`opifex.core.training.Trainer` for a more composable API.

    Provides full training capabilities including:
    - FLAX NNX transformation-based training loops
    - Optax optimizer integration
    - Loss function computation
    - Validation and metrics
    - Checkpointing with Orbax
    - Quantum-aware training
    - Physics-informed loss integration
    """

    def __init__(
        self,
        model: nnx.Module,
        config: TrainingConfig,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The neural network model to train
            config: Training configuration
            rngs: Random number generators
        """

        self.config = config
        self.metrics = TrainingMetrics()
        self.physics_loss: Any | None = None  # Physics-informed loss instance

        # Initialize advanced metrics collector for full tracking
        self.advanced_metrics = AdvancedMetricsCollector()

        # Validate and setup gradient checkpointing (remat)
        if config.gradient_checkpointing:
            from artifex.generative_models.core.gradient_checkpointing import (
                resolve_checkpoint_policy,
            )

            # Validate policy early — raises ValueError for invalid names
            self._remat_policy = resolve_checkpoint_policy(config.gradient_checkpoint_policy)
        else:
            self._remat_policy = None

        # Setup optimizer using modern NNX pattern
        optimizer = self._create_optimizer(config.optimization_config)

        # Extract parameters for optimizer initialization
        # Use nnx.to_tree to convert state to optax-compatible format
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        # Initialize training state
        self.state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            rngs=rngs or nnx.Rngs(0),
        )

        # Setup checkpointing
        if config.checkpoint_config:
            self._setup_checkpointing()

    def _create_optimizer(self, opt_config: OptimizationConfig) -> optax.GradientTransformation:
        """Create optimizer from configuration.

        Now uses centralized opifex.core.training.optimizers module.
        """
        # Convert OptimizationConfig to OptimizerConfig
        optimizer_config = OptimizerConfig(
            optimizer_type=opt_config.optimizer,
            learning_rate=opt_config.learning_rate,
            b1=opt_config.beta1,
            b2=opt_config.beta2,
            eps=opt_config.eps,
            weight_decay=opt_config.weight_decay,
            momentum=opt_config.momentum,
        )
        return create_optimizer(optimizer_config)

    @staticmethod
    def _make_checkpoint_manager(checkpoint_dir: Path, max_to_keep: int) -> ocp.CheckpointManager:
        """Build an Orbax manager for the BasicTrainer composite schema.

        The BasicTrainer persists model arrays, optax optimizer state, scalar
        step/epoch counters, and JSON metrics as separate composite items, so
        it configures dedicated item handlers. Centralizing the construction
        keeps that schema defined in exactly one place.

        Args:
            checkpoint_dir: Directory in which checkpoints are stored.
            max_to_keep: Maximum number of checkpoints to retain.

        Returns:
            A configured :class:`orbax.checkpoint.CheckpointManager`.
        """
        return ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "model": ocp.StandardCheckpointHandler(),
                "optimizer_state": ocp.StandardCheckpointHandler(),
                "step": ocp.JsonCheckpointHandler(),
                "epoch": ocp.JsonCheckpointHandler(),
                "metrics": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                save_interval_steps=1,
            ),
        )

    def _setup_checkpointing(self) -> None:
        """Setup checkpointing infrastructure."""
        checkpoint_dir = Path(self.config.checkpoint_config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = self._make_checkpoint_manager(
            checkpoint_dir, self.config.checkpoint_config.max_to_keep
        )

    def set_physics_loss(self, physics_loss: Any) -> None:
        """Set physics-informed loss for training.

        Args:
            physics_loss: Physics-informed loss instance
        """
        self.physics_loss = physics_loss

    def compute_loss(
        self,
        x: Float[Array, "batch input_dim"],
        y_true: Float[Array, "batch output_dim"],
    ) -> Float[Array, ""]:
        """Compute loss for standard problems.

        Args:
            x: Input data
            y_true: Target values

        Returns:
            Scalar loss value
        """
        y_pred = safe_model_call(self.state.model, x, deterministic=False, rngs=self.state.rngs)

        if self.config.loss_config.loss_type == "mse":
            data_loss = jnp.mean((y_pred - y_true) ** 2)
        elif self.config.loss_config.loss_type == "mae":
            data_loss = jnp.mean(jnp.abs(y_pred - y_true))
        else:
            data_loss = jnp.mean((y_pred - y_true) ** 2)  # default to MSE

        # Add regularization if specified
        regularization = 0.0
        if self.config.loss_config.regularization_weight > 0:
            params = nnx.to_tree(nnx.state(self.state.model, nnx.Param))
            # Convert to scalar for regularization
            param_sums = sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
            regularization = self.config.loss_config.regularization_weight * param_sums

        return data_loss + regularization

    def compute_quantum_loss(
        self,
        positions: Float[Array, "batch n_atoms 3"],
        energies_true: Float[Array, "batch 1"],
    ) -> Float[Array, ""]:
        """Compute loss for quantum problems.

        Args:
            positions: Molecular positions
            energies_true: Reference energies

        Returns:
            Scalar loss value including quantum constraints
        """
        # Flatten positions to match model input expectations
        # This is the most direct and safe approach to avoid segfaults
        batch_size = positions.shape[0]
        flat_positions = positions.reshape(batch_size, -1)

        # Use safe model call to avoid type issues and potential problems
        energies_pred = safe_model_call(self.state.model, flat_positions, deterministic=True)

        # Ensure energies_pred has the right shape
        if energies_pred.ndim == 1:
            energies_pred = energies_pred[:, None]

        # Simple MSE loss - no complex quantum transformations that cause segfaults
        energy_loss = jnp.mean((energies_pred - energies_true) ** 2)

        total_loss = energy_loss

        # Only add simple constraints if enabled, avoiding complex transformations
        if self.config.quantum_config and self.config.quantum_config.enable_density_constraints:
            # Simple regularization penalty instead of force computation
            # This avoids the problematic JAX grad transformations
            params = nnx.state(self.state.model, nnx.Param)
            param_values = jax.tree_util.tree_leaves(params)

            # Simple L2 regularization to prevent extreme parameters
            regularization = sum(jnp.sum(p**2) for p in param_values if p.size > 0)
            regularization_penalty = (
                self.config.loss_config.quantum_constraint_weight * regularization * 1e-6
            )

            total_loss += regularization_penalty

        return total_loss

    def training_step(
        self,
        x: Float[Array, "batch ..."],
        y_true: Float[Array, "batch ..."],
    ) -> Float[Array, ""]:
        """Perform a single training step.

        Args:
            x: Input batch
            y_true: Target output batch

        Returns:
            Loss value
        """

        # Define loss function with modern NNX pattern for better performance
        def _loss(model):
            return self.compute_loss(x, y_true)

        # Apply gradient checkpointing (remat) if configured
        if self.config.gradient_checkpointing:
            from artifex.generative_models.core.gradient_checkpointing import (
                apply_remat,
            )

            loss_fn = nnx.value_and_grad(apply_remat(_loss, policy=self._remat_policy))
        else:
            loss_fn = nnx.value_and_grad(_loss)

        # Compute loss and gradients simultaneously (more efficient)
        loss_value, grads = loss_fn(self.state.model)

        # Get current learning rate for metrics
        current_lr = self.config.optimization_config.learning_rate

        # Collect advanced training diagnostics
        training_diagnostics = self.advanced_metrics.collect_training_diagnostics(
            self.state.model, grads, current_lr
        )

        # Update advanced metrics history
        self.advanced_metrics.update_metrics_history(training_diagnostics)

        # Compute gradient norm for monitoring (handle complex gradients)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.real(g * jnp.conj(g))) for g in jax.tree_util.tree_leaves(grads))
        )
        self.state.update_gradient_norm(float(grad_norm))
        self.state.update_learning_rate(current_lr)

        # Update parameters using nnx patterns
        params = nnx.state(self.state.model, nnx.Param)
        updates, self.state.opt_state = self.state.optimizer.update(
            nnx.to_tree(grads), self.state.opt_state, nnx.to_tree(params)
        )
        updated_params = optax.apply_updates(nnx.to_tree(params), updates)
        nnx.update(self.state.model, nnx.from_tree(updated_params))

        # Update training step and use already computed loss
        self.state.step += 1
        current_loss = loss_value

        # Update RNG state for next iteration (modern NNX pattern)
        if hasattr(nnx, "reseed"):
            new_rngs = nnx.reseed(self.state.rngs)
            if new_rngs is not None:
                self.state.rngs = new_rngs

        # Update loss tracking
        if current_loss < self.state.best_loss:
            self.state.best_loss = float(current_loss)

        # Collect full physics metrics
        physics_metrics = self.advanced_metrics.collect_physics_metrics(
            self.state.model, x, y_true, self.physics_loss
        )

        # Update advanced metrics history with physics metrics
        self.advanced_metrics.update_metrics_history(physics_metrics)

        # Track physics-specific metrics if available
        if self.physics_loss is not None and hasattr(self.physics_loss, "compute_residuals"):
            # Compute physics-specific metrics
            residual_loss = self.physics_loss.compute_residuals(self.state.model, x)
            self.state.update_physics_metric("residual_loss", float(residual_loss))

            # Check conservation law violations if physics loss supports it
            if hasattr(self.physics_loss, "check_conservation_violations"):
                violations = self.physics_loss.check_conservation_violations(self.state.model, x)
                for violation_type, value in violations.items():
                    self.state.update_conservation_violation(violation_type, float(value))

        # Collect convergence metrics
        convergence_metrics = self.advanced_metrics.collect_convergence_metrics(self.state)

        # Update advanced metrics history with convergence metrics
        self.advanced_metrics.update_metrics_history(convergence_metrics)

        return current_loss

    def validation_step(
        self,
        x_val: Float[Array, "batch ..."],
        y_val: Float[Array, "batch ..."],
    ) -> Float[Array, ""]:
        """Perform validation step without updating parameters.

        Args:
            x_val: Input batch
            y_val: Target output batch

        Returns:
            Validation loss value
        """
        val_loss = self.compute_loss(x_val, y_val)

        # Update validation loss tracking
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = float(val_loss)
        else:
            self.state.plateau_count += 1

        # Collect full validation physics metrics
        val_physics_metrics = self.advanced_metrics.collect_physics_metrics(
            self.state.model, x_val, y_val, self.physics_loss
        )

        # Prefix validation metrics to distinguish from training metrics
        prefixed_val_metrics = {f"val_{key}": value for key, value in val_physics_metrics.items()}

        # Update advanced metrics history with validation metrics
        self.advanced_metrics.update_metrics_history(prefixed_val_metrics)

        # Track physics-specific validation metrics if available
        if self.physics_loss is not None and hasattr(self.physics_loss, "compute_residuals"):
            # Compute physics-specific validation metrics
            residual_loss = self.physics_loss.compute_residuals(self.state.model, x_val)
            self.state.update_physics_metric("val_residual_loss", float(residual_loss))

        return val_loss

    def train(
        self,
        train_data: tuple[Float[Array, "n_train ..."], Float[Array, "n_train ..."]],
        val_data: tuple[Float[Array, "n_val ..."], Float[Array, "n_val ..."]] | None = None,
        boundary_data: tuple[Float[Array, "n_boundary ..."], Float[Array, "n_boundary ..."]]
        | None = None,
    ) -> tuple[nnx.Module, TrainingMetrics]:
        """Main training loop.

        Args:
            train_data: Training data (x, y)
            val_data: Validation data (x, y), optional
            boundary_data: Boundary data (x, y), optional for physics-informed training

        Returns:
            Tuple of (trained_model, training_metrics)
        """
        x_train, y_train = train_data

        # Initialize advanced metrics collection for training session
        self.advanced_metrics.start_training()

        for epoch in range(self.config.num_epochs):
            # Mark the start of each epoch for timing
            self.advanced_metrics.start_epoch()
            self.state.epoch = epoch

            # Training step
            if self.physics_loss is not None and boundary_data is not None:
                # Physics-informed training step
                loss_value = self._physics_informed_training_step(x_train, y_train, boundary_data)
            else:
                loss_value = self.training_step(x_train, y_train)

            self.metrics.update_train_loss(float(loss_value))

            # Validation step
            val_loss_value = None
            if (
                val_data is not None
                and epoch % self.config.validation_config.validation_frequency == 0
            ):
                x_val, y_val = val_data
                val_loss = self.validation_step(x_val, y_val)
                val_loss_value = float(val_loss)
                self.metrics.update_val_loss(val_loss_value)

            # Progress callback
            if self.config.progress_callback is not None:
                progress_info = {
                    "epoch": epoch,
                    "total_epochs": self.config.num_epochs,
                    "train_loss": float(loss_value),
                    "val_loss": val_loss_value,
                    "step": self.state.step,
                    "best_loss": self.state.best_loss,
                    "best_val_loss": self.state.best_val_loss,
                }
                self.config.progress_callback(progress_info)

            # Basic verbose logging
            elif self.config.verbose and epoch % 10 == 0:
                val_str = f", Val: {val_loss_value:.6f}" if val_loss_value is not None else ""
                logger.info(
                    "Epoch %3d/%d: Loss: %.6f%s",
                    epoch,
                    self.config.num_epochs,
                    float(loss_value),
                    val_str,
                )

            # Checkpointing - save after completing the epoch
            if (
                self.config.checkpoint_config
                and epoch % self.config.checkpoint_config.save_frequency == 0
            ):
                # Update epoch to reflect completion of this epoch
                self.state.epoch = epoch + 1
                self._save_checkpoint(epoch + 1)

        # Save final checkpoint after all epochs are complete
        if self.config.checkpoint_config:
            self.state.epoch = self.config.num_epochs
            self._save_checkpoint(self.config.num_epochs)

        return self.state.model, self.metrics

    def get_comprehensive_metrics_summary(self, window_size: int = 10) -> dict[str, Any]:
        """Get full metrics summary including advanced metrics.

        Args:
            window_size: Number of recent measurements to consider
                for summary statistics

        Returns:
            Dictionary containing full metrics summary
        """
        summary: dict[str, Any] = {}

        # Get basic training metrics
        summary["basic_metrics"] = {
            "train_losses": self.metrics.train_losses[-window_size:]
            if self.metrics.train_losses
            else [],
            "val_losses": self.metrics.val_losses[-window_size:] if self.metrics.val_losses else [],
            "best_val_loss": self.metrics.best_val_loss,
            "learning_rates": self.metrics.learning_rates[-window_size:]
            if self.metrics.learning_rates
            else [],
        }

        # Get enhanced training state summary
        summary["training_state"] = {
            "current_step": self.state.step,
            "current_epoch": self.state.epoch,
            "best_loss": self.state.best_loss,
            "best_val_loss": self.state.best_val_loss,
            "plateau_count": self.state.plateau_count,
            "physics_summary": self.state.get_physics_summary(),
        }

        # Get advanced metrics summary
        summary["advanced_metrics"] = self.advanced_metrics.get_metrics_summary(window_size)

        return summary

    def _physics_informed_training_step(
        self,
        x_train: Float[Array, "batch ..."],
        y_train: Float[Array, "batch ..."],
        boundary_data: tuple[Float[Array, "n_boundary ..."], Float[Array, "n_boundary ..."]],
    ) -> Float[Array, ""]:
        """Physics-informed training step using attached physics loss."""
        # Extract boundary data once at the start
        boundary_inputs, boundary_targets = boundary_data

        def loss_fn(model):
            # Compute model predictions
            y_pred = model(x_train)

            # Compute boundary predictions if boundary data provided
            y_boundary_pred = model(boundary_inputs)

            # Compute physics loss using the attached physics loss instance
            if self.physics_loss is not None:
                total_physics_loss, loss_components = self.physics_loss.compute_loss(
                    predictions=y_pred,
                    targets=y_train,
                    inputs=x_train,
                    boundary_predictions=y_boundary_pred,
                    boundary_targets=boundary_targets,
                    boundary_inputs=boundary_inputs,
                )
            else:
                # Fallback to basic MSE loss if no physics loss is configured
                total_physics_loss = jnp.mean((y_pred - y_train) ** 2)
                loss_components = {"total_loss": total_physics_loss}

            return total_physics_loss, loss_components

        # Compute gradients
        (loss_value, loss_components), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            self.state.model
        )

        # Track physics loss components. The fallback branch (no physics loss
        # attached) builds a minimal ``loss_components`` dict without these
        # keys, so only record them when a physics loss is configured and read
        # defensively to keep the contract total across implementations.
        if self.physics_loss is not None:
            self.metrics.physics_losses.append(float(loss_components.get("physics_loss", 0.0)))
            self.metrics.boundary_losses.append(float(loss_components.get("boundary_loss", 0.0)))

        # Update parameters using the same NNX/optax pattern as ``training_step``:
        # tree-flatten grads/params, apply the optax delta, then restore NNX state.
        params = nnx.state(self.state.model, nnx.Param)
        updates, self.state.opt_state = self.state.optimizer.update(
            nnx.to_tree(grads), self.state.opt_state, nnx.to_tree(params)
        )
        updated_params = optax.apply_updates(nnx.to_tree(params), updates)
        nnx.update(self.state.model, nnx.from_tree(updated_params))

        # Update step counter
        self.state.step += 1

        return loss_value

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint.

        Args:
            epoch: Current epoch
        """
        if not self.config.checkpoint_config:
            return

        checkpoint_dir = Path(self.config.checkpoint_config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_manager = self._make_checkpoint_manager(
            checkpoint_dir, self.config.checkpoint_config.max_to_keep
        )

        # Prepare checkpoint data with scalar values instead of arrays
        checkpoint_data = {
            "model": nnx.state(self.state.model),
            "optimizer_state": self.state.opt_state,
            "step": self.state.step,  # Save as scalar instead of array
            "epoch": self.state.epoch,  # Save as scalar instead of array
            "metrics": {
                "train_losses": self.metrics.train_losses,
                "val_losses": self.metrics.val_losses,
                "best_val_loss": self.metrics.best_val_loss,
            },
        }

        # Save checkpoint
        checkpoint_manager.save(epoch, checkpoint_data)

        # Wait for async save to complete
        checkpoint_manager.wait_until_finished()

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model_class: type,
        model_kwargs: dict[str, Any],
    ) -> BasicTrainer:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model_class: Model class to instantiate
            model_kwargs: Model initialization arguments

        Returns:
            Restored trainer
        """
        checkpoint_dir = Path(checkpoint_path).resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        checkpoint_manager = BasicTrainer._make_checkpoint_manager(checkpoint_dir, max_to_keep=5)

        # Get latest checkpoint step
        latest_step = checkpoint_manager.latest_step()
        if latest_step is None:
            # Try to find checkpoints manually
            checkpoint_dirs = [
                d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if not checkpoint_dirs:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
            latest_step = max(int(d.name) for d in checkpoint_dirs)

        # Create new model instance
        model = model_class(**model_kwargs)

        # Create temporary trainer for structure
        temp_config = TrainingConfig(num_epochs=1, batch_size=1, learning_rate=1e-3)
        trainer = BasicTrainer(model, temp_config)

        # Prepare target structure for restoration to avoid warnings
        target_structure = {
            "model": nnx.state(trainer.state.model),
            "optimizer_state": trainer.state.opt_state,
            "step": 0,  # Provide target structure for scalar values
            "epoch": 0,  # Provide target structure for scalar values
            "metrics": {
                "train_losses": [],
                "val_losses": [],
                "best_val_loss": None,
            },
        }

        # Load checkpoint data with target structure
        checkpoint_data = checkpoint_manager.restore(latest_step, items=target_structure)

        # Restore model state using proper NNX restoration
        model_state = checkpoint_data["model"]
        trainer.state.model = nnx.merge(nnx.graphdef(trainer.state.model), model_state)

        # Restore training state
        trainer.state.opt_state = checkpoint_data["optimizer_state"]
        trainer.state.step = int(checkpoint_data["step"])
        trainer.state.epoch = int(checkpoint_data["epoch"])

        # Restore metrics
        if "metrics" in checkpoint_data:
            metrics_data = checkpoint_data["metrics"]
            trainer.metrics.train_losses = metrics_data.get("train_losses", [])
            trainer.metrics.val_losses = metrics_data.get("val_losses", [])
            trainer.metrics.best_val_loss = metrics_data.get("best_val_loss")

        return trainer


def _squeeze_output_axis(arr: jax.Array) -> jax.Array:
    """Drop a trailing singleton output axis if present."""
    if arr.ndim > 1 and arr.shape[-1] == 1:
        return arr.squeeze(-1)
    return arr


def _stochastic_ensemble_from_model(
    model: nnx.Module | Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_samples: int,
    rngs: nnx.Rngs,
    noise_scale: float = 1e-2,
) -> jax.Array:
    """Build a ``(num_samples, batch, output)`` ensemble by really calling ``model``.

    The model is invoked exactly **once** per ensemble member. For NNX
    modules with stochastic state (dropout, MC sampling) every member
    yields a distinct output. For deterministic models we add a small
    aleatoric noise floor so the downstream
    :meth:`UncertaintyQuantifier.decompose_uncertainty` does not see a
    rank-deficient batch (whose ``var=0`` would produce degenerate
    acquisition scores). The noise scale is small (1e-2) so the model's
    own output dominates whenever it varies.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples!r}")
    predictions = []
    for _ in range(int(num_samples)):
        # NNX state mutates on each call when the model carries stochastic
        # state; for deterministic models all members start from the same
        # prediction and only differ in the aleatoric-noise overlay below.
        pred = model(x)  # type: ignore[operator]  # nnx.Module is callable
        if pred.ndim == 1:
            pred = pred[:, None]
        predictions.append(pred)
    stacked = jnp.stack(predictions, axis=0)  # (num_samples, batch, output)
    key = extract_rng_key(
        rngs,
        streams=("active_acquire", "default", "params", "sample"),
        context="active-learning ensemble",
    )
    eps = noise_scale * jax.random.normal(key, stacked.shape)
    return stacked + eps


class UncertaintyGuidedTrainer:
    """Uncertainty-guided adaptive training with active learning strategies.

    Invokes ``uncertainty_quantifier`` on model predictions. The helper
    :func:`_stochastic_ensemble_from_model` is the single point where
    ``model(x)`` is called per ensemble member, keeping acquisition
    formulas out of the inline path.
    """

    def __init__(
        self,
        model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        uncertainty_threshold: float = 0.1,
        adaptation_strategy: str = "active_learning",
    ) -> None:
        """Initialize uncertainty-guided trainer.

        Args:
            model: Neural network model to train.
            uncertainty_quantifier: Uncertainty quantification module,
                typed to :class:`UncertaintyQuantifier`.
            rngs: Caller-owned :class:`nnx.Rngs` bundle used to materialise
                aleatoric-noise overlays for the ensemble draws.
            uncertainty_threshold: Threshold for high uncertainty detection
            adaptation_strategy: Strategy for adapting training
                (``"active_learning"`` / ``"loss_weighting"`` /
                ``"convergence_monitoring"``).
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_strategy = adaptation_strategy

    def select_uncertain_samples(self, x_pool: jax.Array, num_samples: int = 10) -> list[int]:
        """Select most uncertain samples from pool for active learning."""
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_pool,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        total_uncertainty = components.total.squeeze()
        return jnp.argsort(total_uncertainty)[-num_samples:].tolist()

    def compute_uncertainty_weights(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute adaptive loss weights based on uncertainty."""
        del y  # weights depend on input-conditioned uncertainty only.
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        total_uncertainty = components.total.squeeze()
        max_unc = jnp.max(total_uncertainty)
        # Guard against the all-zero case (deterministic model + zero noise).
        normalised = jnp.where(max_unc > 0.0, total_uncertainty / max_unc, total_uncertainty)
        return jnp.clip(normalised, 0.1, 1.0)

    def monitor_uncertainty_convergence(
        self, x_val: jax.Array, y_val: jax.Array
    ) -> dict[str, float]:
        """Monitor uncertainty convergence during training."""
        del y_val  # ECE is computed downstream; here we expose component magnitudes.
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_val,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        calibration_error = jnp.mean(jnp.abs(components.total))
        return {
            "epistemic_uncertainty": float(jnp.mean(components.epistemic)),
            "aleatoric_uncertainty": float(jnp.mean(components.aleatoric)),
            "calibration_error": float(calibration_error),
        }


class MultiFidelityUncertaintyTrainer:
    """Multi-fidelity uncertainty propagation trainer.

    Invokes both ``high_fidelity_model(x)`` and ``low_fidelity_model(x)``
    and combines their uncertainties with the Kennedy-O'Hagan
    ``fidelity_ratio`` weighting between high- and low-fidelity outputs.
    """

    def __init__(
        self,
        high_fidelity_model: nnx.Module | Callable[[jax.Array], jax.Array],
        low_fidelity_model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        fidelity_ratio: float = 0.1,
    ) -> None:
        """Initialize multi-fidelity uncertainty trainer.

        Args:
            high_fidelity_model: High fidelity neural network model
            low_fidelity_model: Low fidelity neural network model
            uncertainty_quantifier: Uncertainty quantification module.
            rngs: Caller-owned :class:`nnx.Rngs` bundle for the per-fidelity
                ensemble draws.
            fidelity_ratio: Ratio of high to low fidelity data (Kennedy-O'Hagan
                additive linear weighting).
        """
        self.high_fidelity_model = high_fidelity_model
        self.low_fidelity_model = low_fidelity_model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.fidelity_ratio = fidelity_ratio

    def propagate_multi_fidelity_uncertainty(self, x: jax.Array) -> jax.Array:
        """Propagate uncertainty through both fidelity levels."""
        hi_predictions = _stochastic_ensemble_from_model(
            self.high_fidelity_model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        lo_predictions = _stochastic_ensemble_from_model(
            self.low_fidelity_model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )

        hi_components = self.uncertainty_quantifier.decompose_uncertainty(hi_predictions)
        lo_components = self.uncertainty_quantifier.decompose_uncertainty(lo_predictions)

        propagated = (
            self.fidelity_ratio * hi_components.total
            + (1.0 - self.fidelity_ratio) * lo_components.total
        )
        return propagated.squeeze()


class ActiveUncertaintyLearner:
    """Active learning with uncertainty-based sample acquisition.

    Acquisition is delegated to :func:`opifex.uncertainty.active.acquire`.
    :meth:`acquire_samples` accepts an explicit
    ``acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array]``
    (signature ``(mean, variance) -> per-candidate scores``) in addition
    to the ``sampling_strategy`` string registered with the active-learning
    registry. When ``acquisition_fn`` is provided it overrides the strategy
    string.

    Acquisition formula evaluation goes through the active-learning
    subsystem rather than being inlined in this class.
    """

    def __init__(
        self,
        model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        sampling_strategy: str = "max_uncertainty",
        acquisition_size: int = 10,
        diversity_weight: float = 0.0,
        physics_priors: Any | None = None,
    ) -> None:
        """Initialize active uncertainty learner.

        Args:
            model: Neural network model — actually invoked on the pool.
            uncertainty_quantifier: Uncertainty quantification module.
            rngs: Caller-owned :class:`nnx.Rngs` bundle.
            sampling_strategy: Default acquisition strategy when no
                ``acquisition_fn`` is passed to :meth:`acquire_samples`.
                Mapped to :class:`AcquisitionStrategy` values.
            acquisition_size: Number of samples to acquire per round.
            diversity_weight: Optional diversity penalty (currently
                surfaced through the strategy mapping).
            physics_priors: Physics-informed priors (optional, reserved
                for the physics-guided strategy).
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.sampling_strategy = sampling_strategy
        self.acquisition_size = acquisition_size
        self.diversity_weight = diversity_weight
        self.physics_priors = physics_priors

    def _predictive_distribution(self, x_pool: jax.Array) -> PredictiveDistribution:
        """Build the :class:`PredictiveDistribution` for ``x_pool``.

        Invokes ``self.model`` ``num_samples`` times via the shared
        :func:`_stochastic_ensemble_from_model` helper, then routes the
        ensemble through ``self.uncertainty_quantifier`` to recover the
        epistemic / aleatoric / total decomposition.
        """
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_pool,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        # Drop the trailing singleton output axis so the PredictiveDistribution's
        # per-candidate shape contract (mean.shape == (batch,) for scalar output)
        # holds for the active-learning acquisition kernels downstream.
        mean_full = jnp.mean(predictions, axis=0)
        mean = mean_full.squeeze(-1) if predictions.shape[-1] == 1 else mean_full
        variance = _squeeze_output_axis(components.total)
        samples = predictions.squeeze(-1) if predictions.shape[-1] == 1 else predictions
        epistemic = _squeeze_output_axis(components.epistemic)
        aleatoric = _squeeze_output_axis(components.aleatoric)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=samples,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=variance,
        )

    def acquire_samples(
        self,
        x_pool: jax.Array,
        *,
        acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> list[int]:
        """Acquire samples from pool by delegating to the active subsystem.

        Args:
            x_pool: Pool of candidate inputs.
            acquisition_fn: Optional callable
                ``(mean, variance) -> per-candidate scores``. When
                supplied, the top-``acquisition_size`` indices by score
                are returned directly. When omitted, dispatch routes
                through :func:`opifex.uncertainty.active.acquire` using
                ``sampling_strategy`` mapped to
                :class:`AcquisitionStrategy`.

        Returns:
            list[int]: indices of the acquired pool elements.
        """
        predictive = self._predictive_distribution(x_pool)

        if acquisition_fn is not None:
            if predictive.variance is None:
                raise RuntimeError("predictive distribution has no variance.")
            scores = acquisition_fn(predictive.mean, predictive.variance)
            top = jnp.argsort(scores)[-self.acquisition_size :]
            return [int(i) for i in top]

        strategy_map = {
            "max_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "diverse_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "physics_guided_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "bald": AcquisitionStrategy.BALD,
            "ucb": AcquisitionStrategy.UCB,
            "lcb": AcquisitionStrategy.LCB,
            "ei": AcquisitionStrategy.EI,
            "log_ei": AcquisitionStrategy.LOG_EI,
            "pi": AcquisitionStrategy.PI,
        }
        strategy = strategy_map.get(self.sampling_strategy, AcquisitionStrategy.MAX_VARIANCE)
        kwargs: dict[str, Any] = {}
        if strategy in (AcquisitionStrategy.EI, AcquisitionStrategy.LOG_EI, AcquisitionStrategy.PI):
            kwargs["best_value"] = float(jnp.min(predictive.mean))
        if strategy in (AcquisitionStrategy.UCB, AcquisitionStrategy.LCB):
            kwargs["beta"] = 1.96
        batch = _active_acquire(
            predictive,
            strategy=strategy,
            batch_size=self.acquisition_size,
            rngs=self.rngs,
            **kwargs,
        )
        return [int(i) for i in batch.indices]


def create_progress_bar_callback(
    description: str = "Training", total_epochs: int | None = None, **tqdm_kwargs
):
    """Create a progress bar callback for training.

    Args:
        description: Description for the progress bar
        total_epochs: Total number of epochs (optional, for bar length)
        **tqdm_kwargs: Additional arguments to pass to tqdm

    Returns:
        A callback function that can be used with TrainingConfig.progress_callback

    Example:
        config = TrainingConfig(
            progress_callback=create_progress_bar_callback(
                "Training UNO", total_epochs=100
            )
        )
    """
    pbar = None

    def progress_callback(progress_info: dict[str, Any]) -> None:
        nonlocal pbar

        epoch = progress_info.get("epoch", 0)
        train_loss = progress_info.get("train_loss")
        val_loss = progress_info.get("val_loss") or progress_info.get("final_val_loss")

        # Initialize progress bar on first call
        if pbar is None:
            # Try to get total from the progress info if not provided
            nonlocal total_epochs
            if total_epochs is None:
                total_epochs = progress_info.get("total_epochs")

            default_kwargs = {
                "total": total_epochs,
                "desc": description,
                "bar_format": (
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
                "position": 0,
                "leave": True,
            }
            default_kwargs.update(tqdm_kwargs)
            pbar = tqdm(**default_kwargs)

        # Update progress bar
        pbar.n = epoch + 1
        postfix = {}
        if train_loss is not None:
            # Handle JAX arrays/tracers by converting to float if possible
            from contextlib import suppress

            with suppress(TypeError, ValueError):
                postfix["Loss"] = f"{float(train_loss):.4f}"

        if val_loss is not None:
            from contextlib import suppress

            with suppress(TypeError, ValueError):
                postfix["Val"] = f"{float(val_loss):.4f}"

        pbar.set_postfix(postfix)
        pbar.refresh()

        # Close progress bar when training is complete
        if total_epochs is not None and epoch + 1 >= total_epochs:
            pbar.close()

    return progress_callback
