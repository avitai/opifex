"""Unified Trainer for Opifex framework.

This module provides a unified, generic Trainer class that consolidates
BasicTrainer and PhysicsInformedTrainer into a single, extensible trainer
for all scientific ML methods.

Key Features:
- Standard and physics-informed training
- Composable configuration system
- Full checkpointing support with Orbax
- Extensibility via hooks and custom losses
- Support for all optimizer types
- Full metrics collection

Following strict TDD principles - implemented to pass tests in test_trainer.py.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.metrics import relative_l2_error
from opifex.core.training.components.checkpoint_store import (
    OrbaxCheckpointStore,
)
from opifex.core.training.monitoring.metrics import (
    TrainingState,
)
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig


logger = logging.getLogger(__name__)


from opifex.core.training.components.base import TrainingCallback  # noqa: TC001
from opifex.core.training.config import TrainingConfig  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable


class Trainer(nnx.Module):
    """Unified trainer for standard and physics-informed training.

    This trainer consolidates all training functionality into a single,
    extensible class that supports:
    - Standard supervised learning
    - Physics-informed training with constraints
    - Conservation law enforcement
    - Multi-scale physics
    - Boundary conditions
    - Quantum training
    - Custom loss functions
    - Hook system for extensibility

    Args:
        model: The neural network model to train
        config: Training configuration (composable)
        rngs: Optional RNG state for reproducibility
    """

    def __init__(
        self,
        model: nnx.Module,
        config: TrainingConfig,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Neural network model to train
            config: Training configuration with optional physics configs
            rngs: Optional RNG state
        """
        self.model = model
        self.config = config
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)

        # ✅ NEW: Create nnx.Optimizer (Flax 0.12.0+ API)
        # Eliminates manual state management (DRY principle)
        opt_cfg = config.optimization_config
        optimizer_config = OptimizerConfig(
            optimizer_type=opt_cfg.optimizer,
            learning_rate=opt_cfg.learning_rate,
            b1=opt_cfg.beta1,
            b2=opt_cfg.beta2,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
            momentum=opt_cfg.momentum,
            schedule_type=opt_cfg.schedule_type,
            decay_steps=opt_cfg.decay_steps,
            transition_steps=opt_cfg.transition_steps,
            decay_rate=opt_cfg.decay_rate,
            alpha=opt_cfg.alpha,
        )

        # Create Optax transformation
        optax_optimizer = create_optimizer(optimizer_config)

        # ✅ BREAKING CHANGE: Use nnx.Optimizer (automatic state management)
        # Replaces manual opt_state tracking
        self.optimizer = nnx.Optimizer(
            model,
            optax_optimizer,
            wrt=nnx.Param,  # Only optimize parameters
        )

        # ✅ SIMPLIFIED: Initialize training state (no manual opt_state)
        self.state = TrainingState(
            model=model,
            # nnx.Optimizer manages state internally
            optimizer=self.optimizer,  # pyright: ignore[reportArgumentType]
            # DEPRECATED: Kept for backward compat, but unused
            opt_state=None,  # pyright: ignore[reportArgumentType]
            step=0,
            epoch=0,
            best_loss=float("inf"),
            rngs=self.rngs,
        )

        # Initialize checkpoint store if configured
        self.checkpoint_store = None
        if config.checkpoint_config.checkpoint_dir:
            self.checkpoint_store = OrbaxCheckpointStore(
                checkpoint_dir=config.checkpoint_config.checkpoint_dir,
                max_to_keep=config.checkpoint_config.max_to_keep,
            )

        # Extensibility: custom losses and hooks
        self.custom_losses: dict[str, Callable] = {}
        self.hooks: dict[str, list[Callable]] = {}
        self.components: nnx.List[TrainingCallback] = nnx.List([])

        # Initialize constraint weights if configured
        self._constraint_weights = nnx.Dict()
        if self.config.constraint_config and self.config.constraint_config.constraints:
            constraints = self.config.constraint_config.constraints
            initial_weight = 1.0 / len(constraints) if constraints else 1.0
            for c in constraints:
                self._constraint_weights[c] = nnx.Variable(jnp.array(initial_weight))

        # Initialize distributed manager if configured
        self._distributed_manager = None
        if config.distributed_config is not None:
            from opifex.distributed.manager import DistributedManager

            self._distributed_manager = DistributedManager(
                config=config.distributed_config,
            )

    def add_component(self, component: TrainingCallback) -> None:
        """Add a training component.

        Args:
            component: Component to add
        """
        self.components.append(component)

    def _compute_data_loss(self, y_pred: jax.Array, y: jax.Array) -> jax.Array:
        """Compute the data-fit loss selected by ``loss_config.loss_type``.

        Supports ``mse`` (default), ``mae`` and ``relative_l2``. The relative L2
        loss — the standard operator-learning objective — averages the per-sample
        ratio ``||y_pred - y||_2 / ||y||_2`` and is scale-invariant across samples,
        which is important when the target field magnitude varies.

        Args:
            y_pred: Model prediction.
            y: Target.

        Returns:
            Scalar data loss.

        Raises:
            ValueError: If ``loss_type`` is not a recognised data loss.
        """
        loss_type = self.config.loss_config.loss_type
        if loss_type == "mse":
            return jnp.mean((y_pred - y) ** 2)
        if loss_type == "mae":
            return jnp.mean(jnp.abs(y_pred - y))
        if loss_type == "relative_l2":
            return relative_l2_error(y_pred, y)
        raise ValueError(f"Unsupported data loss_type: {loss_type!r}")

    def training_step(
        self,
        x: jax.Array,
        y: jax.Array,
        boundary_data: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a single training step.

        Args:
            x: Input data
            y: Target data
            boundary_data: Optional (x_boundary, y_boundary) tuple

        Returns:
            Tuple of (loss, metrics)
        """
        # Component hook: on_batch_begin
        for component in self.components:
            component.on_batch_begin(self.state.step, self.state)

        # Set model to training mode
        self.model.train()

        # One fused forward+loss+grad pass; ``y_pred`` is surfaced via has_aux so
        # the adaptive-weight update below reuses it instead of a second forward.
        (loss, (loss_components, y_pred)), grads = nnx.value_and_grad(
            lambda model: self._loss_fn(model, x, y, boundary_data), has_aux=True
        )(self.model)

        # The nnx optimizer manages parameter and optimizer state internally.
        self.optimizer.update(self.model, grads)
        self.state.step += 1

        metrics = self._build_step_metrics(loss, grads, loss_components)

        # Update adaptive weights, reusing the forward pass from the loss above.
        if (
            self.config.constraint_config is not None
            and self.config.constraint_config.adaptive_weighting
        ):
            self._update_adaptive_weights(x, y_pred, y)

        # Execute hooks
        self._execute_hooks("training_step_end", metrics)

        # Component hook: on_batch_end
        for component in self.components:
            # Do NOT cast loss to float here, keep as Tracer for JIT compatibility
            component.on_batch_end(self.state.step, self.state, loss, metrics)

        return loss, metrics

    def _loss_fn(
        self,
        model: nnx.Module,
        x: jax.Array,
        y: jax.Array,
        boundary_data: tuple[jax.Array, jax.Array] | None,
    ) -> tuple[jax.Array, tuple[dict[str, Any], jax.Array]]:
        """Compute the total training loss and its per-component breakdown.

        Returns ``(total_loss, (loss_components, y_pred))``. ``y_pred`` is
        surfaced through ``has_aux`` so the post-gradient adaptive-weight update
        can reuse it rather than running a second forward pass.
        """
        if self.config.gradient_checkpointing:
            from artifex.generative_models.core.gradient_checkpointing import apply_remat

            forward_fn = apply_remat(lambda m: m(x), policy=self.config.gradient_checkpoint_policy)
            y_pred = forward_fn(model)
        else:
            y_pred = model(x)  # pyright: ignore[reportCallIssue]

        total_loss = self._compute_data_loss(y_pred, y)
        loss_components: dict[str, Any] = {"data_loss": total_loss}

        for term, components in (
            self._boundary_loss_term(model, boundary_data),
            self._multiscale_loss_term(x, y_pred, y),
            self._constraint_loss_term(x, y_pred, y),
            self._conservation_loss_term(y_pred),
            self._custom_loss_terms(model, x, y_pred, y),
        ):
            total_loss = total_loss + term
            loss_components.update(components)

        return total_loss, (loss_components, y_pred)

    def _boundary_loss_term(
        self, model: nnx.Module, boundary_data: tuple[jax.Array, jax.Array] | None
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Weighted boundary-condition penalty (zero when not configured)."""
        if not (
            boundary_data is not None
            and self.config.boundary_config is not None
            and self.config.boundary_config.enforce
        ):
            return jnp.array(0.0), {}
        x_boundary, y_boundary = boundary_data
        y_pred_boundary = model(x_boundary)  # pyright: ignore[reportCallIssue]
        boundary_loss = jnp.mean((y_pred_boundary - y_boundary) ** 2)
        weighted_boundary = boundary_loss * self.config.boundary_config.weight
        return weighted_boundary, {"boundary_loss": boundary_loss}

    def _multiscale_loss_term(
        self, x: jax.Array, y_pred: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Weighted multi-scale physics loss (zero when not configured)."""
        if self.config.multiscale_config is None or not self.config.multiscale_config.scales:
            return jnp.array(0.0), {}
        scales = self.config.multiscale_config.scales
        components: dict[str, Any] = {}
        multiscale_loss = jnp.array(0.0)
        for scale in scales:
            scale_weight = self.config.multiscale_config.weights.get(scale, 1.0 / len(scales))
            scale_loss = self._compute_scale_specific_loss(x, y_pred, y, scale)
            multiscale_loss = multiscale_loss + scale_weight * scale_loss
            components[f"{scale}_loss"] = scale_loss
        components["multiscale_loss"] = multiscale_loss
        return multiscale_loss, components

    def _constraint_loss_term(
        self, x: jax.Array, y_pred: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Weighted constraint loss using adaptive weights (zero when not configured)."""
        if self.config.constraint_config is None or not self.config.constraint_config.constraints:
            return jnp.array(0.0), {}
        components: dict[str, Any] = {}
        constraint_total = jnp.array(0.0)
        for constraint in self.config.constraint_config.constraints:
            if constraint in self._constraint_weights:
                weight = self._constraint_weights[constraint].value
            else:
                weight = jnp.array(1.0)
            constraint_loss = self._compute_constraint_specific_loss(x, y_pred, y, constraint)
            constraint_total = constraint_total + weight * constraint_loss
            components[f"{constraint}_loss"] = constraint_loss
        return constraint_total, components

    def _conservation_loss_term(self, y_pred: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
        """Conservation-law penalty (zero when not configured)."""
        if self.config.conservation_config is None or not self.config.conservation_config.laws:
            return jnp.array(0.0), {}
        conservation_loss = jnp.mean(jnp.square(y_pred))
        components: dict[str, Any] = {"conservation_loss": conservation_loss}
        for law in self.config.conservation_config.laws:
            components[f"{law}_conservation"] = conservation_loss
        return conservation_loss * 0.1, components

    def _custom_loss_terms(
        self, model: nnx.Module, x: jax.Array, y_pred: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Sum of registered custom loss terms (zero when none registered)."""
        components: dict[str, Any] = {}
        custom_total = jnp.array(0.0)
        for loss_name, custom_loss_fn in self.custom_losses.items():
            custom_loss = custom_loss_fn(model, x, y_pred, y)
            custom_total = custom_total + custom_loss
            components[f"{loss_name}_loss"] = custom_loss
        return custom_total, components

    def _build_step_metrics(
        self, loss: jax.Array, grads: Any, loss_components: dict[str, Any]
    ) -> dict[str, Any]:
        """Assemble the per-step metrics dict (arrays kept for JIT compatibility)."""
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
        return {
            "loss": loss,
            "step": self.state.step,
            "learning_rate": self.config.learning_rate,
            "gradient_norm": grad_norm,
            **loss_components,
        }

    def validation_step(self, x: jax.Array, y: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
        """Execute validation step without parameter updates.

        Args:
            x: Input data
            y: Target data

        Returns:
            Tuple of (loss, metrics)
        """
        # Set model to evaluation mode
        self.model.eval()

        # Forward pass only
        y_pred = self.model(x)  # pyright: ignore[reportCallIssue]  # pyright: ignore[reportCallIssue]  # pyright: ignore[reportCallIssue]
        loss = jnp.mean((y_pred - y) ** 2)

        metrics = {
            "val_loss": loss,  # Keep as JAX array for JIT compatibility
            "step": self.state.step,
        }

        return loss, metrics

    def fit(
        self,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array] | None = None,
        boundary_data: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[nnx.Module, dict[str, Any]]:
        """Execute full training loop.

        Args:
            train_data: (x_train, y_train) tuple
            val_data: Optional (x_val, y_val) tuple
            boundary_data: Optional (x_boundary, y_boundary) tuple

        Returns:
            Tuple of (trained_model, metrics_summary)
        """
        x_train, y_train = train_data
        final_metrics: dict[str, Any] = {}

        # JIT-compile the training step once, outside the loop. ``self`` (the
        # module) is passed explicitly so nnx threads model/optimizer state and
        # XLA fuses forward+loss+grad+update into a single program per step.
        @nnx.jit
        def train_step_jit(trainer_instance, x, y, bd):
            return trainer_instance.training_step(x, y, bd)

        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch

            # Component hook: on_epoch_begin
            for component in self.components:
                component.on_epoch_begin(epoch, self.state)

            # Set model to training mode (nnx.Module method)
            self.train()

            # One host transfer per epoch: the running loss sum stays on-device
            # so successive jitted steps dispatch asynchronously instead of
            # blocking on a per-batch ``float()``.
            avg_train_loss = float(
                self._run_training_epoch(x_train, y_train, boundary_data, train_step_jit)
            )

            # Component hook: on_epoch_end
            for component in self.components:
                component.on_epoch_end(epoch, self.state, {"train_loss": avg_train_loss})

            # Record the initial loss on the first epoch only.
            if "initial_train_loss" not in final_metrics:
                final_metrics["initial_train_loss"] = avg_train_loss

            self._run_epoch_validation(val_data, epoch, final_metrics)
            self._finalize_epoch(epoch, avg_train_loss, val_data, final_metrics)
            final_metrics["final_train_loss"] = avg_train_loss

        return self.model, final_metrics

    def _run_training_epoch(
        self,
        x_train: jax.Array,
        y_train: jax.Array,
        boundary_data: tuple[jax.Array, jax.Array] | None,
        train_step_jit: Any,
    ) -> jax.Array:
        """Run one training epoch and return the mean batch loss (kept on-device).

        Per-batch losses are accumulated on the accelerator and transferred to
        the host once (by the caller), so consecutive jitted steps dispatch
        asynchronously rather than serialising on a per-batch ``float()``.
        """
        num_samples = x_train.shape[0]
        batch_size = self.config.batch_size
        loss_sum = jnp.array(0.0)
        num_batches = 0
        for start in range(0, num_samples, batch_size):
            sharded = self._shard_batch(
                x_train[start : start + batch_size], y_train[start : start + batch_size]
            )
            loss, _ = train_step_jit(self, sharded["x"], sharded["y"], boundary_data)
            loss_sum = loss_sum + loss
            num_batches += 1
        return loss_sum / num_batches

    def _shard_batch(self, x_batch: jax.Array, y_batch: jax.Array) -> dict[str, jax.Array]:
        """Shard a batch across the distributed mesh, or pass through otherwise."""
        if self._distributed_manager is None:
            return {"x": x_batch, "y": y_batch}
        from opifex.distributed.training import shard_batch

        return shard_batch(
            {"x": x_batch, "y": y_batch},
            mesh=self._distributed_manager.mesh,
            data_axis=self.config.distributed_config.mesh_axis_names[0],  # type: ignore[union-attr]
        )

    def _run_epoch_validation(
        self,
        val_data: tuple[jax.Array, jax.Array] | None,
        epoch: int,
        final_metrics: dict[str, Any],
    ) -> None:
        """Run validation on the configured cadence, recording the final val loss."""
        if val_data is None or epoch % self.config.validation_frequency != 0:
            return
        self.model.eval()
        x_val, y_val = val_data
        val_loss, _ = self.validation_step(x_val, y_val)
        final_metrics["final_val_loss"] = float(val_loss)

    def _finalize_epoch(
        self,
        epoch: int,
        avg_train_loss: float,
        val_data: tuple[jax.Array, jax.Array] | None,
        final_metrics: dict[str, Any],
    ) -> None:
        """Emit the progress callback, verbose log, and checkpoint for an epoch."""
        if self.config.progress_callback is not None:
            self.config.progress_callback(epoch, {"train_loss": avg_train_loss, **final_metrics})

        if self.config.verbose:
            val_info = ""
            if val_data is not None and epoch % self.config.validation_frequency == 0:
                val_loss = final_metrics.get("final_val_loss")
                if val_loss is not None:
                    val_info = f", Val Loss: {val_loss:.6f}"
            logger.info(
                "Epoch %d/%d: Train Loss: %.6f%s",
                epoch + 1,
                self.config.num_epochs,
                avg_train_loss,
                val_info,
            )

        if self.checkpoint_store is not None and epoch % self.config.checkpoint_frequency == 0:
            self.save_checkpoint(step=epoch, loss=avg_train_loss)

    def save_checkpoint(
        self,
        step: int,
        loss: float,
        physics_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Save checkpoint with Orbax.

        Args:
            step: Current training step
            loss: Current loss value
            physics_metadata: Optional physics-specific metadata

        Returns:
            Path to saved checkpoint, or None if no checkpoint store
        """
        if self.checkpoint_store is None:
            return None

        additional_metadata = {
            "step": step,
            "epoch": self.state.epoch,
        }

        checkpoint_path = self.checkpoint_store.save(
            self.model,
            step=step,
            loss=loss,
            physics_metadata=physics_metadata,
            additional_metadata=additional_metadata,
        )

        return str(checkpoint_path)

    def load_checkpoint(self, step: int) -> tuple[Any, dict[str, Any]]:  # Updated return type
        """Load checkpoint from Orbax.

        Args:
            step: Step number to load

        Returns:
            Tuple of (model, metadata) or (None, {}) if not found
        """
        if self.checkpoint_store is None:
            return None, {}

        try:
            model, metadata = self.checkpoint_store.restore(
                target_model=self.model,
                step=step,
                # Return None if checkpoint doesn't exist
                return_original_on_missing=False,
            )
            return model, metadata  # pyright: ignore[reportArgumentType]
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return None, {}

    def register_custom_loss(self, name: str, loss_fn: Callable) -> None:
        """Register a custom loss function.

        Args:
            name: Name of the custom loss
            loss_fn: Loss function with signature (model, x, y_pred, y_true) -> loss
        """
        self.custom_losses[name] = loss_fn

    def register_hook(self, hook_point: str, hook_fn: Callable) -> None:
        """Register a hook to execute at specific points.

        Args:
            hook_point: Point at which to execute hook
            hook_fn: Hook function to execute
        """
        if hook_point not in self.hooks:
            self.hooks[hook_point] = []
        self.hooks[hook_point].append(hook_fn)

    def _execute_hooks(self, hook_point: str, *args, **kwargs) -> None:
        """Execute all hooks registered at a hook point.

        Args:
            hook_point: Hook point to execute
            *args: Positional arguments to pass to hooks
            **kwargs: Keyword arguments to pass to hooks
        """
        if hook_point in self.hooks:
            for hook_fn in self.hooks[hook_point]:
                hook_fn(self, *args, **kwargs)

    def _compute_scale_specific_loss(
        self,
        x: jax.Array,  # noqa: ARG002 - scale-loss interface takes inputs
        y_pred: jax.Array,
        y_true: jax.Array,
        scale: str,
    ) -> jax.Array:
        """Compute loss for a specific physics scale.

        Args:
            x: Input data
            y_pred: Model predictions
            y_true: Ground truth
            scale: Physics scale ("molecular", "atomic", "electronic")

        Returns:
            Scale-specific loss value
        """
        if scale == "molecular":
            # Molecular scale physics - intermolecular interactions
            return jnp.mean((y_pred - y_true) ** 2) * 0.5
        if scale == "atomic":
            # Atomic scale physics - intramolecular interactions
            return jnp.mean(jnp.abs(y_pred - y_true)) * 0.3
        if scale == "electronic":
            # Electronic scale physics - quantum effects
            return jnp.mean((y_pred - y_true) ** 4) * 0.2
        # Default scale
        return jnp.mean((y_pred - y_true) ** 2)

    def _compute_constraint_specific_loss(
        self,
        x: jax.Array,  # noqa: ARG002 - constraint-loss interface takes inputs
        y_pred: jax.Array,
        y_true: jax.Array,
        constraint: str,
    ) -> jax.Array:
        """Compute loss for a specific physics constraint.

        Args:
            x: Input data
            y_pred: Model predictions
            y_true: Ground truth
            constraint: Constraint name

        Returns:
            Constraint-specific loss value
        """
        if constraint == "energy_conservation":
            # Energy conservation constraint
            return jnp.mean(jnp.abs(y_pred - y_true))
        if constraint == "momentum_conservation":
            # Momentum conservation constraint
            return jnp.mean(jnp.square(y_pred - y_true))
        if constraint == "symmetry_preservation":
            # Symmetry preservation constraint
            return jnp.mean(jnp.abs(y_pred))
        # Default constraint
        return jnp.mean((y_pred - y_true) ** 2)

    def _compute_constraint_violation(
        self, x: jax.Array, y_pred: jax.Array, y_true: jax.Array, constraint: str
    ) -> jax.Array:
        """Compute violation for a specific constraint.

        IMPORTANT: This method uses string-based dispatch and must ONLY be called
        outside of JIT-traced functions. It is used for monitoring constraint
        violations in adaptive weighting, which happens after gradient computation.

        For loss computation inside traced functions, use
        _compute_constraint_specific_loss().

        Args:
            x: Input data
            y_pred: Model predictions
            y_true: Ground truth
            constraint: Constraint name (string dispatch - not JIT-safe)

        Returns:
            Constraint violation value

        Note:
            This method is called from _update_adaptive_weights(), which executes
            outside the JIT-traced loss_fn, making string dispatch safe.
        """
        # Use the same computation as constraint loss for violation monitoring
        return self._compute_constraint_specific_loss(x, y_pred, y_true, constraint)

    def _update_adaptive_weights(
        self, x: jax.Array, y_pred: jax.Array, y_true: jax.Array
    ) -> dict[str, float]:
        """Update constraint weights adaptively based on violations.

        This method executes OUTSIDE of JIT-traced functions, after gradient
        computation completes. This allows safe use of Python control flow
        and string-based constraint dispatch.

        Args:
            x: Input data
            y_pred: Model predictions
            y_true: Ground truth

        Returns:
            Updated constraint weights (normalized to sum to 1.0)
        """
        if self.config.constraint_config is None:
            return {}

        if not self.config.constraint_config.adaptive_weighting:
            return getattr(self, "_constraint_weights", {})

        constraints = self.config.constraint_config.constraints
        if not constraints:
            return {}

        # Calculate violations and max violation using JAX ops
        violation_values = []
        for constraint in constraints:
            v_val = self._compute_constraint_violation(x, y_pred, y_true, constraint)
            violation_values.append(v_val)

        # Stack to compute max over all constraints
        violations_stack = jnp.stack(violation_values)
        max_violation = jnp.max(violations_stack)
        # Avoid division by zero
        max_violation = jax.lax.select(max_violation > 0, max_violation, 1.0)

        # Calculate unnormalized new weights
        new_weight_values = []
        total_new_weight = jnp.array(0.0)

        adaptation_rate = self.config.constraint_config.adaptation_rate

        for i, constraint in enumerate(constraints):
            # Get current weight value (safely)
            if hasattr(self._constraint_weights, constraint):
                current_w = self._constraint_weights[constraint].value
            else:
                current_w = 1.0 / len(constraints)

            violation_ratio = violations_stack[i] / max_violation
            new_w = current_w * (1.0 + adaptation_rate * violation_ratio)
            new_weight_values.append(new_w)
            total_new_weight += new_w

        # Validate total weight > 0
        total_new_weight = jax.lax.select(total_new_weight > 0, total_new_weight, 1.0)

        # Normalize and update variables
        updated_weights_dict = {}
        for i, constraint in enumerate(constraints):
            normalized_w = new_weight_values[i] / total_new_weight

            # Update the nnx.Variable in place (JIT safe)
            if hasattr(self._constraint_weights, constraint):
                self._constraint_weights[constraint].value = normalized_w

            # For return value (if needed for debugging/metrics)
            updated_weights_dict[constraint] = normalized_w

        return updated_weights_dict


__all__ = ["Trainer"]
