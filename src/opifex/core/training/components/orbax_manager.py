"""
Orbax-based checkpoint manager for Opifex framework.

This module provides a high-level interface to Orbax checkpointing functionality,
simplifying the save/load operations for neural network models and associated
metadata. Supports complete model state checkpointing including weights,
optimizer state, and training metadata.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

# type: ignore[import-untyped]
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax import nnx
from flax.training import train_state


logger = logging.getLogger(__name__)


class OrbaxCheckpointManager:
    """Orbax-based checkpoint manager following workshop patterns.

    Provides a simplified interface for saving and loading JAX/Flax models
    using Orbax checkpointing infrastructure with proper error handling
    and metadata management. Supports complete model state checkpointing
    including weights, optimizer state, and training metadata.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
        create: bool = True,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            create: Whether to create the directory if it doesn't exist

        Raises:
            ValueError: If checkpoint_dir is empty or invalid
        """
        # Validate checkpoint directory
        if not checkpoint_dir or str(checkpoint_dir).strip() == "":
            raise ValueError("Checkpoint directory cannot be empty")

        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.max_to_keep = max_to_keep

        if create:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create Orbax CheckpointManager with proper options
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=create,
        )

        # Use StandardCheckpointer for PyTree checkpointing
        self.checkpointer = ocp.StandardCheckpointer()
        self.checkpoint_manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=self.options,
        )

        logger.info(f"Initialized OrbaxCheckpointManager at {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        model: nnx.Module | train_state.TrainState | dict[str, Any],
        step: int,
        loss: float,
        physics_metadata: dict[str, Any] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a checkpoint with complete model state and metadata.

        Args:
            model: The model to save. Can be:
                - nnx.Module: Will save model state
                - train_state.TrainState: Will save complete training state
                - dict: Will save the complete state dictionary
            step: Training step number
            loss: Current loss value
            physics_metadata: Physics-specific metadata
            additional_metadata: Additional metadata to save

        Returns:
            Path to the saved checkpoint

        Raises:
            TypeError: If model is not a supported type
            ValueError: If step is negative
        """
        # Validate inputs
        if not isinstance(model, (nnx.Module, train_state.TrainState, dict)):
            raise TypeError("Expected model to be nnx.Module, TrainState, or dict")

        if step < 0:
            raise ValueError("Step must be a non-negative integer")

        try:
            # Prepare the complete checkpoint state
            save_target: Any
            if isinstance(model, nnx.Module):
                save_target = nnx.state(model)
                model_type = "nnx_module"
                model_class = model.__class__.__name__
            elif isinstance(model, train_state.TrainState):
                save_target = model.replace(step=step)
                model_type = "train_state"
                model_class = type(model).__name__
            else:
                save_target = model
                model_type = "dict"
                model_class = type(model).__name__

            metadata = {
                "step": step,
                "loss": float(loss),
                "timestamp": time.time(),
                "model_type": model_type,
                "model_class": model_class,
            }
            if physics_metadata:
                metadata["physics_metadata"] = physics_metadata
            if additional_metadata:
                metadata.update(additional_metadata)

            # Save checkpoint using composite save args
            # Always add checkpoint_version for compatibility
            metadata["checkpoint_version"] = "2.0"
            # Save model state and metadata together
            save_args = ocp.args.Composite(
                model=ocp.args.StandardSave(save_target),  # type: ignore[reportCallIssue]
                metadata=ocp.args.JsonSave(metadata),  # type: ignore[reportCallIssue]
            )
            self.checkpoint_manager.save(step, args=save_args)  # type: ignore[reportCallIssue]
            self.checkpoint_manager.wait_until_finished()
            logger.info(f"Successfully saved complete checkpoint for step {step}")
            return str(self.checkpoint_dir / str(step))
        except Exception:
            logger.exception("Error saving checkpoint")
            raise

    def _restore_nnx_module(self, target_model, model_restored, step):
        if model_restored is not None:
            nnx.update(target_model, model_restored)
            logger.info(f"Successfully restored nnx.Module state from step {step}")
        return target_model

    def _restore_train_state(self, target_model, model_restored, step):
        if model_restored is not None:
            logger.info(f"Successfully restored TrainState from step {step}")
            return model_restored
        return target_model

    def _restore_dict(self, target_model, model_restored, step):
        if model_restored is not None:
            target_model.update(model_restored)
            logger.info(f"Successfully restored dict state from step {step}")
        return target_model

    def _get_restore_args(self, target_model):
        if target_model is None:
            return ocp.args.Composite(metadata=ocp.args.JsonRestore)  # type: ignore[call-arg, reportCallIssue]
        if isinstance(target_model, nnx.Module):
            return ocp.args.Composite(
                model=ocp.args.StandardRestore(nnx.state(target_model)),  # type: ignore[arg-type, reportCallIssue]
                metadata=ocp.args.JsonRestore,  # type: ignore[call-arg, reportCallIssue]
            )
        # train_state or dict
        return ocp.args.Composite(
            model=ocp.args.StandardRestore(target_model),  # type: ignore[arg-type, reportCallIssue]
            metadata=ocp.args.JsonRestore,  # type: ignore[call-arg, reportCallIssue]
        )

    def load_checkpoint(
        self,
        target_model: nnx.Module
        | train_state.TrainState
        | dict[str, Any]
        | None = None,
        step: int | None = None,
        return_original_on_missing: bool = True,
        restrict_to_nnx_module: bool = False,
    ) -> tuple[
        nnx.Module | train_state.TrainState | dict[str, Any] | None, dict[str, Any]
    ]:
        """Load a checkpoint and restore complete model state and metadata.

        Args:
            target_model: Model to restore into
            step: Step to restore
            return_original_on_missing: If True, return original model on
                missing checkpoint
            restrict_to_nnx_module: If True, only allow nnx.Module or None as
                return type
        """
        model_restored = None
        metadata = {}
        result_model = target_model
        error = (
            step is None
            or step not in self.checkpoint_manager.all_steps()
            or (
                target_model is not None
                and not isinstance(
                    target_model, (nnx.Module, train_state.TrainState, dict)
                )
            )
        )
        try:
            if not error:
                restore_args = self._get_restore_args(target_model)
                # Restore checkpoint
                restored_data = self.checkpoint_manager.restore(step, args=restore_args)  # type: ignore[reportCallIssue]
                model_restored = restored_data.get("model", None)
                metadata = restored_data.get("metadata", {})
                if target_model is None:
                    result_model = model_restored
                elif isinstance(target_model, nnx.Module):
                    result_model = self._restore_nnx_module(
                        target_model, model_restored, step
                    )
                elif isinstance(target_model, train_state.TrainState):
                    result_model = self._restore_train_state(
                        target_model, model_restored, step
                    )
                elif isinstance(target_model, dict):
                    result_model = self._restore_dict(
                        target_model, model_restored, step
                    )
        except Exception:
            logger.exception("Error loading checkpoint")
            error = True
        if error:
            result_model = target_model if return_original_on_missing else None
            metadata = {}
        if restrict_to_nnx_module and not (
            result_model is None or isinstance(result_model, nnx.Module)
        ):
            result_model = None
        return result_model, metadata

    def create_train_state(
        self, model: nnx.Module, optimizer, step: int = 0, **kwargs
    ) -> train_state.TrainState:
        """Create a TrainState for complete checkpointing.

        Args:
            model: The nnx.Module model
            optimizer: The optimizer (e.g., optax optimizer)
            step: Initial step number
            **kwargs: Additional arguments for TrainState.create

        Returns:
            TrainState with complete training state
        """
        # Get model parameters
        try:
            model_state = nnx.state(model)
        except (RuntimeError, TypeError, ValueError, IndexError, AttributeError) as e:
            raise TypeError(
                f"Expected model to be nnx.Module, got {type(model)}"
            ) from e
        params = model_state.get("params", model_state)
        # Always pass 'step' as a kwarg to TrainState.create, never as a named argument
        # Remove 'step' from kwargs if present, and set it after creation
        step_val = kwargs.pop("step", step)
        ts = train_state.TrainState.create(
            apply_fn=model, params=params, tx=optimizer, **kwargs
        )
        return ts.replace(step=step_val)

    def save_train_state_checkpoint(
        self,
        train_state_obj: train_state.TrainState,
        step: int,
        loss: float,
        physics_metadata: dict[str, Any] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a checkpoint with complete TrainState.

        Args:
            train_state_obj: The TrainState to save
            step: Training step number
            loss: Current loss value
            physics_metadata: Physics-specific metadata
            additional_metadata: Additional metadata to save

        Returns:
            Path to the saved checkpoint
        """
        return self.save_checkpoint(
            train_state_obj, step, loss, physics_metadata, additional_metadata
        )

    def load_train_state_checkpoint(
        self,
        target_train_state: train_state.TrainState,
        step: int | None = None,
    ) -> tuple[train_state.TrainState, dict[str, Any]]:
        """Load a checkpoint and restore TrainState.

        Args:
            target_train_state: Target TrainState to restore into
            step: Specific step to load (if None, loads latest)

        Returns:
            Tuple of (restored_train_state, metadata)
        """
        restored_state, metadata = self.load_checkpoint(target_train_state, step)
        if isinstance(restored_state, train_state.TrainState):
            return restored_state, metadata
        # If restoration failed, return original
        return target_train_state, metadata

    def list_checkpoints(self) -> list[int]:
        """List all available checkpoint steps."""
        return list(self.checkpoint_manager.all_steps())

    def get_latest_step(self) -> int | None:
        """Get the latest checkpoint step."""
        return self.checkpoint_manager.latest_step()

    def delete_checkpoint(self, step: int) -> bool:
        """Delete a specific checkpoint.

        Args:
            step: Step number to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.checkpoint_manager.delete(step)
            logger.info(f"Successfully deleted checkpoint for step {step}")
            return True
        except Exception:
            logger.exception(f"Error deleting checkpoint for step {step}")
            return False

    def close(self) -> None:
        """Close the checkpoint manager and clean up resources."""
        try:
            self.checkpoint_manager.close()
            logger.info("OrbaxCheckpointManager closed successfully")
        except Exception:
            logger.exception("Error closing checkpoint manager")
