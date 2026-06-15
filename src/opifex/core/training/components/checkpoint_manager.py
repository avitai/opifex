"""
CheckpointManager for Opifex Training Infrastructure

A focused component for managing model checkpoints during training.
Provides save, load, and management functionality for neural operator models.

Serialization is performed with safe mechanisms only: JAX/Flax NNX array
state is persisted with Orbax (:class:`orbax.checkpoint.StandardCheckpointer`)
and plain-Python metadata is persisted as JSON. No ``pickle`` is used, so
loading a checkpoint can never execute arbitrary code.
"""

import importlib
import json
import logging
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax import nnx


logger = logging.getLogger(__name__)

# Filenames used inside each checkpoint directory.
_STATE_DIRNAME = "state"
_METADATA_FILENAME = "metadata.json"

# Keys describing a single NNX variable leaf in the metadata schema. The
# schema lets ``load_checkpoint`` rebuild an abstract ``nnx.State`` target
# (a tree of ``jax.ShapeDtypeStruct``) without a live model instance, so the
# array data can be safely restored from the Orbax store.
_PATH_KEY = "path"
_SHAPE_KEY = "shape"
_DTYPE_KEY = "dtype"
_VARIABLE_MODULE_KEY = "variable_module"
_VARIABLE_QUALNAME_KEY = "variable_qualname"


def _state_to_schema(state: nnx.State) -> list[dict[str, Any]]:
    """Serialize an ``nnx.State`` structure into a JSON-friendly schema.

    The schema records, per array leaf, its key path, shape, dtype, and the
    fully-qualified Flax variable class (e.g. ``flax.nnx.Param``). This is the
    minimum needed to reconstruct an abstract restore target.
    """
    schema: list[dict[str, Any]] = []
    for path, variable in state.flat_state():
        value = variable.value
        variable_type = type(variable)
        schema.append(
            {
                _PATH_KEY: list(path),
                _SHAPE_KEY: list(value.shape),
                _DTYPE_KEY: jnp.dtype(value.dtype).name,
                _VARIABLE_MODULE_KEY: variable_type.__module__,
                _VARIABLE_QUALNAME_KEY: variable_type.__qualname__,
            }
        )
    return schema


def _resolve_variable_type(module_name: str, qualname: str) -> type[nnx.Variable]:
    """Resolve a Flax NNX variable class from its module and qualified name."""
    module = importlib.import_module(module_name)
    resolved: Any = module
    for attribute in qualname.split("."):
        resolved = getattr(resolved, attribute)
    if not (isinstance(resolved, type) and issubclass(resolved, nnx.Variable)):
        raise TypeError(f"{module_name}.{qualname} is not an nnx.Variable subclass")
    return resolved


def _schema_to_abstract_state(schema: list[dict[str, Any]]) -> nnx.State:
    """Rebuild an abstract ``nnx.State`` (``ShapeDtypeStruct`` leaves) from schema."""
    items: list[tuple[tuple[Any, ...], nnx.Variable]] = []
    for entry in schema:
        variable_type = _resolve_variable_type(
            entry[_VARIABLE_MODULE_KEY], entry[_VARIABLE_QUALNAME_KEY]
        )
        abstract_value = jax.ShapeDtypeStruct(
            tuple(entry[_SHAPE_KEY]), jnp.dtype(entry[_DTYPE_KEY])
        )
        items.append((tuple(entry[_PATH_KEY]), variable_type(abstract_value)))
    return nnx.State.from_flat_path(items)


class CheckpointManager:
    """
    Manages model checkpoints for training workflows.

    Provides functionality to save, load, list, and manage model checkpoints
    with metadata tracking and validation. Each checkpoint is a directory
    containing Orbax-serialized NNX state and a JSON metadata sidecar.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        auto_cleanup: bool = True,
    ) -> None:
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            auto_cleanup: Whether to automatically clean up old checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1")

    def save_checkpoint(
        self,
        model: nnx.Module,
        step: int,
        loss: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a model checkpoint.

        Array state is written with Orbax and metadata with JSON; no pickle
        is involved.

        Args:
            model: Model to save
            step: Training step number
            loss: Current loss value
            metadata: Additional metadata to save

        Returns:
            Path to the saved checkpoint
        """
        # Validate inputs
        if not isinstance(model, nnx.Module):
            raise TypeError("Model must be an nnx.Module")
        if not isinstance(step, int):
            raise TypeError("Step must be an integer")
        if step < 0:
            raise ValueError("Step cannot be negative")
        if not isinstance(loss, int | float):
            raise TypeError("Loss must be a number")

        # Create checkpoint directory with timezone-aware timestamp. The
        # ``.pkl`` suffix is retained only as a stable checkpoint-name marker;
        # the contents are Orbax + JSON, never a pickle stream.
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step}_{timestamp}.pkl"
        filepath = self.checkpoint_dir / filename
        filepath.mkdir(parents=True, exist_ok=True)

        model_state = nnx.state(model)

        # Persist array state with Orbax.
        state_dir = (filepath / _STATE_DIRNAME).resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(state_dir, model_state)

        # Persist plain-Python metadata (plus the structural schema needed to
        # rebuild the abstract restore target) as JSON.
        metadata_payload = {
            "step": step,
            "loss": float(loss),
            "timestamp": timestamp,
            "metadata": metadata or {},
            "structure": _state_to_schema(model_state),
        }
        with open(filepath / _METADATA_FILENAME, "w") as metadata_file:
            json.dump(metadata_payload, metadata_file, indent=2)

        # Auto cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_old_checkpoints()

        return str(filepath)

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary containing checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint file is corrupted or invalid
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            metadata_path = checkpoint_file / _METADATA_FILENAME
            with open(metadata_path) as metadata_file:
                metadata_payload = json.load(metadata_file)

            abstract_state = _schema_to_abstract_state(metadata_payload["structure"])
            state_dir = (checkpoint_file / _STATE_DIRNAME).resolve()
            with ocp.StandardCheckpointer() as checkpointer:
                model_state = checkpointer.restore(state_dir, target=abstract_state)

            return {
                "model_state": model_state,
                "step": metadata_payload["step"],
                "loss": metadata_payload["loss"],
                "timestamp": metadata_payload["timestamp"],
                "metadata": metadata_payload.get("metadata", {}),
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

    def restore_model(self, model: nnx.Module, checkpoint_path: str) -> nnx.Module:
        """
        Restore a model from a checkpoint.

        Args:
            model: Model instance to restore into
            checkpoint_path: Path to the checkpoint file

        Returns:
            Restored model
        """
        checkpoint_data = self.load_checkpoint(checkpoint_path)

        # Restore model state
        nnx.update(model, checkpoint_data["model_state"])

        return model

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        for filepath in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                checkpoint_data = self.load_checkpoint(str(filepath))
                checkpoint_info = {
                    "path": str(filepath),
                    "step": checkpoint_data["step"],
                    "loss": checkpoint_data["loss"],
                    "timestamp": checkpoint_data["timestamp"],
                    "metadata": checkpoint_data.get("metadata", {}),
                }
                checkpoints.append(checkpoint_info)
            except (OSError, ValueError, KeyError) as e:
                # Log corrupted checkpoints instead of silently skipping
                logger.warning("Skipping corrupted checkpoint %s: %s", filepath, e)
                continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"])

        return checkpoints

    def get_latest_checkpoint(self) -> str | None:
        """
        Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Return the checkpoint with the highest step number
        latest = max(checkpoints, key=lambda x: x["step"])
        return latest["path"]

    def get_best_checkpoint(self, metric: str = "loss", minimize: bool = True) -> str | None:
        """
        Get the path to the best checkpoint based on a metric.

        Args:
            metric: Metric to use for comparison ('loss' or metadata key)
            minimize: Whether to minimize the metric (True) or maximize (False)

        Returns:
            Path to the best checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        # Find best checkpoint based on metric
        def get_metric_value(checkpoint):
            if metric == "loss":
                return checkpoint["loss"]
            return checkpoint["metadata"].get(metric, float("inf") if minimize else float("-inf"))

        if minimize:
            best = min(checkpoints, key=get_metric_value)
        else:
            best = max(checkpoints, key=get_metric_value)

        return best["path"]

    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.is_dir():
                shutil.rmtree(checkpoint_file)
                return True
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                return True
            return False
        except OSError:
            return False

    def cleanup_all_checkpoints(self) -> int:
        """
        Delete all checkpoints in the directory.

        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0

        for filepath in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            if self.delete_checkpoint(str(filepath)):
                deleted_count += 1
            else:
                logger.warning("Failed to delete checkpoint %s", filepath)

        return deleted_count

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate that a checkpoint file is valid and loadable.

        Args:
            checkpoint_path: Path to the checkpoint to validate

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            checkpoint_data = self.load_checkpoint(checkpoint_path)

            # Check required fields
            required_fields = ["model_state", "step", "loss", "timestamp"]
            for field in required_fields:
                if field not in checkpoint_data:
                    return False

            # Check data types
            if not isinstance(checkpoint_data["step"], int):
                return False
            if not isinstance(checkpoint_data["loss"], int | float):
                return False

            # Return condition directly
            return isinstance(checkpoint_data["timestamp"], str)
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            # Sort by step number and keep only the latest max_checkpoints
            checkpoints.sort(key=lambda x: x["step"])
            checkpoints_to_delete = checkpoints[: -self.max_checkpoints]

            for checkpoint in checkpoints_to_delete:
                self.delete_checkpoint(checkpoint["path"])
