"""
CheckpointManager for Opifex Training Infrastructure

A focused component for managing model checkpoints during training.
Provides save, load, and management functionality for neural operator models.
"""

import logging
import pickle
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from flax import nnx


# Configure logging for security warnings
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints for training workflows.

    Provides functionality to save, load, list, and manage model checkpoints
    with metadata tracking and validation.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        auto_cleanup: bool = True,
    ):
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
        if not isinstance(loss, (int, float)):
            raise TypeError("Loss must be a number")

        # Create checkpoint filename with timezone-aware timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step}_{timestamp}.pkl"
        filepath = self.checkpoint_dir / filename

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state": nnx.state(model),
            "step": step,
            "loss": float(loss),
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        # Save checkpoint (security note: pickle should only be used with trusted data)
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

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

        # Security warning for pickle usage
        logger.warning(
            "Loading checkpoint with pickle - ensure file is from trusted source"
        )

        try:
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)  # noqa: S301  # nosec B301
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
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
            except Exception as e:
                # Log corrupted checkpoints instead of silently skipping
                logger.warning(f"Skipping corrupted checkpoint {filepath}: {e}")
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

    def get_best_checkpoint(
        self, metric: str = "loss", minimize: bool = True
    ) -> str | None:
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
            return checkpoint["metadata"].get(
                metric, float("inf") if minimize else float("-inf")
            )

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
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                return True
            return False
        except Exception:
            return False

    def cleanup_all_checkpoints(self) -> int:
        """
        Delete all checkpoints in the directory.

        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0

        for filepath in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                filepath.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {filepath}: {e}")
                continue

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
            if not isinstance(checkpoint_data["loss"], (int, float)):
                return False

            # Return condition directly
            return isinstance(checkpoint_data["timestamp"], str)
        except Exception:
            return False

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            # Sort by step number and keep only the latest max_checkpoints
            checkpoints.sort(key=lambda x: x["step"])
            checkpoints_to_delete = checkpoints[: -self.max_checkpoints]

            for checkpoint in checkpoints_to_delete:
                self.delete_checkpoint(checkpoint["path"])
