"""Unified, orbax-backed checkpoint store for the Opifex training stack.

This module is the single source of truth for persisting and restoring
JAX/Flax (NNX) model state during training. It replaces the two formerly
duplicated managers (a step-int ``OrbaxCheckpointManager`` and a
path-string ``CheckpointManager``) with one abstraction:

* :class:`CheckpointStore` — the abstraction-layer ``Protocol`` describing
  the step-int checkpoint contract. Domain/service code depends on this
  type, never on the concrete implementation (dependency inversion).
* :class:`OrbaxCheckpointStore` — the infrastructure-layer implementation,
  backed by `Orbax <https://orbax.readthedocs.io>`_
  (``orbax.checkpoint``). Orbax-native step-int addressing is the canonical
  on-disk contract; retention is delegated to Orbax's ``max_to_keep``.

Serialization is pickle-free: array state is written with Orbax's
``StandardSave``/``StandardRestore`` and plain-Python metadata with
``JsonSave``/``JsonRestore``, so restoring a checkpoint can never execute
arbitrary code.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Self

import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax import nnx
from flax.training import train_state


logger = logging.getLogger(__name__)

# Tag stamped onto every checkpoint's metadata so future readers can detect
# the on-disk schema version.
_CHECKPOINT_VERSION = "2.0"

# Restorable model payloads. NNX modules, Flax ``TrainState`` objects, and
# plain state dictionaries are all supported.
ModelLike = nnx.Module | train_state.TrainState | dict[str, Any]


@runtime_checkable
class CheckpointStore(Protocol):
    """Step-int addressed checkpoint store contract.

    Implementations persist a model payload plus JSON metadata under an
    integer ``step`` and restore it back. This is the single abstraction the
    training layer depends on; concrete backends live in the infrastructure
    layer.
    """

    def save(
        self,
        model: ModelLike,
        step: int,
        loss: float,
        *,
        physics_metadata: dict[str, Any] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist ``model`` at ``step`` and return the checkpoint path."""
        ...

    def restore(
        self,
        target_model: ModelLike | None = None,
        step: int | None = None,
        *,
        return_original_on_missing: bool = True,
        restrict_to_nnx_module: bool = False,
    ) -> tuple[ModelLike | None, dict[str, Any]]:
        """Restore the payload + metadata saved at ``step``."""
        ...

    def list_steps(self) -> list[int]:
        """Return all available checkpoint steps."""
        ...

    def latest_step(self) -> int | None:
        """Return the most recent checkpoint step, or ``None``."""
        ...

    def best_step(self, metric: str = "loss", *, minimize: bool = True) -> int | None:
        """Return the step optimizing ``metric``, or ``None``."""
        ...

    def delete(self, step: int) -> bool:
        """Delete the checkpoint at ``step``; return success."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


class OrbaxCheckpointStore:
    """Orbax-backed implementation of :class:`CheckpointStore`.

    Wraps :class:`orbax.checkpoint.CheckpointManager` with step-int
    addressing. A single checkpoint bundles the model array state
    (``StandardSave``) and a JSON metadata sidecar (``JsonSave``) carrying
    the loss, timestamp, model type, physics metadata, and any caller
    extras. Old checkpoints are pruned by Orbax according to ``max_to_keep``.

    The store doubles as a context manager so backend resources are released
    deterministically::

        with OrbaxCheckpointStore(path) as store:
            store.save(model, step=0, loss=loss)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
        create: bool = True,
    ) -> None:
        """Initialize the store.

        Args:
            checkpoint_dir: Directory in which checkpoints are stored.
            max_to_keep: Maximum number of checkpoints Orbax retains.
            create: Whether to create ``checkpoint_dir`` if it is missing.

        Raises:
            ValueError: If ``checkpoint_dir`` is empty or whitespace-only.
        """
        if not checkpoint_dir or str(checkpoint_dir).strip() == "":
            raise ValueError("Checkpoint directory cannot be empty")

        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.max_to_keep = max_to_keep

        if create:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=create,
        )
        self._manager = ocp.CheckpointManager(self.checkpoint_dir, options=self._options)
        logger.info("Initialized OrbaxCheckpointStore at %s", self.checkpoint_dir)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the backend on context exit."""
        self.close()

    @staticmethod
    def _build_save_payload(
        model: ModelLike,
        step: int,
        loss: float,
        physics_metadata: dict[str, Any] | None,
        additional_metadata: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Build the (save_target, metadata) pair for a checkpoint."""
        if isinstance(model, nnx.Module):
            save_target: Any = nnx.state(model)
            model_type = "nnx_module"
        elif isinstance(model, train_state.TrainState):
            save_target = model.replace(step=step)
            model_type = "train_state"
        else:
            save_target = model
            model_type = "dict"

        metadata: dict[str, Any] = {
            "step": step,
            "loss": float(loss),
            "timestamp": time.time(),
            "model_type": model_type,
            "model_class": type(model).__name__,
            "checkpoint_version": _CHECKPOINT_VERSION,
        }
        if physics_metadata:
            metadata["physics_metadata"] = physics_metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        return save_target, metadata

    def save(
        self,
        model: ModelLike,
        step: int,
        loss: float,
        *,
        physics_metadata: dict[str, Any] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a checkpoint with complete model state and metadata.

        Args:
            model: Model payload — an ``nnx.Module``, a Flax ``TrainState``,
                or a plain state ``dict``.
            step: Non-negative training step number (the checkpoint key).
            loss: Current loss value, recorded in metadata.
            physics_metadata: Optional physics-specific metadata.
            additional_metadata: Optional extra metadata merged into the
                top-level metadata record.

        Returns:
            Filesystem path of the saved checkpoint.

        Raises:
            TypeError: If ``model`` is not a supported type.
            ValueError: If ``step`` is negative.
        """
        if not isinstance(model, nnx.Module | train_state.TrainState | dict):
            raise TypeError("Expected model to be nnx.Module, TrainState, or dict")
        if step < 0:
            raise ValueError("Step must be a non-negative integer")

        save_target, metadata = self._build_save_payload(
            model, step, loss, physics_metadata, additional_metadata
        )
        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(save_target),  # type: ignore[reportCallIssue]
            metadata=ocp.args.JsonSave(metadata),  # type: ignore[reportCallIssue]
        )
        self._manager.save(step, args=save_args)  # type: ignore[reportCallIssue]
        self._manager.wait_until_finished()
        logger.info("Saved checkpoint for step %s", step)
        return str(self.checkpoint_dir / str(step))

    def _restore_args(self, target_model: ModelLike | None) -> Any:
        """Build Orbax restore args for ``target_model``."""
        if target_model is None:
            return ocp.args.Composite(metadata=ocp.args.JsonRestore)  # type: ignore[call-arg, reportCallIssue]
        abstract = nnx.state(target_model) if isinstance(target_model, nnx.Module) else target_model
        return ocp.args.Composite(
            model=ocp.args.StandardRestore(abstract),  # type: ignore[arg-type, reportCallIssue]
            metadata=ocp.args.JsonRestore,  # type: ignore[call-arg, reportCallIssue]
        )

    @staticmethod
    def _merge_restored(
        target_model: ModelLike | None,
        model_restored: Any,
    ) -> ModelLike | None:
        """Merge restored arrays into ``target_model`` by payload kind."""
        if model_restored is None:
            return target_model
        if target_model is None:
            return model_restored
        if isinstance(target_model, nnx.Module):
            nnx.update(target_model, model_restored)
            return target_model
        if isinstance(target_model, train_state.TrainState):
            return model_restored
        target_model.update(model_restored)
        return target_model

    def restore(
        self,
        target_model: ModelLike | None = None,
        step: int | None = None,
        *,
        return_original_on_missing: bool = True,
        restrict_to_nnx_module: bool = False,
    ) -> tuple[ModelLike | None, dict[str, Any]]:
        """Restore model state and metadata for ``step``.

        Args:
            target_model: Model to restore arrays into. ``None`` requests a
                metadata-only restore (and returns ``None`` for the model).
            step: Step to restore.
            return_original_on_missing: If ``True``, return ``target_model``
                unchanged when the checkpoint is missing; otherwise ``None``.
            restrict_to_nnx_module: If ``True``, coerce a non-``nnx.Module``
                result to ``None``.

        Returns:
            Tuple of ``(model_or_none, metadata)``. ``metadata`` is empty
            when the checkpoint is missing or could not be read.
        """
        is_missing = (
            step is None
            or step not in self._manager.all_steps()
            or (
                target_model is not None
                and not isinstance(target_model, nnx.Module | train_state.TrainState | dict)
            )
        )
        result_model: ModelLike | None = target_model
        metadata: dict[str, Any] = {}
        if not is_missing:
            try:
                restored = self._manager.restore(step, args=self._restore_args(target_model))  # type: ignore[reportCallIssue]
                metadata = restored.get("metadata", {})
                result_model = self._merge_restored(target_model, restored.get("model"))
            except (KeyError, ValueError, TypeError, OSError):
                logger.exception("Error restoring checkpoint at step %s", step)
                is_missing = True

        if is_missing:
            result_model = target_model if return_original_on_missing else None
            metadata = {}
        if restrict_to_nnx_module and not (
            result_model is None or isinstance(result_model, nnx.Module)
        ):
            result_model = None
        return result_model, metadata

    def create_train_state(
        self,
        model: nnx.Module,
        optimizer: Any,
        step: int = 0,
        **kwargs: Any,
    ) -> train_state.TrainState:
        """Create a Flax ``TrainState`` for complete-state checkpointing.

        Args:
            model: NNX module to wrap.
            optimizer: An optax optimizer used as the ``TrainState`` ``tx``.
            step: Initial step number.
            **kwargs: Extra arguments forwarded to ``TrainState.create``.

        Returns:
            A ``TrainState`` carrying the model parameters and optimizer.

        Raises:
            TypeError: If ``model`` is not an ``nnx.Module``.
        """
        try:
            model_state = nnx.state(model)
        except (RuntimeError, TypeError, ValueError, IndexError, AttributeError) as e:
            raise TypeError(f"Expected model to be nnx.Module, got {type(model)}") from e
        params = model_state.get("params", model_state)
        step_val = kwargs.pop("step", step)
        created = train_state.TrainState.create(
            apply_fn=model, params=params, tx=optimizer, **kwargs
        )
        return created.replace(step=step_val)

    def save_train_state(
        self,
        state: train_state.TrainState,
        step: int,
        loss: float,
        *,
        physics_metadata: dict[str, Any] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a ``TrainState`` checkpoint (see :meth:`save`)."""
        return self.save(
            state,
            step,
            loss,
            physics_metadata=physics_metadata,
            additional_metadata=additional_metadata,
        )

    def restore_train_state(
        self,
        target_state: train_state.TrainState,
        step: int | None = None,
    ) -> tuple[train_state.TrainState, dict[str, Any]]:
        """Restore a ``TrainState`` checkpoint, falling back to the target.

        Args:
            target_state: ``TrainState`` whose structure guides restoration.
            step: Step to restore.

        Returns:
            Tuple of ``(train_state, metadata)``.
        """
        restored, metadata = self.restore(target_state, step)
        if isinstance(restored, train_state.TrainState):
            return restored, metadata
        return target_state, metadata

    def list_steps(self) -> list[int]:
        """Return all available checkpoint steps."""
        return list(self._manager.all_steps())

    def latest_step(self) -> int | None:
        """Return the most recent checkpoint step, or ``None``."""
        return self._manager.latest_step()

    def best_step(self, metric: str = "loss", *, minimize: bool = True) -> int | None:
        """Return the checkpoint step optimizing ``metric``.

        The metric is read from each checkpoint's JSON metadata. ``"loss"``
        reads the top-level loss; any other name is looked up in metadata.

        Args:
            metric: Metadata key to optimize (``"loss"`` by default).
            minimize: Whether the best value is the minimum (``True``) or
                maximum (``False``).

        Returns:
            The optimizing step, or ``None`` if no checkpoints exist.
        """
        steps = self.list_steps()
        if not steps:
            return None
        default = float("inf") if minimize else float("-inf")

        def metric_value(step: int) -> float:
            _, metadata = self.restore(target_model=None, step=step)
            if metric == "loss":
                return float(metadata.get("loss", default))
            return float(metadata.get(metric, default))

        chooser = min if minimize else max
        return chooser(steps, key=metric_value)

    def delete(self, step: int) -> bool:
        """Delete the checkpoint at ``step``.

        Args:
            step: Step number to delete.

        Returns:
            ``True`` on success, ``False`` if deletion failed.
        """
        try:
            self._manager.delete(step)
            logger.info("Deleted checkpoint for step %s", step)
            return True
        except (ValueError, KeyError, FileNotFoundError, OSError):
            logger.exception("Error deleting checkpoint for step %s", step)
            return False

    def close(self) -> None:
        """Close the underlying Orbax manager and release resources."""
        try:
            self._manager.close()
            logger.info("OrbaxCheckpointStore closed")
        except (ValueError, OSError, RuntimeError):
            logger.exception("Error closing checkpoint store")
