"""The run logger an experiment backend records through."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class RunLogger(Protocol):
    """What a backend needs from a tracking run; substrax's ``MLFlowLogger`` provides it.

    A run has an id, takes scalar metrics with an optional step, hyperparameters,
    single files and whole directories as artifacts, and is closed once.
    """

    run_id: str

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        """Log several metrics at ``step``."""
        ...

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log run parameters."""
        ...

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log one file under ``artifact_path`` in the run's artifacts."""
        ...

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log every file of a directory under ``artifact_path`` in the run's artifacts."""
        ...

    def close(self) -> None:
        """End the run and release the logger's resources."""
        ...
