"""Experiment creation over registered backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.mlops.backends.mlflow_backend import MLflowBackend


if TYPE_CHECKING:
    from opifex.mlops.experiment import Experiment, ExperimentConfig


class ExperimentTracker:
    """Creates experiments on a named backend; MLflow is registered from the start.

    Args:
        default_backend: The backend used when a configuration's ``backend`` is
            ``"auto"``.
    """

    def __init__(self, default_backend: str = "mlflow") -> None:
        self.default_backend = default_backend
        self._backends: dict[str, type[Experiment]] = {"mlflow": MLflowBackend}

    @property
    def backends(self) -> tuple[str, ...]:
        """The registered backend names, in registration order."""
        return tuple(self._backends)

    def register_backend(self, name: str, backend_class: type[Experiment]) -> None:
        """Register ``backend_class`` under ``name``, replacing any previous registration."""
        self._backends[name] = backend_class

    async def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create the experiment ``config`` describes on its backend, or the default one.

        Args:
            config: The experiment configuration; ``backend == "auto"`` selects
                ``default_backend``.

        Returns:
            The experiment, not yet started.

        Raises:
            ValueError: If the backend is not registered.
        """
        name = self.default_backend if config.backend == "auto" else config.backend
        if name not in self._backends:
            registered = ", ".join(self._backends)
            raise ValueError(f"Backend {name!r} is not registered; registered: {registered}")
        return self._backends[name](config)
