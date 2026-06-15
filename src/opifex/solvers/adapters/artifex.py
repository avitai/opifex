"""Adapter for Artifex Generative Models to the :class:`SciMLSolver` interface.

Allows usage of Avitai Artifex Diffusion/Flow models within Opifex workflows.

.. warning::
    Experimental — planned for Version 5 (Foundation Models & Generative).
    The Artifex backend exposes no stable ``DDPMModel`` with a
    ``sample(rngs, condition, num_samples)`` interface: the only ``sample``
    methods live on modality wrappers with an incompatible
    ``sample(n_samples, **kwargs)`` signature, and
    ``artifex.generative_models.models`` does not currently import. Until a
    real delegation is wired, every entry point raises
    :class:`NotImplementedError` (following the "not wired" convention used in
    :mod:`opifex.platform.registry.core` and the inference-backend sampler
    hooks) rather than returning a placeholder solution.
"""

from typing import Any

from flax import nnx

from opifex.core.problems import Problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
    SolverConfig,
    SolverState,
)


_NOT_WIRED = "artifex solver adapter not implemented"


class ArtifexSolverAdapter(SciMLSolver):
    """Adapts an Artifex generative model to the :class:`SciMLSolver` protocol.

    This is intended to let Opifex use Artifex's generative models (Diffusion,
    Flows) to generate field solutions conditioned on a :class:`Problem`. The
    Artifex sampling API is not yet stable, so both entry points raise
    :class:`NotImplementedError` until the backend is wired.
    """

    def __init__(self, artifex_model: Any) -> None:
        """Initialize with an Artifex model instance.

        Args:
            artifex_model: An instantiated Artifex generative model (nnx.Module).
                           Must support a ``sample(rngs, condition, ...)`` or
                           similar API once the backend is wired.
        """
        self.model = artifex_model

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Generate a solution using the Artifex model.

        Raises:
            NotImplementedError: The Artifex sampling backend is not yet wired
                into this adapter; see the module docstring. Returning a dummy
                converged :class:`Solution` would be a silent-success trap, so
                the adapter fails fast instead.
        """
        del problem, initial_state, config
        raise NotImplementedError(_NOT_WIRED)

    def sample_batch(self, problem: Problem, num_samples: int, rngs: nnx.Rngs) -> Any:
        """Sample a batch of solutions from the underlying generative model.

        Intended for use upstream of
        :func:`opifex.uncertainty.scientific.summarize_stacked_sample_solution`,
        which then summarises the stacked sample axis.

        Raises:
            NotImplementedError: The Artifex sampling backend is not yet wired
                into this adapter; see the module docstring. Returning ``None``
                would be an implicit-success trap, so the adapter fails fast.
        """
        del problem, num_samples, rngs
        raise NotImplementedError(_NOT_WIRED)
