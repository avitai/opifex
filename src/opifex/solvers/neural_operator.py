"""Neural Operator Solver implementation.

This module provides the NeuralOperatorSolver, which solves PDE problems using
data-driven neural operators (e.g., FNO, DeepONet).
"""

import jax.numpy as jnp
from flax import nnx

from opifex.core.problems import Problem
from opifex.core.solver.interface import (
    Solution,
    SolverConfig,
    SolverState,
)
from opifex.core.solver.status import Status
from opifex.core.training.config import TrainingConfig
from opifex.core.training.trainer import Trainer


class NeuralOperatorSolver:
    """Solver for data-driven Neural Operators.

    Training is what this performs; prediction is not wired, so the returned solution
    carries metrics and an outcome but no fields. A problem carrying no data trains
    nothing and reports ``Status.DEFAULT`` rather than success, so an untrained run
    cannot be mistaken for a converged one.
    """

    def __init__(self, model: nnx.Module) -> None:
        self.model = model

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Execute the solver.

        Args:
             problem: The problem definition (provides geometry/physics OR data).
             initial_state: Initial solver state (step, params).
             config: Solver configuration.

        Returns:
            Solution object containing metrics and status.
        """
        cfg = config or SolverConfig()
        _state = initial_state or SolverState()

        training_config = TrainingConfig(
            num_epochs=cfg.max_iterations,
        )

        trainer = Trainer(
            model=self.model,
            config=training_config,
        )

        # Check if problem provides data
        from opifex.core.problems import DataDrivenProblem

        metrics = {}
        trained = isinstance(problem, DataDrivenProblem)
        if trained:
            # Execute real training loop with data
            _, metrics = trainer.fit(
                train_data=(problem.x_train, problem.y_train),
                val_data=problem.val_dataset,
            )

        return Solution(
            fields={},
            metrics=metrics,
            status=jnp.asarray(Status.SUCCESS if trained else Status.DEFAULT, dtype=jnp.int32),
        )
