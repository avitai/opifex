"""Unified Solver Interface."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from opifex.core.problems import Problem


@dataclass(frozen=True)
class Solution:
    """Standardized solution object for all SciML solvers.

    Attributes:
        fields: Dictionary of solution fields (e.g., {"u": array, "p": array})
        metrics: Dictionary of final metrics (e.g., {"loss": 0.01})
        execution_time: Total execution time in seconds
        auxiliary_data: Any additional metadata or artifacts
    """

    fields: dict[str, Any]
    metrics: dict[str, Any]
    execution_time: float
    auxiliary_data: dict[str, Any] = field(default_factory=dict)
    converged: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverConfig:
    """Configuration for the solver."""

    max_iterations: int = 1000
    tolerance: float = 1e-4
    verbose: bool = True
    # Can hold a TrainingConfig if using the Trainer
    training_config: Any | None = None


@dataclass
class SolverState:
    """State of the solver (parameters, optimizer state, etc.)."""

    params: Any | None = None
    optim_state: Any | None = None
    step: int = 0
    rng_key: Any | None = None


@runtime_checkable
class SciMLSolver(Protocol):
    """Protocol defining the interface for all Scientific Machine Learning solvers.

    A solver encapsulates the algorithm to solve a specific Problem using
    a specific computational Strategy.
    """

    @abstractmethod
    @abstractmethod
    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Solve the given problem and return the solution.

        Args:
            problem: The problem instance (PDE, ODE, Optimization, etc.)
            initial_state: Optional initial state (params, etc.)
            config: Optional configuration override

        Returns:
            Solution object containing fields and metrics
        """
        ...
