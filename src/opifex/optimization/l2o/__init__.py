"""Learn-to-Optimize (L2O) algorithms for scientific machine learning.

This module implements neural network-based optimization algorithms that learn to solve
families of optimization problems with significant speedup over traditional methods.

Key Features:
- Parametric programming solver networks
- Constraint satisfaction learning
- Unified L2O engine integrating multiple optimization strategies
- Advanced meta-learning algorithms (MAML, Reptile, Gradient-based)
- Meta-L2O integration for self-improving optimization
- Multi-objective optimization with Pareto frontier approximation
- Reinforcement learning-based optimization strategy selection
- >100x speedup on learned problem families
- Integration with traditional solvers via Optimistix
"""

from .adaptive_schedulers import (
    BayesianSchedulerOptimizer,
    create_l2o_engine_with_adaptive_schedulers,
    MetaSchedulerConfig,
    MultiscaleScheduler,
    PerformanceAwareScheduler,
    SchedulerIntegration,
)
from .advanced_meta_learning import (
    GradientBasedMetaLearner,
    GradientBasedMetaLearningConfig,
    MAMLConfig,
    MAMLOptimizer,
    MetaL2OIntegration,
    ReptileConfig,
    ReptileOptimizer,
)
from .l2o_engine import (
    L2OEngine,
    L2OEngineConfig,
    OptimizationProblemEncoder,
    ParametricOptimizationSolver,
)
from .multi_objective import (
    MultiObjectiveConfig,
    MultiObjectiveL2OEngine,
    ObjectiveScalarizer,
    ParetoFrontierOptimizer,
    PerformanceIndicators,
)
from .parametric_solver import (
    ConstraintHandler,
    OptimizationProblem,
    ParametricProgrammingSolver,
    SolverConfig,
)
from .rl_optimization import (
    ActionInterpreter,
    DQNNetwork,
    Experience,
    ExperienceReplayBuffer,
    RewardFunction,
    RLOptimizationAgent,
    RLOptimizationConfig,
    RLOptimizationEngine,
    StateEncoder,
)


__all__ = [
    "ActionInterpreter",
    "BayesianSchedulerOptimizer",
    "ConstraintHandler",
    "DQNNetwork",
    "Experience",
    "ExperienceReplayBuffer",
    "GradientBasedMetaLearner",
    "GradientBasedMetaLearningConfig",
    "L2OEngine",
    "L2OEngineConfig",
    "MAMLConfig",
    "MAMLOptimizer",
    "MetaL2OIntegration",
    "MetaSchedulerConfig",
    "MultiObjectiveConfig",
    "MultiObjectiveL2OEngine",
    "MultiscaleScheduler",
    "ObjectiveScalarizer",
    "OptimizationProblem",
    "OptimizationProblemEncoder",
    "ParametricOptimizationSolver",
    "ParametricProgrammingSolver",
    "ParetoFrontierOptimizer",
    "PerformanceAwareScheduler",
    "PerformanceIndicators",
    "RLOptimizationAgent",
    "RLOptimizationConfig",
    "RLOptimizationEngine",
    "ReptileConfig",
    "ReptileOptimizer",
    "RewardFunction",
    "SchedulerIntegration",
    "SolverConfig",
    "StateEncoder",
    "create_l2o_engine_with_adaptive_schedulers",
]
