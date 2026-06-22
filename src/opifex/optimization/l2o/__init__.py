"""Learn-to-Optimize (L2O): meta-learned optimisers for scientific optimisation.

A learned optimiser is meta-trained to minimise a *distribution* of objectives, each carried by
a :class:`~opifex.optimization.l2o.core.Task`, and applied to held-out tasks where it can beat a
*tuned* classical baseline. The design follows Google's ``learned_optimization`` library and the
L2O literature (Andrychowicz et al. 2016, ``arXiv:1606.04474``; Metz et al. 2020,
``arXiv:2009.11243``; Vicol et al. 2021 PES, ``arXiv:2112.13835``); classical baselines use
``optimistix``. Every reported number is measured — there are no fabricated objectives or
speedups.

Public surface:

- :class:`~opifex.optimization.l2o.core.Task` / :class:`~opifex.optimization.l2o.core.TaskFamily`
  — objective-carrying abstractions (``init``/``loss``/``normalizer``) and a task distribution.
- :class:`~opifex.optimization.l2o.optimizers.Optimizer` ABC + ``OptaxOptimizer`` — the shared
  stateful optimiser interface, with an optax-wrapped hand-designed family.
- :class:`~opifex.optimization.l2o.learned.LearnedOptimizer` ABC + ``MLPLearnedOptimizer`` /
  ``LearnableSGD`` — coordinatewise meta-learned update rules.
- :func:`~opifex.optimization.l2o.meta_train.meta_train` — PES meta-training.
- baselines/benchmark helpers and the high-level :class:`L2OEngine` orchestrator.
"""

from opifex.optimization.l2o.baselines import (
    loss_curve,
    optimistix_minimise,
    tuned_optax_baseline,
)
from opifex.optimization.l2o.benchmark import (
    benchmark_on_held_out_tasks,
    speedup_at_target,
    steps_to_target,
)
from opifex.optimization.l2o.core import single_task_to_family, Task, TaskFamily
from opifex.optimization.l2o.engine import L2OEngine
from opifex.optimization.l2o.learned import (
    AdafacMLPLearnedOptimizer,
    LearnableSGD,
    LearnedOptimizer,
    MLPLearnedOptimizer,
)
from opifex.optimization.l2o.meta_learning import adapt, maml_meta_train, reptile_meta_train
from opifex.optimization.l2o.meta_train import init_pes_state, meta_train, pes_gradient_step
from opifex.optimization.l2o.optimizers import OptaxOptimizer, Optimizer
from opifex.optimization.l2o.tasks import (
    MLPTask,
    MLPTaskFamily,
    QuadraticTask,
    QuadraticTaskFamily,
)


__all__ = [
    "AdafacMLPLearnedOptimizer",
    "L2OEngine",
    "LearnableSGD",
    "LearnedOptimizer",
    "MLPLearnedOptimizer",
    "MLPTask",
    "MLPTaskFamily",
    "OptaxOptimizer",
    "Optimizer",
    "QuadraticTask",
    "QuadraticTaskFamily",
    "Task",
    "TaskFamily",
    "adapt",
    "benchmark_on_held_out_tasks",
    "init_pes_state",
    "loss_curve",
    "maml_meta_train",
    "meta_train",
    "optimistix_minimise",
    "pes_gradient_step",
    "reptile_meta_train",
    "single_task_to_family",
    "speedup_at_target",
    "steps_to_target",
    "tuned_optax_baseline",
]
