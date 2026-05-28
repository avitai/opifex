"""UQ capability declarations for the L2O surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.optimization.l2o.__init__``.

Phase 7 records :class:`BayesianSchedulerOptimizer` honestly:

* ``native_bayesian=False`` and
  ``default_strategy=DefaultStrategy.UNSUPPORTED`` because the current
  ``_bayesian_parameter_suggestion`` is a random heuristic (see
  ``adaptive_schedulers.py``).
* :attr:`notes` points at Phase 8 Task 8.3, which replaces the heuristic
  with a real GP-backed acquisition function using
  ``experimental_design.py``. The Bayesian flag flips to ``True`` only
  after Phase 8 Task 8.5 audits the upgraded implementation.

Plan reference: ``07-phase-registry-docs-examples.md`` lines 632-642 (Task
6.3 expansion note dated 2026-05-20).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_BAYESIAN_SCHEDULER_OPTIMIZER_CAPABILITY = UQCapability(
    native_bayesian=False,
    default_strategy=DefaultStrategy.UNSUPPORTED,
    native_nnx_module=True,
    source_package="opifex",
    notes=(
        "BayesianSchedulerOptimizer is named for an upcoming "
        "GP-acquisition-backed scheduler tuner. The current "
        "_bayesian_parameter_suggestion implementation is a random "
        "exploration heuristic (adaptive_schedulers.py); Phase 8 "
        "Task 8.3 replaces it with a real GP using experimental_design.py "
        "and Phase 8 Task 8.5 flips native_bayesian + default_strategy "
        "after the upgrade lands. Declared UNSUPPORTED in Phase 7."
    ),
)


L2O_CAPABILITIES: dict[str, UQCapability] = {
    "l2o:BayesianSchedulerOptimizer": _BAYESIAN_SCHEDULER_OPTIMIZER_CAPABILITY,
}


__all__ = ["L2O_CAPABILITIES"]
