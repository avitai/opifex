"""UQ capability declarations for the L2O surfaces.

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.optimization.l2o.__init__``.

Phase 7 (Task 7.5) recorded :class:`BayesianSchedulerOptimizer` honestly
as :attr:`DefaultStrategy.UNSUPPORTED` because the original
``_bayesian_parameter_suggestion`` used a random exploration heuristic.
Phase 8 Task 8.3 replaced the heuristic with a real expected-improvement
acquisition via :func:`opifex.uncertainty.active.expected_improvement`
(``adaptive_schedulers.py``). Phase 8 Task 8.5 flips
:attr:`native_bayesian` and :attr:`default_strategy` to advertise the
real BO-style acquisition surface.

Plan reference: ``08-phase-pac-bayes-sbi-active-stochastic-fields.md``
lines 755-790.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_BAYESIAN_SCHEDULER_OPTIMIZER_CAPABILITY = UQCapability(
    native_bayesian=True,
    default_strategy=DefaultStrategy.BAYESIAN,
    native_nnx_module=True,
    source_package="opifex",
    notes=(
        "BayesianSchedulerOptimizer tunes scheduler hyperparameters via "
        "GP-style expected-improvement acquisition. Phase 8 Task 8.3 "
        "replaced the original random-exploration heuristic with a real "
        "EI acquisition: ``_score_candidate`` builds a "
        ":class:`PredictiveDistribution` from the candidate history and "
        "delegates to :func:`opifex.uncertainty.active.expected_improvement` "
        "(``adaptive_schedulers.py``). Phase 8 Task 8.5 flipped "
        "native_bayesian + default_strategy to BAYESIAN to advertise "
        "the real acquisition surface."
    ),
)


L2O_CAPABILITIES: dict[str, UQCapability] = {
    "l2o:BayesianSchedulerOptimizer": _BAYESIAN_SCHEDULER_OPTIMIZER_CAPABILITY,
}


__all__ = ["L2O_CAPABILITIES"]
