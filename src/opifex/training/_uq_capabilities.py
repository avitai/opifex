"""UQ capability declarations for the training surfaces.

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.training.__init__``.

Phase 7 (Task 7.5) recorded both ``UncertaintyGuidedTrainer`` and
``MultiFidelityUncertaintyTrainer`` honestly as
:attr:`DefaultStrategy.UNSUPPORTED` because they used a hardcoded
``jax.random.PRNGKey(42)`` mock prediction and never called the wrapped
model. Phase 8 Task 8.3 rewrote both trainers (and added the
``ActiveUncertaintyLearner``) to invoke the wrapped model + uncertainty
quantifier for real and to delegate acquisition formulas to
:mod:`opifex.uncertainty.active.acquisition`. Phase 8 Task 8.5 flips the
capability flags here accordingly.

Plan reference: ``08-phase-pac-bayes-sbi-active-stochastic-fields.md``
lines 755-790 (capability-flag flip after the Task 8.3 TDD steps pass).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_UNCERTAINTY_GUIDED_TRAINER_CAPABILITY = UQCapability(
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes=(
        "UncertaintyGuidedTrainer drives active-learning sample "
        "selection. Phase 8 Task 8.3 rewrote it to invoke the wrapped "
        "model via _stochastic_ensemble_from_model and route the "
        "predictions through self.uncertainty_quantifier; the formula "
        "evaluation is delegated to "
        "opifex.uncertainty.active.acquire so no acquisition body lives "
        "inside the trainer. Phase 8 Task 8.5 flipped "
        "supports_active_learning + default_strategy to match the new "
        "implementation."
    ),
)


_MULTI_FIDELITY_UNCERTAINTY_TRAINER_CAPABILITY = UQCapability(
    supports_ensemble=True,
    default_strategy=DefaultStrategy.ENSEMBLE,
    source_package="opifex",
    notes=(
        "MultiFidelityUncertaintyTrainer propagates uncertainty through "
        "high/low fidelity model ensembles with Kennedy-O'Hagan "
        "additive linear weighting (fidelity_ratio). Phase 8 Task 8.3 "
        "rewrote it to actually invoke both fidelity models via "
        "_stochastic_ensemble_from_model and combine the resulting "
        "ensemble decompositions; Phase 8 Task 8.5 flipped "
        "supports_ensemble + default_strategy to ENSEMBLE."
    ),
)


_ACTIVE_UNCERTAINTY_LEARNER_CAPABILITY = UQCapability(
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes=(
        "ActiveUncertaintyLearner ranks pool samples via the "
        "opifex.uncertainty.active subsystem (named strategies: "
        "max_uncertainty / bald / expected_improvement / "
        "physics_guided). Phase 8 Task 8.3 introduced this class as the "
        "delegated user-facing active-learning surface; the duplicate-"
        "code gate forbids inlined acquisition formulas, so every "
        "score call routes through :func:`opifex.uncertainty.active.acquire`."
    ),
)


TRAINING_CAPABILITIES: dict[str, UQCapability] = {
    "trainer:UncertaintyGuidedTrainer": _UNCERTAINTY_GUIDED_TRAINER_CAPABILITY,
    "trainer:MultiFidelityUncertaintyTrainer": (_MULTI_FIDELITY_UNCERTAINTY_TRAINER_CAPABILITY),
    "trainer:ActiveUncertaintyLearner": _ACTIVE_UNCERTAINTY_LEARNER_CAPABILITY,
}


__all__ = ["TRAINING_CAPABILITIES"]
