"""UQ capability declarations for the training surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.training.__init__``.

Phase 7 records :class:`UncertaintyGuidedTrainer` and
:class:`MultiFidelityUncertaintyTrainer` honestly: both currently use a
hardcoded ``jax.random.PRNGKey(42)`` mock predictor and never call the
wrapped model (``basic_trainer.py:1168/1274``). The Phase 7 declaration
therefore sets ``default_strategy=DefaultStrategy.UNSUPPORTED`` with a
note pointing at Phase 8 Task 8.3 (rewrite to call real uncertainty
quantifiers) and Phase 8 Task 8.5 (capability-flag flip after the
rewrite's TDD step passes).

Plan reference: ``07-phase-registry-docs-examples.md`` lines 644-657
(audit gap noted 2026-05-20).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_UNCERTAINTY_GUIDED_TRAINER_CAPABILITY = UQCapability(
    supports_active_learning=False,
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "UncertaintyGuidedTrainer is named for active-learning-driven "
        "adaptive training, but the current implementation uses a "
        "hardcoded jax.random.PRNGKey(42) mock prediction and never "
        "calls the wrapped uncertainty quantifier (basic_trainer.py:1168). "
        "Phase 8 Task 8.3 rewrites the trainer to call the real "
        "quantifier; Phase 8 Task 8.5 flips supports_active_learning + "
        "default_strategy. Declared UNSUPPORTED in Phase 7."
    ),
)


_MULTI_FIDELITY_UNCERTAINTY_TRAINER_CAPABILITY = UQCapability(
    native_distributional=False,
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "MultiFidelityUncertaintyTrainer is named for multi-fidelity "
        "uncertainty propagation, but the current implementation uses "
        "hardcoded jax.random.PRNGKey(42)/PRNGKey(43) mock predictions "
        "rather than calling the wrapped models (basic_trainer.py:1280). "
        "Phase 8 Task 8.3 rewrites the trainer to call the real "
        "models / quantifier; Phase 8 Task 8.5 flips "
        "native_distributional + default_strategy. Declared UNSUPPORTED "
        "in Phase 7."
    ),
)


TRAINING_CAPABILITIES: dict[str, UQCapability] = {
    "trainer:UncertaintyGuidedTrainer": _UNCERTAINTY_GUIDED_TRAINER_CAPABILITY,
    "trainer:MultiFidelityUncertaintyTrainer": (_MULTI_FIDELITY_UNCERTAINTY_TRAINER_CAPABILITY),
}


__all__ = ["TRAINING_CAPABILITIES"]
