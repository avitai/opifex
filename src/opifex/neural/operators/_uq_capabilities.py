"""Per-operator UQ capability declarations (Task 7.1).

Static, module-level constants — no import-time mutable side effects
beyond the constants themselves (Rule 13). Imported by
``opifex.neural.operators.__init__`` after the operator registry is
built; that module then registers each declaration into the
sibling-backed :class:`UQRegistry` singleton.

UQNO is the sole native-Bayesian operator. The other 18 entries are
deterministic baselines with adapter strategies layered on via
:meth:`UQCapability.with_adapter`, reflecting the Phase 3 Task 3.7
adapter inventory (ensemble / conformal / calibration).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


def _adapter_baseline() -> UQCapability:
    """Deterministic-NNX baseline that admits all three adapter strategies."""
    return (
        UQCapability(
            default_strategy=DefaultStrategy.DETERMINISTIC,
            native_nnx_module=True,
        )
        .with_adapter("ensemble")
        .with_adapter("conformal")
        .with_adapter("calibration")
    )


# UQNO is the only operator with native Bayesian uncertainty.
_UQNO_CAPABILITY = UQCapability(
    native_bayesian=True,
    supports_function_space=True,
    supports_conformal=True,
    supports_calibration=True,
    default_strategy=DefaultStrategy.BAYESIAN,
    native_nnx_module=True,
    notes=(
        "UQNO (uncertainty-quantification neural operator) owns a "
        "Bayesian posterior over its own parameters and predicts over "
        "function-valued outputs."
    ),
)


# All 18 non-UQNO operators get the same deterministic adapter baseline.
# Keeping them as separate names lets future per-operator overrides
# (e.g. ``supports_function_space=True`` for SFNO once function-space
# conformal lands) attach without disturbing the others.
_OPERATOR_CAPABILITIES: dict[str, UQCapability] = {
    # Traditional operators
    "FNO": _adapter_baseline(),
    "DeepONet": _adapter_baseline(),
    "PINO": _adapter_baseline(),
    # FNO variants
    "TFNO": _adapter_baseline(),
    "UFNO": _adapter_baseline(),
    "SFNO": _adapter_baseline(),
    "LocalFNO": _adapter_baseline(),
    "AM-FNO": _adapter_baseline(),
    "MS-FNO": _adapter_baseline(),
    # DeepONet variants
    "FourierDeepONet": _adapter_baseline(),
    "AdaptiveDeepONet": _adapter_baseline(),
    "MultiPhysicsDeepONet": _adapter_baseline(),
    # Specialised operators
    "GINO": _adapter_baseline(),
    "MGNO": _adapter_baseline(),
    "UQNO": _UQNO_CAPABILITY,
    "LNO": _adapter_baseline(),
    "WNO": _adapter_baseline(),
    "GNO": _adapter_baseline(),
    "OperatorNet": _adapter_baseline(),
}


def _categories_from_capabilities(
    capabilities: dict[str, UQCapability],
) -> dict[str, tuple[str, ...]]:
    """Derive registry categories from the static capability table.

    The two categories the Task 7.1 plan wants exposed:

    * ``"uncertainty_aware"`` — operators with at least one ``supports_*``
      or ``native_*`` UQ flag set to ``True``.
    * ``"adapter_capable"`` — operators with at least one adapter
      strategy declared (ensemble / conformal / calibration).

    Returns a frozen tuple per category so downstream callers can't
    mutate the lists (Pattern (A) immutability per GUIDE_ALIGNMENT
    item 22a).
    """
    adapter_capable = tuple(
        name
        for name, cap in capabilities.items()
        if cap.supports_ensemble or cap.supports_conformal or cap.supports_calibration
    )
    uncertainty_aware = tuple(
        name
        for name, cap in capabilities.items()
        if any(
            getattr(cap, attr)
            for attr in (
                "native_bayesian",
                "native_distributional",
                "supports_ensemble",
                "supports_conformal",
                "supports_calibration",
                "supports_function_space",
                "supports_solver_uncertainty",
                "supports_ood_detection",
                "supports_selective_risk",
                "supports_likelihood_free",
                "supports_active_learning",
                "supports_pac_bayes_certificate",
                "supports_stochastic_field_input",
            )
        )
    )
    return {
        "uncertainty_aware": uncertainty_aware,
        "adapter_capable": adapter_capable,
    }


OPERATOR_CAPABILITY_CATEGORIES: dict[str, tuple[str, ...]] = _categories_from_capabilities(
    _OPERATOR_CAPABILITIES
)


__all__ = ["OPERATOR_CAPABILITY_CATEGORIES", "_OPERATOR_CAPABILITIES"]
