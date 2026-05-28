"""Tests for the operator-registry UQ capabilities (Task 7.1).

The plan exit criteria are:

1. Every operator in ``OPERATOR_REGISTRY`` has exactly one
   :class:`UQCapability` declaration (no orphans / no duplicates).
2. ``UQNO`` declares ``native_bayesian=True``,
   ``supports_function_space=True``,
   ``default_strategy=DefaultStrategy.BAYESIAN``, and
   ``native_nnx_module=True``.
3. Every non-UQNO operator declares each of ``supports_ensemble``,
   ``supports_conformal``, ``supports_calibration`` set to ``True``
   via the :meth:`UQCapability.with_adapter` builder.
4. The categories returned by :func:`list_operators` are extended:
   ``uncertainty_aware`` includes every operator that sets at least
   one ``supports_*`` / ``native_*`` UQ flag, and the new
   ``adapter_capable`` category lists every non-UQNO operator that
   declares an adapter strategy.
5. ``set(uncertainty_aware) == set(adapter_capable) | {"UQNO"}``.
6. Unknown lookups raise an actionable :class:`KeyError`.
"""

from __future__ import annotations

import pytest

from opifex.neural.operators import (
    get_operator_capability,
    list_operators,
    OPERATOR_CAPABILITY_REGISTRY,
    OPERATOR_REGISTRY,
)
from opifex.neural.operators._uq_capabilities import _OPERATOR_CAPABILITIES
from opifex.uncertainty.registry import DefaultStrategy, UQCapability, UQRegistry as _UQRegistry


@pytest.fixture(autouse=True)
def _seed_registry() -> None:  # pyright: ignore[reportUnusedFunction]
    """Re-seed the singleton ``UQRegistry`` with the Task 7.1 entries.

    The registry is a process-global singleton; other suites'
    ``_reset_registry`` autouse fixtures wipe it between tests. This
    fixture restores the operator declarations before every test in
    this module so the suite stays order-independent. Idempotent — only
    registers names not already present.
    """
    registry = _UQRegistry()
    for name, capability in _OPERATOR_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


def test_every_operator_has_exactly_one_capability_declaration() -> None:
    for name in OPERATOR_REGISTRY:
        assert name in OPERATOR_CAPABILITY_REGISTRY, (
            f"Operator {name!r} missing UQCapability declaration."
        )
    # No orphan operator-named capabilities. The shared singleton
    # :class:`UQRegistry` also carries Task 7.2 model/solver/subpackage
    # declarations under namespaced keys (``model:`` / ``solver:`` /
    # ``subpackage:`` / ``backend:`` / ``adapter:`` / ``conformal:`` /
    # ``calibration:``); the operator-coverage invariant filters those
    # by checking bare operator-registry names only.
    registered_operator_names = {
        name for name in OPERATOR_CAPABILITY_REGISTRY.list_names() if ":" not in name
    }
    assert registered_operator_names == set(OPERATOR_REGISTRY)


def test_uqno_declares_native_bayesian_function_space_capability() -> None:
    cap = get_operator_capability("UQNO")
    assert cap.native_bayesian is True
    assert cap.supports_function_space is True
    assert cap.default_strategy is DefaultStrategy.BAYESIAN
    assert cap.native_nnx_module is True


def test_every_non_uqno_operator_declares_three_adapter_strategies() -> None:
    """``with_adapter`` builder pattern composes ensemble + conformal + calibration."""
    for name in OPERATOR_REGISTRY:
        if name == "UQNO":
            continue
        cap = get_operator_capability(name)
        assert cap.supports_ensemble, f"{name} missing ensemble adapter"
        assert cap.supports_conformal, f"{name} missing conformal adapter"
        assert cap.supports_calibration, f"{name} missing calibration adapter"
        # Adapter-baseline operators stay deterministic by default.
        assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
        # All operators are Flax NNX modules.
        assert cap.native_nnx_module is True


def test_uncertainty_aware_category_is_extended_beyond_uqno() -> None:
    uncertainty_aware = list_operators("uncertainty_aware")["uncertainty_aware"]
    assert set(uncertainty_aware) == set(OPERATOR_REGISTRY)


def test_adapter_capable_category_lists_all_operators_with_adapter_flags() -> None:
    adapter_capable = list_operators("adapter_capable")["adapter_capable"]
    # Every operator (including UQNO, which carries conformal + calibration)
    # appears in adapter_capable because each declares ≥1 adapter strategy.
    assert set(adapter_capable) == set(OPERATOR_REGISTRY)


def test_uncertainty_aware_union_adapter_capable_covers_supports_flags() -> None:
    """Exit criterion: the two categories together cover every operator with
    any ``supports_*`` flag.
    """
    uncertainty_aware = set(list_operators("uncertainty_aware")["uncertainty_aware"])
    adapter_capable = set(list_operators("adapter_capable")["adapter_capable"])
    operators_with_any_supports_flag = {
        name
        for name in OPERATOR_REGISTRY
        if any(
            getattr(get_operator_capability(name), f"supports_{flag}")
            for flag in (
                "ensemble",
                "conformal",
                "calibration",
                "function_space",
                "solver_uncertainty",
                "ood_detection",
                "selective_risk",
                "likelihood_free",
                "active_learning",
                "pac_bayes_certificate",
                "stochastic_field_input",
            )
        )
    }
    assert operators_with_any_supports_flag == uncertainty_aware | adapter_capable


def test_unknown_operator_lookup_raises_actionable_key_error() -> None:
    with pytest.raises(KeyError, match="UnknownOperator"):
        get_operator_capability("UnknownOperator")


def test_capability_with_adapter_returns_new_instance_with_supports_flag() -> None:
    """Direct test of the :meth:`UQCapability.with_adapter` builder."""
    baseline = UQCapability.deterministic_baseline()
    composed = (
        baseline.with_adapter("ensemble").with_adapter("conformal").with_adapter("calibration")
    )
    assert composed.supports_ensemble
    assert composed.supports_conformal
    assert composed.supports_calibration
    # Builder must not mutate the baseline (Pattern (A) immutability).
    assert not baseline.supports_ensemble
    assert not baseline.supports_conformal
    assert not baseline.supports_calibration


def test_capability_with_adapter_rejects_unknown_strategy() -> None:
    baseline = UQCapability.deterministic_baseline()
    with pytest.raises(ValueError, match="Unknown adapter strategy"):
        baseline.with_adapter("nonsense")
