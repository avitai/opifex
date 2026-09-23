"""UQ capability declarations: frozen, hashable metadata with validated flag combinations."""

import dataclasses
from enum import StrEnum

import pytest

from opifex.core.capabilities import DefaultStrategy, UQCapability


def test_uq_capability_is_pattern_a_frozen_slotted_dataclass() -> None:
    cap = UQCapability.deterministic_baseline()
    assert dataclasses.is_dataclass(UQCapability)
    assert hasattr(UQCapability, "__slots__")
    assert not hasattr(cap, "__dict__")
    field_names = {f.name for f in dataclasses.fields(UQCapability)}
    expected = {
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
        "native_jax_kernel",
        "native_nnx_module",
        "requires_graph_adapter",
        "default_strategy",
        "source_package",
        "notes",
    }
    assert expected <= field_names


def test_uq_capability_is_hashable() -> None:
    cap_a = UQCapability.deterministic_baseline()
    cap_b = UQCapability.deterministic_baseline()
    assert hash(cap_a) == hash(cap_b)


def test_default_strategy_strenum_membership() -> None:
    assert issubclass(DefaultStrategy, StrEnum)
    expected_members = {
        "DETERMINISTIC",
        "BAYESIAN",
        "VARIATIONAL",
        "ENSEMBLE",
        "MC_DROPOUT",
        "VBLL",
        "LAPLACE",
        "SNGP",
        "SWAG",
        "CONFORMAL",
        "CALIBRATION",
        "LIKELIHOOD_FREE_SBI",
        "ACTIVE_LEARNING",
        "PAC_BAYES",
        "POLYNOMIAL_CHAOS",
        "KARHUNEN_LOEVE",
        "STOCHASTIC_GALERKIN",
        "PROBABILISTIC_NUMERICS",
        "UNSUPPORTED",
    }
    actual = {member.name for member in DefaultStrategy}
    assert expected_members <= actual


def test_default_strategy_string_values_are_snake_case() -> None:
    assert DefaultStrategy.DETERMINISTIC == "deterministic"
    assert DefaultStrategy.LIKELIHOOD_FREE_SBI == "likelihood_free_sbi"


def test_default_strategy_includes_probabilistic_numerics_foundation_members() -> None:
    """Probabilistic-numerics foundation adds three strategy members."""
    assert DefaultStrategy.RANDOMIZED_LINALG == "randomized_linalg"
    assert DefaultStrategy.BAYESIAN_QUADRATURE == "bayesian_quadrature"
    assert DefaultStrategy.STATE_SPACE_FILTERING == "state_space_filtering"


def test_deterministic_baseline_has_all_support_flags_false() -> None:
    cap = UQCapability.deterministic_baseline()
    for field in dataclasses.fields(UQCapability):
        if field.name.startswith("supports_") or field.name.startswith("native_"):
            if field.name == "native_jax_kernel":
                continue  # not a support flag
            value = getattr(cap, field.name)
            if isinstance(value, bool):
                assert value is False, f"{field.name} must be False in baseline"
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC


def test_pure_metric_kernel_can_declare_jax_native_only() -> None:
    cap = UQCapability(
        native_jax_kernel=True,
        native_nnx_module=False,
        default_strategy=DefaultStrategy.DETERMINISTIC,
        source_package="opifex",
    )
    assert cap.native_jax_kernel is True
    assert cap.native_nnx_module is False


def test_stateful_trainable_model_can_declare_nnx_native() -> None:
    cap = UQCapability(
        native_nnx_module=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="opifex",
    )
    assert cap.native_nnx_module is True


def test_requires_graph_adapter_requires_non_empty_notes() -> None:
    with pytest.raises(ValueError, match=r"notes"):
        UQCapability(
            requires_graph_adapter=True,
            default_strategy=DefaultStrategy.BAYESIAN,
            source_package="opifex",
            notes="",
        )
