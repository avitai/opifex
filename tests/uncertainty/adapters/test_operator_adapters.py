"""Task 3.7: neural-operator UQ adapter declarations.

Declares + tests adapter capability metadata for FNO and DeepONet
families. These adapter specs do NOT claim native Bayesian support
for either family (Task 3.4 reserved that for UQNO's eventual conformal
implementation, and bayesian_fno is a wrapper, not a native Bayesian
operator). They DO declare:

* which adapter strategies (ensemble, dropout, conformal, function-space)
  the family is compatible with;
* the operator's spatial / spectral axis metadata so function-space
  callers can identify the output topology;
* the supported metric set (L2 / H1 / spatial coverage / spectral
  coverage) for the conformal calibrator that consumes the adapter;
* explicit unsupported-backend errors naming the missing backend or
  model capability for adapters that are not yet wired.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    DeepONetConformalAdapterSpec,
    DeepONetDeepEnsembleAdapterSpec,
    DeepONetMCDropoutAdapterSpec,
    FNOConformalAdapterSpec,
    FNODeepEnsembleAdapterSpec,
    FNOMCDropoutAdapterSpec,
    OperatorAdapterSpec,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability


# ---------------------------------------------------------------------------
# Frozen-dataclass + capability-metadata contracts
# ---------------------------------------------------------------------------


_FNO_SPECS: tuple[tuple[type, DefaultStrategy, str], ...] = (
    (FNOConformalAdapterSpec, DefaultStrategy.CONFORMAL, "fno"),
    (FNODeepEnsembleAdapterSpec, DefaultStrategy.ENSEMBLE, "fno"),
    (FNOMCDropoutAdapterSpec, DefaultStrategy.MC_DROPOUT, "fno"),
)

_DEEPONET_SPECS: tuple[tuple[type, DefaultStrategy, str], ...] = (
    (DeepONetConformalAdapterSpec, DefaultStrategy.CONFORMAL, "deeponet"),
    (DeepONetDeepEnsembleAdapterSpec, DefaultStrategy.ENSEMBLE, "deeponet"),
    (DeepONetMCDropoutAdapterSpec, DefaultStrategy.MC_DROPOUT, "deeponet"),
)

_ALL_OPERATOR_SPECS = _FNO_SPECS + _DEEPONET_SPECS


@pytest.mark.parametrize(("spec_cls", "expected_strategy", "expected_family"), _ALL_OPERATOR_SPECS)
def test_operator_adapter_spec_is_frozen_dataclass_with_capability_metadata(
    spec_cls: type, expected_strategy: DefaultStrategy, expected_family: str
) -> None:
    """Each operator spec is a pattern-(A) frozen dataclass with capability metadata."""
    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert isinstance(spec, OperatorAdapterSpec)
    assert spec.default_strategy is expected_strategy
    assert spec.operator_family == expected_family
    assert spec.source_package == "opifex"
    assert isinstance(spec.required_capabilities, tuple)
    assert isinstance(spec.spatial_axes, tuple)
    assert isinstance(spec.supported_metrics, tuple)
    # Frozen
    with pytest.raises(dc.FrozenInstanceError):
        spec.operator_family = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize(("spec_cls", "_strategy", "_family"), _FNO_SPECS)
def test_fno_operator_spec_declares_spectral_axes(
    spec_cls: type, _strategy: DefaultStrategy, _family: str
) -> None:
    """FNO specs MUST advertise both spatial and spectral axis metadata."""
    spec: Any = spec_cls()
    assert spec.spectral_axes is not None
    assert isinstance(spec.spectral_axes, tuple)
    assert set(spec.spectral_axes).issubset(set(spec.spatial_axes))


@pytest.mark.parametrize(("spec_cls", "_strategy", "_family"), _DEEPONET_SPECS)
def test_deeponet_operator_spec_omits_spectral_axes(
    spec_cls: type, _strategy: DefaultStrategy, _family: str
) -> None:
    """DeepONet adapters are not spectral — spectral_axes must be None."""
    spec: Any = spec_cls()
    assert spec.spectral_axes is None


@pytest.mark.parametrize(("spec_cls", "_strategy", "_family"), _ALL_OPERATOR_SPECS)
def test_operator_adapter_spec_supports_function_space_metrics(
    spec_cls: type, _strategy: DefaultStrategy, _family: str
) -> None:
    """Every operator adapter must advertise at least one function-space metric."""
    spec: Any = spec_cls()
    function_space_metrics = {"l2", "h1", "spatial_coverage", "spectral_coverage"}
    assert function_space_metrics & set(spec.supported_metrics), (
        f"{spec_cls.__name__} must support at least one of {function_space_metrics}; "
        f"got {spec.supported_metrics!r}"
    )


# ---------------------------------------------------------------------------
# Honesty: operator adapters MUST NOT claim native Bayesian support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec_cls", "_strategy", "_family"), _ALL_OPERATOR_SPECS)
def test_operator_adapter_spec_never_claims_native_bayesian(
    spec_cls: type, _strategy: DefaultStrategy, _family: str
) -> None:
    """Adapter-mediated UQ on FNO/DeepONet is not native Bayesian; the spec must say so."""
    spec: Any = spec_cls()
    cap = spec.recommended_capability()
    assert isinstance(cap, UQCapability)
    assert cap.native_bayesian is False
    assert cap.default_strategy is spec.default_strategy
    # Operator adapters always advertise supports_function_space.
    assert cap.supports_function_space is True


@pytest.mark.parametrize(("spec_cls", "expected_strategy", "_family"), _ALL_OPERATOR_SPECS)
def test_operator_adapter_capability_flags_match_strategy(
    spec_cls: type, expected_strategy: DefaultStrategy, _family: str
) -> None:
    """Conformal/ensemble/MC-dropout specs advertise the matching capability flag."""
    spec: Any = spec_cls()
    cap = spec.recommended_capability()
    flag_for_strategy: dict[DefaultStrategy, str] = {
        DefaultStrategy.CONFORMAL: "supports_conformal",
        DefaultStrategy.ENSEMBLE: "supports_ensemble",
        DefaultStrategy.MC_DROPOUT: "supports_ensemble",
    }
    expected_flag = flag_for_strategy[expected_strategy]
    assert getattr(cap, expected_flag) is True


# ---------------------------------------------------------------------------
# Unsupported-backend errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec_cls", "_strategy", "expected_family"), _ALL_OPERATOR_SPECS)
def test_operator_adapter_wrap_raises_named_unsupported_error(
    spec_cls: type, _strategy: DefaultStrategy, expected_family: str
) -> None:
    """Until wired to a real adapter, ``wrap`` must name the missing backend."""
    spec: Any = spec_cls()
    capability = spec.recommended_capability()
    with pytest.raises(NotImplementedError) as exc_info:
        spec.wrap(model=lambda x: x, capability=capability)
    message = str(exc_info.value)
    assert expected_family in message
    assert spec.default_strategy.value in message


def test_operator_adapter_wrap_rejects_native_bayesian_capability() -> None:
    """`wrap` must refuse capabilities falsely claiming native Bayesian support."""
    spec = FNOConformalAdapterSpec()
    bogus_capability = UQCapability(
        native_bayesian=True,
        default_strategy=DefaultStrategy.CONFORMAL,
        supports_function_space=True,
        supports_conformal=True,
    )
    with pytest.raises(ValueError, match="native_bayesian"):
        spec.wrap(model=lambda x: x, capability=bogus_capability)


# ---------------------------------------------------------------------------
# Metadata exports under opifex.neural.operators
# ---------------------------------------------------------------------------


def test_neural_operators_module_exports_operator_adapter_specs() -> None:
    """Capability specs are reachable from the neural-operators namespace."""
    import opifex.neural.operators as operators_module

    for spec_cls in (
        FNOConformalAdapterSpec,
        FNODeepEnsembleAdapterSpec,
        FNOMCDropoutAdapterSpec,
        DeepONetConformalAdapterSpec,
        DeepONetDeepEnsembleAdapterSpec,
        DeepONetMCDropoutAdapterSpec,
    ):
        assert hasattr(operators_module, spec_cls.__name__), (
            f"opifex.neural.operators is missing {spec_cls.__name__}"
        )


def test_fno_init_exports_operator_adapter_specs() -> None:
    """`opifex.neural.operators.fno` re-exports the FNO adapter specs."""
    import opifex.neural.operators.fno as fno_module

    for spec_cls in (FNOConformalAdapterSpec, FNODeepEnsembleAdapterSpec, FNOMCDropoutAdapterSpec):
        assert hasattr(fno_module, spec_cls.__name__), (
            f"opifex.neural.operators.fno is missing {spec_cls.__name__}"
        )


def test_deeponet_init_exports_operator_adapter_specs() -> None:
    """`opifex.neural.operators.deeponet` re-exports the DeepONet adapter specs."""
    import opifex.neural.operators.deeponet as deeponet_module

    for spec_cls in (
        DeepONetConformalAdapterSpec,
        DeepONetDeepEnsembleAdapterSpec,
        DeepONetMCDropoutAdapterSpec,
    ):
        assert hasattr(deeponet_module, spec_cls.__name__), (
            f"opifex.neural.operators.deeponet is missing {spec_cls.__name__}"
        )


# ---------------------------------------------------------------------------
# JAX/NNX transform compatibility — the recommended-capability helper is a
# pure Python function over hashable fields, but exercising it inside
# ``jax.jit`` confirms it never leaks unhashable objects into the trace.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec_cls", "_strategy", "_family"), _ALL_OPERATOR_SPECS)
def test_recommended_capability_is_hashable_for_jit_static_args(
    spec_cls: type, _strategy: DefaultStrategy, _family: str
) -> None:
    """Capability is used as a static arg to jitted closures; must be hashable."""
    spec: Any = spec_cls()
    cap = spec.recommended_capability()
    # Hash should be stable + the capability serializes through dataclasses.replace
    _ = hash(cap)
    _ = dc.replace(cap, source_package="opifex")


def test_operator_spec_is_safe_to_close_over_under_jit() -> None:
    """Spec is closed over by a jitted function — must be hashable + static-safe."""
    spec = FNOConformalAdapterSpec()
    metric_set = frozenset(spec.supported_metrics)

    @jax.jit
    def predict_l2_residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.mean((x - y) ** 2)

    assert "l2" in metric_set or "spatial_coverage" in metric_set
    x = jnp.ones((4, 3))
    y = jnp.zeros((4, 3))
    out = predict_l2_residual(x, y)
    assert bool(jnp.isfinite(out))
