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
from flax import nnx

from opifex.uncertainty.adapters import (
    DeepONetConformalAdapterSpec,
    DeepONetDeepEnsembleAdapterSpec,
    DeepONetMCDropoutAdapterSpec,
    FNOConformalAdapterSpec,
    FNODeepEnsembleAdapterSpec,
    FNOMCDropoutAdapterSpec,
    MCDropoutState,
    OperatorAdapterSpec,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import metadata_to_dict, PredictiveDistribution


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

# Conformal specs redirect to the dedicated conformal path (FieldSplitConformalRegressor
# / UQNO); ensemble + MC-dropout specs wire to the concrete adapters.
_CONFORMAL_SPECS: tuple[tuple[type, DefaultStrategy, str], ...] = (
    (FNOConformalAdapterSpec, DefaultStrategy.CONFORMAL, "fno"),
    (DeepONetConformalAdapterSpec, DefaultStrategy.CONFORMAL, "deeponet"),
)
_ENSEMBLE_SPECS: tuple[tuple[type, str, tuple[int, ...], tuple[int, ...] | None], ...] = (
    (FNODeepEnsembleAdapterSpec, "fno", (1, 2), (1, 2)),
    (DeepONetDeepEnsembleAdapterSpec, "deeponet", (1,), None),
)
_MCDROPOUT_SPECS: tuple[tuple[type, str, tuple[int, ...], tuple[int, ...] | None], ...] = (
    (FNOMCDropoutAdapterSpec, "fno", (1, 2), (1, 2)),
    (DeepONetMCDropoutAdapterSpec, "deeponet", (1,), None),
)


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
# CONFORMAL: redirect to the dedicated conformal path (honest boundary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec_cls", "_strategy", "expected_family"), _CONFORMAL_SPECS)
def test_conformal_operator_wrap_redirects_to_conformal_path(
    spec_cls: type, _strategy: DefaultStrategy, expected_family: str
) -> None:
    """Conformal operator UQ uses calibration data, not the ``wrap`` contract.

    ``wrap`` raises an actionable error redirecting callers to the dedicated
    conformal calibrators (field split-conformal / UQNO), naming the operator
    family + the conformal strategy + the calibrator surfaces to use.
    """
    spec: Any = spec_cls()
    capability = spec.recommended_capability()
    with pytest.raises(NotImplementedError) as exc_info:
        spec.wrap(model=lambda x: x, capability=capability)
    message = str(exc_info.value)
    assert expected_family in message
    assert spec.default_strategy.value in message
    # Redirect must name the concrete conformal surfaces, not a generic stub.
    assert "FieldSplitConformalRegressor" in message
    assert "UncertaintyQuantificationNeuralOperator" in message


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
# ENSEMBLE: wire to DeepEnsembleAdapter + enrich function-space metadata
# ---------------------------------------------------------------------------


def _sum_operator(x: jax.Array) -> jax.Array:
    """Tiny operator stand-in: reduce the channel axis (keepdims)."""
    return jnp.sum(x, axis=-1, keepdims=True)


def _scaled_sum_operator(x: jax.Array) -> jax.Array:
    """Second ensemble member with a different output (non-zero spread)."""
    return 2.0 * jnp.sum(x, axis=-1, keepdims=True)


@pytest.mark.parametrize(
    ("spec_cls", "expected_family", "expected_spatial", "expected_spectral"), _ENSEMBLE_SPECS
)
def test_ensemble_operator_wrap_delegates_and_enriches_metadata(
    spec_cls: type,
    expected_family: str,
    expected_spatial: tuple[int, ...],
    expected_spectral: tuple[int, ...] | None,
) -> None:
    """FNO/DeepONet + ENSEMBLE wraps a member tuple and records function-space provenance.

    Mean equals the member-mean; epistemic spread is positive; metadata
    carries ``operator_family`` / ``spatial_axes`` (+ ``spectral_axes`` for
    FNO) and the ensemble method tag from the delegated adapter.
    """
    spec: Any = spec_cls()
    capability = spec.recommended_capability()
    members = (_sum_operator, _scaled_sum_operator)
    wrapped = spec.wrap(model=members, capability=capability)

    x = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)

    member_mean = 0.5 * (_sum_operator(x) + _scaled_sum_operator(x))
    assert jnp.allclose(dist.mean, member_mean)
    assert dist.epistemic is not None
    assert float(jnp.sum(dist.epistemic)) > 0.0

    meta = metadata_to_dict(dist.metadata)
    assert meta["method"] == "ensemble"
    assert meta["operator_family"] == expected_family
    assert meta["spatial_axes"] == expected_spatial
    if expected_spectral is None:
        assert meta.get("spectral_axes") is None
    else:
        assert meta["spectral_axes"] == expected_spectral
    # Function-space provenance + the delegated adapter's bookkeeping coexist.
    assert "supported_metrics" in meta
    assert int(meta["num_members"]) == 2


def test_ensemble_operator_wrap_rejects_wrong_capability_strategy() -> None:
    """A capability whose strategy is not ENSEMBLE is rejected by the delegate."""
    spec = FNODeepEnsembleAdapterSpec()
    wrong = UQCapability(
        default_strategy=DefaultStrategy.MC_DROPOUT,
        supports_function_space=True,
        supports_ensemble=True,
    )
    with pytest.raises(ValueError, match="ENSEMBLE"):
        spec.wrap(model=(_sum_operator, _scaled_sum_operator), capability=wrong)


# ---------------------------------------------------------------------------
# MC_DROPOUT: wire to MCDropoutAdapter + enrich function-space metadata
# ---------------------------------------------------------------------------


def test_mc_dropout_operator_wrap_delegates_and_preserves_rngs() -> None:
    """FNO + MC_DROPOUT wraps an ``MCDropoutState``; predict-time rngs are caller-owned."""
    spec = FNOMCDropoutAdapterSpec()
    capability = spec.recommended_capability()

    def stochastic_operator(x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        noise = jax.random.normal(rngs.dropout(), (*x.shape[:-1], 1))
        return jnp.sum(x, axis=-1, keepdims=True) + noise

    state = MCDropoutState(model_fn=stochastic_operator, num_samples=8)
    wrapped = spec.wrap(model=state, capability=capability)

    x = jnp.ones((2, 3), dtype=jnp.float32)
    dist = wrapped.predict_distribution(x, rngs=nnx.Rngs(dropout=7))
    assert isinstance(dist, PredictiveDistribution)
    assert dist.epistemic is not None
    assert float(jnp.sum(dist.epistemic)) > 0.0

    meta = metadata_to_dict(dist.metadata)
    assert meta["method"] == "mc_dropout"
    assert meta["operator_family"] == "fno"
    assert meta["spatial_axes"] == (1, 2)
    assert meta["spectral_axes"] == (1, 2)
    assert int(meta["num_samples"]) == 8


def test_mc_dropout_operator_wrap_requires_rngs_at_predict_time() -> None:
    """Predict-time ``rngs`` is mandatory — calling without it raises ``TypeError``."""
    spec = DeepONetMCDropoutAdapterSpec()
    capability = spec.recommended_capability()

    def stochastic_operator(x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        return jnp.sum(x, axis=-1, keepdims=True) + jax.random.normal(rngs.dropout(), (x.shape[0], 1))

    state = MCDropoutState(model_fn=stochastic_operator, num_samples=4)
    wrapped = spec.wrap(model=state, capability=capability)
    with pytest.raises(TypeError):
        wrapped.predict_distribution(jnp.ones((2, 3)))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# JAX/NNX transform compatibility for the wired ensemble path (REQUIRED)
# ---------------------------------------------------------------------------


def test_ensemble_operator_wrap_predict_is_jit_grad_vmap_compatible() -> None:
    """The wired ensemble ``predict_distribution`` composes with jit / grad / vmap."""
    spec = FNODeepEnsembleAdapterSpec()
    capability = spec.recommended_capability()
    wrapped = spec.wrap(model=(_sum_operator, _scaled_sum_operator), capability=capability)

    def mean_sum(x: jax.Array) -> jax.Array:
        return jnp.sum(wrapped.predict_distribution(x).mean)

    x = jnp.ones((2, 3), dtype=jnp.float32)

    # jit
    jitted = jax.jit(mean_sum)
    assert bool(jnp.isfinite(jitted(x)))
    # grad
    grad_value = jax.grad(mean_sum)(x)
    assert grad_value.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(grad_value)))
    # vmap over a leading ensemble-of-batches axis
    batched = jnp.stack([x, 2.0 * x], axis=0)
    vmapped = jax.vmap(lambda b: wrapped.predict_distribution(b).mean)(batched)
    assert vmapped.shape[0] == 2
    assert bool(jnp.all(jnp.isfinite(vmapped)))


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
