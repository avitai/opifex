"""Phase 1 Task 1.5 — inference backend protocol and base result types.

Sibling Reuse Gate decisions:

* Artifex ``SamplingAlgorithm`` (``../artifex/src/artifex/generative_models/
  core/sampling/base.py``) and ``BlackJAXSamplerState``
  (``blackjax_samplers.py:64``) are reused directly — the protocol must accept
  them without modification so Phase 2 Task 2.5 can implement over them.
* CalibraX ``StatisticalResult``/``BenchmarkResult`` are reused for typed
  diagnostics; the backend result container only adds UQ-specific provenance.

Container patterns: ``BackendDiagnostics`` and fitted ``BackendResult`` are
pattern (B) (array fields flow through transforms); ``InferenceBackendSpec``
metadata is pattern (A).
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.sampling.blackjax_samplers import (
    BlackJAXSamplerState,
)

from opifex.uncertainty.inference_backends import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    UnsupportedBackendError,
)
from opifex.uncertainty.types import PredictiveDistribution


if TYPE_CHECKING:
    from flax import nnx


def _make_predictive_distribution() -> PredictiveDistribution:
    return PredictiveDistribution(mean=jnp.zeros(4))


class _ToyBlackJAXBackend:
    """Implements ``InferenceBackendProtocol`` to verify structural conformance."""

    def fit(
        self,
        target_log_prob: Any,
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        del target_log_prob, rngs
        state = BlackJAXSamplerState(
            x=jnp.zeros(4),
            sampler_state=object(),
            key=jax.random.PRNGKey(0),
        )
        return BackendResult(
            sampler_state=state,
            diagnostics=BackendDiagnostics(
                ess=jnp.array([100.0]),
                rhat=jnp.array([1.01]),
                acceptance_rate=jnp.array([0.95]),
            ),
        )

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        del x, rngs
        return _make_predictive_distribution()

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        del x, rngs
        return _make_predictive_distribution()


def test_inference_backend_protocol_accepts_conforming_class() -> None:
    instance: Any = _ToyBlackJAXBackend()
    assert isinstance(instance, InferenceBackendProtocol)


def test_inference_backend_protocol_rejects_class_missing_posterior_predictive() -> None:
    class MissingPosteriorPredictive:
        def fit(self, target_log_prob: Any, *, rngs: nnx.Rngs) -> BackendResult:
            del target_log_prob, rngs
            return BackendResult(sampler_state=None, diagnostics=BackendDiagnostics())

        def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
            del x, rngs
            return _make_predictive_distribution()

    instance: Any = MissingPosteriorPredictive()
    assert not isinstance(instance, InferenceBackendProtocol)


def test_backend_result_carries_artifex_blackjax_sampler_state_unchanged() -> None:
    state = BlackJAXSamplerState(x=jnp.zeros(4), sampler_state=object(), key=jax.random.PRNGKey(0))
    result = BackendResult(sampler_state=state, diagnostics=BackendDiagnostics())
    assert result.sampler_state is state


def test_backend_diagnostics_uses_pattern_b_struct_dataclass() -> None:
    diagnostics = BackendDiagnostics()
    assert hasattr(diagnostics, "__slots__")
    assert hasattr(diagnostics, "replace")
    flat, treedef = jax.tree_util.tree_flatten(diagnostics)
    rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
    assert isinstance(rebuilt, BackendDiagnostics)


def test_backend_diagnostics_default_fields_are_none() -> None:
    diagnostics = BackendDiagnostics()
    assert diagnostics.ess is None
    assert diagnostics.rhat is None
    assert diagnostics.acceptance_rate is None
    assert diagnostics.divergences is None
    assert diagnostics.step_size is None
    assert diagnostics.tree_depth is None


def test_inference_backend_spec_is_pattern_a_frozen_dataclass() -> None:
    spec = InferenceBackendSpec(
        name="blackjax_nuts",
        family="MCMC",
        sampler_names=("HMC", "NUTS"),
        source_package="artifex",
    )
    assert dataclasses.is_dataclass(InferenceBackendSpec)
    assert hasattr(InferenceBackendSpec, "__slots__")
    assert not hasattr(spec, "__dict__")
    assert hash(spec) == hash(
        InferenceBackendSpec(
            name="blackjax_nuts",
            family="MCMC",
            sampler_names=("HMC", "NUTS"),
            source_package="artifex",
        )
    )


def test_inference_backend_spec_rejects_list_for_sampler_names() -> None:
    """GUIDE_ALIGNMENT item 22a — static sequence fields must be tuples."""
    with pytest.raises(TypeError):
        # mypy-equivalent: a list is not assignable to tuple, but dataclass
        # accepts it silently; we enforce tuple at construction time.
        InferenceBackendSpec(
            name="x",
            family="MCMC",
            sampler_names=["HMC"],  # type: ignore[arg-type]
            source_package="opifex",
        )


def test_unsupported_backend_error_message_includes_backend_name() -> None:
    err = UnsupportedBackendError("flowjax", reason="not installed")
    assert "flowjax" in str(err)
    assert "not installed" in str(err)
