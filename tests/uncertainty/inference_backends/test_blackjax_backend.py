"""Tests for the BlackJAX inference backend.

``BlackJAXBackend`` is a thin adapter over Artifex's
``hmc_sampling`` / ``nuts_sampling`` / ``mala_sampling`` wrappers that
conforms to :class:`opifex.uncertainty.inference_backends.InferenceBackendProtocol`.
It must:

* delegate to Artifex's sampler functions (no direct ``blackjax`` import);
* route every RNG argument through
  ``artifex.generative_models.core.rng.extract_rng_key``;
* return a :class:`BackendResult` carrying the raw posterior samples and a
  populated :class:`BackendDiagnostics`;
* convert posterior samples to a :class:`PredictiveDistribution` via
  ``predict_distribution`` / ``posterior_predictive``;
* raise the canonical Phase 1 exceptions for unsupported samplers
  (``NotImplementedError`` with a name + actionable message).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.rng import (
    extract_rng_key as artifex_extract_rng_key,
)
from flax import nnx

from opifex.uncertainty.inference_backends import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendProtocol,
    UnsupportedBackendError,
)
from opifex.uncertainty.inference_backends.blackjax import BlackJAXBackend
from opifex.uncertainty.types import PredictiveDistribution


def _gaussian_log_density(x: jax.Array) -> jax.Array:
    """Standard-normal log density (unnormalized constant suffices for MCMC)."""
    return -0.5 * jnp.sum(x * x)


def _make_backend(method: str = "nuts") -> BlackJAXBackend:
    return BlackJAXBackend(
        target_log_prob=_gaussian_log_density,
        init_state=jnp.zeros(3),
        n_samples=20,
        n_burnin=5,
        method=method,
    )


def test_blackjax_backend_satisfies_inference_backend_protocol() -> None:
    backend: Any = _make_backend()
    assert isinstance(backend, InferenceBackendProtocol)


def test_blackjax_backend_fit_returns_backend_result_with_samples() -> None:
    backend = _make_backend("nuts")
    result = backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert isinstance(result, BackendResult)
    samples = result.sampler_state
    assert isinstance(samples, jax.Array)
    assert samples.shape == (20, 3)
    assert jnp.all(jnp.isfinite(samples))


def test_blackjax_backend_returns_typed_backend_diagnostics() -> None:
    backend = _make_backend("nuts")
    result = backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert isinstance(result.diagnostics, BackendDiagnostics)


def test_blackjax_backend_diagnostics_surface_post_hoc_ess() -> None:
    """``BackendDiagnostics.ess`` is computed post-hoc from the samples array.

    Artifex's sampler wrappers discard BlackJAX's per-step ``info`` dict, so
    acceptance / divergences / step-size / tree-depth are unavailable. ESS,
    however, is computable from a single chain via the autocorrelation
    method — the backend MUST populate that field rather than returning an
    empty diagnostics record.
    """
    backend = _make_backend("nuts")
    result = backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert result.diagnostics.ess is not None
    # ESS is per-parameter; init_state has 3 dims, n_samples=20.
    assert result.diagnostics.ess.shape == (3,)
    # ESS must be positive and bounded above by n_samples.
    assert jnp.all(result.diagnostics.ess > 0)
    assert jnp.all(result.diagnostics.ess <= 20)


def test_compute_ess_is_jit_compatible() -> None:
    """``_compute_ess`` is a pure JAX function and must trace under ``jax.jit``."""
    from opifex.uncertainty.inference_backends.blackjax import _compute_ess

    samples = jax.random.normal(jax.random.PRNGKey(0), (32, 3))
    jit_ess = jax.jit(_compute_ess)
    out = jit_ess(samples)
    assert out.shape == (3,)
    assert jnp.all(out > 0)


def test_compute_ess_is_grad_compatible() -> None:
    """``_compute_ess`` must differentiate through samples for use in jit/grad pipelines."""
    from opifex.uncertainty.inference_backends.blackjax import _compute_ess

    samples = jax.random.normal(jax.random.PRNGKey(0), (32, 3))
    grad_fn = jax.grad(lambda s: jnp.sum(_compute_ess(s)))
    grads = grad_fn(samples)
    assert grads.shape == samples.shape
    assert jnp.all(jnp.isfinite(grads))


def test_blackjax_backend_spec_advertises_supported_and_unsupported_samplers() -> None:
    """``BLACKJAX_BACKEND_SPEC`` advertises the full sampler family list."""
    from opifex.uncertainty.inference_backends.base import InferenceBackendSpec
    from opifex.uncertainty.inference_backends.blackjax import BLACKJAX_BACKEND_SPEC

    assert isinstance(BLACKJAX_BACKEND_SPEC, InferenceBackendSpec)
    assert BLACKJAX_BACKEND_SPEC.name == "blackjax"
    # Implemented (delegated to Artifex).
    for impl in ("hmc", "nuts", "mala"):
        assert impl in BLACKJAX_BACKEND_SPEC.sampler_names
    # Audit-mandated unsupported families show up too, marked with the
    # ``unsupported:`` prefix so the router can surface them with an
    # actionable error.
    for unsupported in ("sgld", "sghmc", "smc", "advi", "pathfinder"):
        assert (
            unsupported in BLACKJAX_BACKEND_SPEC.sampler_names
            or f"unsupported:{unsupported}" in BLACKJAX_BACKEND_SPEC.sampler_names
        )


def test_blackjax_backend_delegates_to_artifex_nuts_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``method="nuts"`` MUST call ``artifex...nuts_sampling`` — not blackjax directly."""
    from artifex.generative_models.core.sampling import blackjax_samplers as artifex_bjx

    calls: list[str] = []
    real_nuts = artifex_bjx.nuts_sampling

    def spy(*args: Any, **kwargs: Any) -> jax.Array:
        calls.append("nuts_sampling")
        return real_nuts(*args, **kwargs)

    monkeypatch.setattr("opifex.uncertainty.inference_backends.blackjax.nuts_sampling", spy)
    backend = _make_backend("nuts")
    backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert calls == ["nuts_sampling"]


def test_blackjax_backend_delegates_to_artifex_hmc_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.generative_models.core.sampling import blackjax_samplers as artifex_bjx

    calls: list[str] = []
    real_hmc = artifex_bjx.hmc_sampling

    def spy(*args: Any, **kwargs: Any) -> jax.Array:
        calls.append("hmc_sampling")
        return real_hmc(*args, **kwargs)

    monkeypatch.setattr("opifex.uncertainty.inference_backends.blackjax.hmc_sampling", spy)
    backend = _make_backend("hmc")
    backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert calls == ["hmc_sampling"]


def test_blackjax_backend_delegates_to_artifex_mala_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from artifex.generative_models.core.sampling import blackjax_samplers as artifex_bjx

    calls: list[str] = []
    real_mala = artifex_bjx.mala_sampling

    def spy(*args: Any, **kwargs: Any) -> jax.Array:
        calls.append("mala_sampling")
        return real_mala(*args, **kwargs)

    monkeypatch.setattr("opifex.uncertainty.inference_backends.blackjax.mala_sampling", spy)
    backend = _make_backend("mala")
    backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert calls == ["mala_sampling"]


def test_blackjax_backend_routes_rng_through_extract_rng_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RNG ownership is enforced via Artifex's canonical helper."""
    calls: list[str] = []

    def spy(
        rng: jax.Array | nnx.Rngs | None,
        *,
        streams: tuple[str, ...] = ("sample", "default"),
        context: str = "sampling",
    ) -> jax.Array:
        calls.append(context)
        return artifex_extract_rng_key(rng, streams=streams, context=context)

    monkeypatch.setattr("opifex.uncertainty.inference_backends.blackjax.extract_rng_key", spy)
    backend = _make_backend("nuts")
    backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))
    assert calls, "extract_rng_key must be invoked at every sampling entry point"
    assert all("BlackJAX" in c for c in calls)


def test_blackjax_backend_predict_distribution_returns_predictive_distribution() -> None:
    backend = _make_backend("nuts")
    x = jnp.zeros((4, 3))
    out = backend.predict_distribution(x, rngs=nnx.Rngs(sample=0))
    assert isinstance(out, PredictiveDistribution)
    assert out.mean.shape == (4, 3)


def test_blackjax_backend_posterior_predictive_returns_predictive_distribution() -> None:
    backend = _make_backend("nuts")
    x = jnp.zeros((2, 3))
    out = backend.posterior_predictive(nnx.Rngs(sample=0), x)
    assert isinstance(out, PredictiveDistribution)
    assert out.mean.shape == (2, 3)


def test_blackjax_backend_rejects_unknown_sampler_with_unsupported_backend_error() -> None:
    """An unknown sampler name MUST raise UnsupportedBackendError naming the backend."""
    with pytest.raises(UnsupportedBackendError, match=r"blackjax"):
        BlackJAXBackend(
            target_log_prob=_gaussian_log_density,
            init_state=jnp.zeros(3),
            n_samples=5,
            n_burnin=1,
            method="sgld",  # Artifex does not yet ship SGLD
        )


def test_blackjax_backend_rejects_malformed_arguments_with_value_error() -> None:
    with pytest.raises(ValueError, match=r"n_samples"):
        BlackJAXBackend(
            target_log_prob=_gaussian_log_density,
            init_state=jnp.zeros(3),
            n_samples=0,
            n_burnin=0,
            method="nuts",
        )


def test_blackjax_backend_no_direct_blackjax_import() -> None:
    """Static scan: zero direct ``blackjax`` imports — must delegate via Artifex."""
    from pathlib import Path

    source = Path("src/opifex/uncertainty/inference_backends/blackjax.py").read_text()
    forbidden = [
        line
        for line in source.splitlines()
        if line.startswith(("import blackjax", "from blackjax"))
    ]
    assert not forbidden, (
        "BlackJAXBackend must not import blackjax directly. The only "
        "legitimate blackjax import lives in artifex's wrapper. Offending: "
        f"{forbidden}"
    )
