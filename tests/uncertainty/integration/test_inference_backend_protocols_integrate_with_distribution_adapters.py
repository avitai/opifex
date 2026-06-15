"""Pin inference-backend protocol and distribution-adapter integration.

The Phase 1 inference-backend / distribution-adapter protocols
(:mod:`opifex.uncertainty.inference_backends.base`,
:mod:`opifex.uncertainty.adapters.base`) act as the contract surface that
Phase 2 (``BlackJAXBackend``), Phase 8 (PAC-Bayes / SBI / active-learning),
and any future backend wires plug into. These integration tests pin the
cross-component wiring:

1. ``BlackJAXBackend`` satisfies ``InferenceBackendProtocol`` (already
   pinned in the unit test); here we exercise it through the protocol
   surface to confirm a generic caller works.
2. ``BLACKJAX_BACKEND_SPEC`` (an ``InferenceBackendSpec``) is registerable
   in the registry pattern the optional-backend router will rely on.
3. ``from_distribution`` accepts both an Artifex ``Distribution`` and a
   Distrax-like object and returns a :class:`PredictiveDistribution` that
   composes with ``UQLossComponents``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from artifex.generative_models.core.distributions.continuous import Normal
from flax import nnx

from opifex.uncertainty import (
    BackendDiagnostics,
    BackendResult,
    BlackJAXBackend,
    from_distribution,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    PredictiveDistribution,
)
from opifex.uncertainty.inference_backends.blackjax import BLACKJAX_BACKEND_SPEC


def _gaussian_log_density(x: jax.Array) -> jax.Array:
    return -0.5 * jnp.sum(x * x)


def test_blackjax_backend_exercised_through_inference_backend_protocol() -> None:
    """A generic caller using only the protocol surface drives the backend end-to-end."""

    def run(backend: InferenceBackendProtocol) -> BackendResult:
        return backend.fit(_gaussian_log_density, rngs=nnx.Rngs(sample=0))

    backend = BlackJAXBackend(
        target_log_prob=_gaussian_log_density,
        init_state=jnp.zeros(2),
        n_samples=16,
        n_burnin=4,
        method="nuts",
    )
    result = run(backend)
    assert isinstance(result, BackendResult)
    assert isinstance(result.diagnostics, BackendDiagnostics)
    assert result.diagnostics.ess is not None


def test_blackjax_backend_spec_is_registrable_in_inference_backend_spec_shape() -> None:
    """``BLACKJAX_BACKEND_SPEC`` is a frozen pattern-(A) container suitable for static dispatch."""
    assert isinstance(BLACKJAX_BACKEND_SPEC, InferenceBackendSpec)
    # Static / hashable: a router can use it as a dict key.
    spec_table = {BLACKJAX_BACKEND_SPEC: "blackjax"}
    assert spec_table[BLACKJAX_BACKEND_SPEC] == "blackjax"


def test_from_distribution_artifex_path_composes_with_predictive_distribution() -> None:
    """Wrap an Artifex ``Normal`` and confirm the round-trip into ``PredictiveDistribution``."""
    rngs = nnx.Rngs(0)
    dist = Normal(loc=jnp.zeros(3), scale=jnp.ones(3), rngs=rngs)
    predictive = from_distribution(dist)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == (3,)
    assert predictive.variance is not None
    assert predictive.variance.shape == (3,)


def test_from_distribution_distrax_like_path_composes_with_predictive_distribution() -> None:
    """A minimal Distrax-like object passes through the fallback adapter."""

    class _FakeDistrax:
        def sample(
            self, sample_shape: tuple[int, ...] = (), *, rngs: nnx.Rngs | None = None
        ) -> jax.Array:
            del rngs
            return jnp.zeros(sample_shape or (1,))

        def log_prob(self, x: jax.Array) -> jax.Array:
            return jnp.zeros_like(x)

        def mean(self) -> jax.Array:
            return jnp.array([1.0, 2.0])

        def variance(self) -> jax.Array:
            return jnp.array([0.25, 0.25])

    predictive = from_distribution(_FakeDistrax())
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == (2,)
