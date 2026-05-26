"""Pin migrated behavior: ``BlackJAXBackend`` replaces the deleted ``BlackJAXIntegration``.

Pre-migration, ``opifex.neural.bayesian.blackjax_integration:BlackJAXIntegration``
exposed a high-level MCMC wrapper coupled to model + data + a fake
perturbation-based log-likelihood. The migration deleted that class and
introduced ``opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend``
— a thin adapter around Artifex's BlackJAX wrappers that accepts a
log-density callable directly.

These tests pin three properties of the migrated behavior:

1. The legacy symbol is gone from every searched root.
2. The new backend integrates with the Phase 1 contract (``BackendResult``,
   ``BackendDiagnostics``, ``InferenceBackendSpec``) end-to-end on a
   non-trivial target log-density.
3. The migrated stack composes with Opifex's predictive-distribution
   container via ``BlackJAXBackend.predict_distribution``.
"""

from __future__ import annotations

import subprocess

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty import (
    BackendDiagnostics,
    BackendResult,
    BlackJAXBackend,
    PredictiveDistribution,
)
from opifex.uncertainty.inference_backends.blackjax import BLACKJAX_BACKEND_SPEC


def _banana_log_density(x: jax.Array) -> jax.Array:
    """Rosenbrock-style banana — non-trivial 2D target for MCMC."""
    a, b = 1.0, 5.0
    return -((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)


def test_legacy_blackjax_integration_symbol_no_longer_exists() -> None:
    """Migration deleted the legacy class; zero remaining references in src/.

    The search is restricted to ``src/`` because this very test file
    necessarily mentions the legacy name in its own assertion / docstring.
    """
    result = subprocess.run(
        ["rg", "-l", r"\bBlackJAXIntegration\b", "src/"],
        capture_output=True,
        text=True,
        check=False,
    )
    matches = [line for line in result.stdout.splitlines() if line.strip()]
    assert not matches, (
        f"Legacy class name still referenced under src/ after migration. Offending files: {matches}"
    )


def test_blackjax_backend_runs_end_to_end_on_non_trivial_target() -> None:
    backend = BlackJAXBackend(
        target_log_prob=_banana_log_density,
        init_state=jnp.zeros(2),
        n_samples=64,
        n_burnin=32,
        method="nuts",
        step_size=0.1,
    )
    result = backend.fit(_banana_log_density, rngs=nnx.Rngs(sample=0))
    assert isinstance(result, BackendResult)
    samples = result.sampler_state
    assert samples.shape == (64, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_blackjax_backend_populates_post_hoc_ess_diagnostic() -> None:
    """Post-migration diagnostics surface ESS even though Artifex drops BlackJAX's info dict."""
    backend = BlackJAXBackend(
        target_log_prob=_banana_log_density,
        init_state=jnp.zeros(2),
        n_samples=64,
        n_burnin=32,
        method="nuts",
        step_size=0.1,
    )
    result = backend.fit(_banana_log_density, rngs=nnx.Rngs(sample=0))
    assert isinstance(result.diagnostics, BackendDiagnostics)
    assert result.diagnostics.ess is not None
    assert result.diagnostics.ess.shape == (2,)
    assert jnp.all(result.diagnostics.ess > 0)


def test_blackjax_backend_predict_distribution_carries_backend_metadata_for_banana_density() -> (
    None
):
    """Integration sibling of the unit test in test_blackjax_backend.py — verifies
    the banana-density posterior also emits ``backend == "blackjax"`` metadata.
    """
    backend = BlackJAXBackend(
        target_log_prob=_banana_log_density,
        init_state=jnp.zeros(2),
        n_samples=32,
        n_burnin=16,
        method="nuts",
    )
    x = jnp.zeros((4, 2))
    predictive = backend.predict_distribution(x, rngs=nnx.Rngs(sample=0))
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == (4, 2)
    assert predictive.metadata_dict()["backend"] == "blackjax"


def test_blackjax_backend_spec_metadata_advertises_full_sampler_family() -> None:
    """``InferenceBackendSpec`` surfaces both implemented and explicitly-unsupported samplers."""
    assert BLACKJAX_BACKEND_SPEC.name == "blackjax"
    assert BLACKJAX_BACKEND_SPEC.source_package == "artifex"
    for impl in ("hmc", "nuts", "mala"):
        assert impl in BLACKJAX_BACKEND_SPEC.sampler_names
    for unsupp in ("sgld", "sghmc", "smc"):
        assert f"unsupported:{unsupp}" in BLACKJAX_BACKEND_SPEC.sampler_names
    for moved in ("advi", "pathfinder"):
        assert f"unsupported:{moved}" not in BLACKJAX_BACKEND_SPEC.sampler_names
