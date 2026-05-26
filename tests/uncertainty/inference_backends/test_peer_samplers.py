"""Tests for the Pathfinder / SVGD / ADVI peer-sampler backends.

Each backend lives under :mod:`opifex.uncertainty.inference_backends` as
a Pattern-A frozen dataclass implementing
:class:`InferenceBackendProtocol`. The concrete sampling algorithms
themselves are deferred to a follow-up slice; the metadata + protocol
implementations land in this slice so the registry router can resolve
them by name.

Canonical references:
* Zhang, L. et al. 2022 — *Pathfinder: Parallel quasi-Newton variational
  inference*, JMLR 23(306).
* Liu, Q. & Wang, D. 2016 — *Stein Variational Gradient Descent*,
  NeurIPS 29.
* Kucukelbir, A. et al. 2017 — *Automatic Differentiation Variational
  Inference*, JMLR 18(14).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.inference_backends import (
    ADVIBackend,
    InferenceBackendProtocol,
    PathfinderBackend,
    SVGDBackend,
)


_PEER_BACKENDS: tuple[type, ...] = (
    PathfinderBackend,
    SVGDBackend,
    ADVIBackend,
)


_DEFERRED_PEER_BACKENDS: tuple[type, ...] = (
    PathfinderBackend,
    ADVIBackend,
)


_CONCRETIZED_PEER_BACKENDS: tuple[type, ...] = (
    # Task 6.3.9a: SVGD algorithm now wired through `fit`.
    SVGDBackend,
)


@pytest.mark.parametrize("backend_cls", _PEER_BACKENDS)
def test_peer_backend_is_frozen_dataclass_with_capability_metadata(
    backend_cls: type,
) -> None:
    """Every peer backend is a frozen dataclass with capability metadata."""
    import dataclasses as dc

    assert dc.is_dataclass(backend_cls)
    backend: Any = backend_cls()
    assert isinstance(backend.name, str)
    assert isinstance(backend.source_package, str)
    assert isinstance(backend.method_names, tuple)
    with pytest.raises(dc.FrozenInstanceError):
        backend.name = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("backend_cls", _PEER_BACKENDS)
def test_peer_backend_implements_inference_protocol(backend_cls: type) -> None:
    """Each peer backend conforms to ``InferenceBackendProtocol`` structurally."""
    backend = backend_cls()
    assert isinstance(backend, InferenceBackendProtocol)


@pytest.mark.parametrize("backend_cls", _DEFERRED_PEER_BACKENDS)
def test_deferred_peer_backend_fit_raises_until_algorithm_lands(backend_cls: type) -> None:
    """Deferred peer backends still raise ``NotImplementedError`` from ``fit``.

    Pathfinder + ADVI ports land in follow-up slices.
    """
    backend = backend_cls()

    def target_log_prob(_: jax.Array) -> jax.Array:
        return jnp.zeros(())

    with pytest.raises(NotImplementedError, match=backend.name):
        backend.fit(target_log_prob=target_log_prob, rngs=nnx.Rngs(sampler=0))


def test_svgd_backend_fit_returns_particle_cloud_concentrating_around_target_mean() -> None:
    """``SVGDBackend.fit`` runs the algorithm and returns concentrated particles.

    On a standard-normal target the final particles should have a mean
    near zero (the target mean).
    """
    from opifex.uncertainty.inference_backends.base import BackendResult

    backend = SVGDBackend(
        init_state=jnp.zeros(2),
        num_particles=20,
        num_iterations=200,
        learning_rate=0.3,
        init_scale=2.0,
    )

    def target_log_prob(x: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(x**2)

    result = backend.fit(target_log_prob=target_log_prob, rngs=nnx.Rngs(sampler=42))
    assert isinstance(result, BackendResult)
    particles = result.sampler_state
    assert particles.shape == (20, 2)
    empirical_mean = jnp.mean(particles, axis=0)
    assert jnp.allclose(empirical_mean, jnp.zeros(2), atol=0.5)


def test_svgd_backend_predict_distribution_remains_deferred() -> None:
    """Until a model-aware adapter exists, prediction hooks still raise."""
    backend = SVGDBackend()
    with pytest.raises(NotImplementedError, match="predict_distribution"):
        backend.predict_distribution(jnp.zeros(1), rngs=nnx.Rngs(sampler=0))
    with pytest.raises(NotImplementedError, match="posterior_predictive"):
        backend.posterior_predictive(nnx.Rngs(sampler=0), jnp.zeros(1))


def test_pathfinder_backend_advertises_quasi_newton_family() -> None:
    """``PathfinderBackend`` notes its variational quasi-Newton lineage."""
    backend = PathfinderBackend()
    assert "pathfinder" in backend.method_names
    assert "quasi" in backend.notes.lower() or "Pathfinder" in backend.notes


def test_svgd_backend_advertises_stein_kernelised_gradient() -> None:
    """``SVGDBackend`` notes its kernelised-gradient Stein lineage."""
    backend = SVGDBackend()
    assert "svgd" in backend.method_names
    assert "stein" in backend.notes.lower()


def test_advi_backend_advertises_meanfield_or_fullrank_choice() -> None:
    """``ADVIBackend`` notes its mean-field / full-rank choice."""
    backend = ADVIBackend()
    assert "advi" in backend.method_names
    assert "mean-field" in backend.notes.lower() or "meanfield" in backend.notes.lower()


def test_blackjax_no_longer_lists_advi_and_pathfinder_as_unsupported() -> None:
    """``advi`` and ``pathfinder`` are now owned by peer backends, not BlackJAX."""
    from opifex.uncertainty.inference_backends.blackjax import _UNSUPPORTED_METHODS

    assert "advi" not in _UNSUPPORTED_METHODS
    assert "pathfinder" not in _UNSUPPORTED_METHODS
