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
from opifex.uncertainty.types import PredictiveDistribution


def _unit_gaussian_log_prob(theta: jax.Array) -> jax.Array:
    """Standard-normal log density over the parameter vector ``theta``."""
    return -0.5 * jnp.sum(theta**2)


def _linear_predict(params: jax.Array, x: jax.Array) -> jax.Array:
    """Known linear forward model ``x @ params`` mapping params → predictions."""
    return x @ params


_PEER_BACKENDS: tuple[type, ...] = (
    PathfinderBackend,
    SVGDBackend,
    ADVIBackend,
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


def test_svgd_backend_predict_distribution_raises_without_stored_target() -> None:
    """A backend built without a stored ``target_log_prob`` cannot re-fit."""
    backend = SVGDBackend()
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.predict_distribution(jnp.zeros(1), rngs=nnx.Rngs(sampler=0))
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.posterior_predictive(nnx.Rngs(sampler=0), jnp.zeros(1))


def test_pathfinder_backend_fit_returns_samples_concentrating_around_target_mode() -> None:
    """``PathfinderBackend.fit`` runs L-BFGS + samples; draws concentrate near the mode.

    On a standard-normal target, Pathfinder draws should have mean
    near zero.
    """
    from opifex.uncertainty.inference_backends.base import BackendResult

    backend = PathfinderBackend(
        init_state=jnp.array([2.0, -1.5]),
        num_samples=512,
        num_elbo_samples=32,
        maxiter=30,
        maxcor=6,
    )

    def target_log_prob(x: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(x**2)

    result = backend.fit(target_log_prob=target_log_prob, rngs=nnx.Rngs(sampler=7))
    assert isinstance(result, BackendResult)
    samples = result.sampler_state
    assert samples.shape == (512, 2)
    empirical_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_mean, jnp.zeros(2), atol=0.2)


def test_pathfinder_backend_predict_distribution_raises_without_stored_target() -> None:
    """Pathfinder's prediction hooks need a stored ``target_log_prob`` to re-fit."""
    backend = PathfinderBackend()
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.predict_distribution(jnp.zeros(1), rngs=nnx.Rngs(sampler=0))
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.posterior_predictive(nnx.Rngs(sampler=0), jnp.zeros(1))


def test_pathfinder_backend_predict_distribution_model_aware_and_lightweight() -> None:
    """A stored-target Pathfinder backend yields real + lightweight predictives."""
    d = 2
    backend = PathfinderBackend(
        init_state=jnp.zeros(d),
        target_log_prob=_unit_gaussian_log_prob,
        num_samples=128,
        num_elbo_samples=32,
    )
    x = jnp.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])

    model_aware = backend.predict_distribution(
        x, rngs=nnx.Rngs(sampler=3), predict_fn=_linear_predict
    )
    assert isinstance(model_aware, PredictiveDistribution)
    assert model_aware.mean.shape == (3,)
    assert model_aware.samples is not None
    assert model_aware.samples.shape == (128, 3)

    lightweight = backend.predict_distribution(x, rngs=nnx.Rngs(sampler=3))
    assert isinstance(lightweight, PredictiveDistribution)
    assert lightweight.mean.shape == x.shape


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


def test_advi_backend_fit_returns_meanfield_samples_concentrating_around_target_mean() -> None:
    """``ADVIBackend.fit`` runs ELBO optimisation and returns concentrated draws.

    On a standard-normal target the mean-field Gaussian should converge to
    zero mean and unit scale, so the samples concentrate near zero.
    """
    from opifex.uncertainty.inference_backends.base import BackendResult

    backend = ADVIBackend(
        init_state=jnp.array([1.0, -1.0]),
        num_samples=512,
        num_iterations=400,
        num_mc_samples=8,
        learning_rate=0.05,
    )

    def target_log_prob(x: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(x**2)

    result = backend.fit(target_log_prob=target_log_prob, rngs=nnx.Rngs(sampler=11))
    assert isinstance(result, BackendResult)
    samples = result.sampler_state
    assert samples.shape == (512, 2)
    empirical_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_mean, jnp.zeros(2), atol=0.3)


def test_advi_backend_predict_distribution_raises_without_stored_target() -> None:
    """ADVI's prediction hooks need a stored ``target_log_prob`` to re-fit."""
    backend = ADVIBackend()
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.predict_distribution(jnp.zeros(1), rngs=nnx.Rngs(sampler=0))
    with pytest.raises(ValueError, match="target_log_prob"):
        backend.posterior_predictive(nnx.Rngs(sampler=0), jnp.zeros(1))


def test_advi_backend_predict_distribution_model_aware_depends_on_model() -> None:
    """ADVI model-aware predictive marginalises the forward model over draws.

    The previously-raising hook now returns a real
    :class:`PredictiveDistribution` whose moments depend on the supplied
    ``predict_fn`` (distinct from the lightweight parameter-moment form).
    """
    d = 2
    backend = ADVIBackend(
        init_state=jnp.zeros(d),
        target_log_prob=_unit_gaussian_log_prob,
        num_samples=256,
        num_iterations=200,
        num_mc_samples=8,
        learning_rate=0.05,
    )
    x = jnp.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])

    model_aware = backend.predict_distribution(
        x, rngs=nnx.Rngs(sampler=5), predict_fn=_linear_predict
    )
    assert isinstance(model_aware, PredictiveDistribution)
    assert model_aware.mean.shape == (3,)
    assert model_aware.samples is not None
    assert model_aware.samples.shape == (256, 3)
    # The model-aware predictive's per-output mean reflects ``x @ E[params]``,
    # not the raw parameter moments — it genuinely depends on the model.
    assert model_aware.mean.shape != x.shape


def test_advi_backend_predict_distribution_lightweight_form() -> None:
    """Without ``predict_fn`` the ADVI predictive falls back to parameter moments."""
    d = 3
    backend = ADVIBackend(
        init_state=jnp.zeros(d),
        target_log_prob=_unit_gaussian_log_prob,
        num_samples=256,
        num_iterations=200,
    )
    x = jnp.zeros((4, d))

    lightweight = backend.predict_distribution(x, rngs=nnx.Rngs(sampler=5))
    assert isinstance(lightweight, PredictiveDistribution)
    assert lightweight.mean.shape == x.shape
    assert lightweight.variance is not None
    assert lightweight.variance.shape == x.shape


def test_advi_backend_posterior_predictive_returns_predictive_distribution() -> None:
    """The ADVI ``posterior_predictive`` hook now returns a predictive too."""
    d = 2
    backend = ADVIBackend(
        init_state=jnp.zeros(d),
        target_log_prob=_unit_gaussian_log_prob,
        num_samples=128,
        num_iterations=150,
    )
    x = jnp.array([[1.0, 2.0], [0.5, 0.5]])
    out = backend.posterior_predictive(nnx.Rngs(sampler=9), x, predict_fn=_linear_predict)
    assert isinstance(out, PredictiveDistribution)
    assert out.mean.shape == (2,)
    assert dict(out.metadata).get("method") == "posterior_predictive"


def test_blackjax_no_longer_lists_advi_and_pathfinder_as_unsupported() -> None:
    """``advi`` and ``pathfinder`` are now owned by peer backends, not BlackJAX."""
    from opifex.uncertainty.inference_backends.blackjax import _UNSUPPORTED_METHODS

    assert "advi" not in _UNSUPPORTED_METHODS
    assert "pathfinder" not in _UNSUPPORTED_METHODS
