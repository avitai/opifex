r"""Shared predictive-factory + Markov support helpers (DRY consolidation 12.3).

Locks the behaviour of the extracted shared modules introduced when the
quadruplicated Markov predict path, the duplicated ``_replace_metadata*``
helpers, and the duplicated ``_latent_variance`` guard were consolidated:

* :mod:`opifex.uncertainty._predictive` — ``gaussian_process_predictive`` /
  ``replace_predictive_metadata``.
* :mod:`opifex.uncertainty.markov._likelihood_support` — ``latent_variance`` /
  ``interpolate_smoothed_state``.

The factory must reproduce, byte-for-byte, the objects the hand-written sites
built; the interpolation must be jit/grad/vmap-compatible (JAX/NNX transform
compatibility is paramount for the UQ inference code).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty._predictive import (
    gaussian_process_predictive,
    replace_predictive_metadata,
    sample_based_predictive,
)
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.markov._likelihood_support import (
    interpolate_smoothed_state,
    latent_variance,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# gaussian_process_predictive
# -----------------------------------------------------------------------------


def test_gaussian_process_predictive_matches_hand_written_construction() -> None:
    """Factory reproduces the exact fields a hand-written constructor sets."""
    mean = jnp.asarray([1.0, 2.0, 3.0])
    variance = jnp.asarray([0.1, 0.2, 0.3])
    meta = (("method", "gaussian_process"), ("estimator", "demo"))

    built = gaussian_process_predictive(
        mean, variance, epistemic=variance, total_uncertainty=variance, metadata=meta
    )
    reference = PredictiveDistribution(
        mean=mean,
        variance=variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=meta,
    )

    assert built.variance is not None
    assert built.epistemic is not None
    assert built.total_uncertainty is not None
    assert jnp.array_equal(built.mean, reference.mean)
    assert jnp.array_equal(built.variance, variance)
    assert jnp.array_equal(built.epistemic, variance)
    assert jnp.array_equal(built.total_uncertainty, variance)
    assert built.metadata == reference.metadata
    assert built.samples is None
    assert built.covariance is None


def test_gaussian_process_predictive_optional_components_default_none() -> None:
    """Omitting epistemic/total leaves them ``None`` (latent-only predictive)."""
    mean = jnp.asarray([0.0, 1.0])
    variance = jnp.asarray([1.0, 1.0])
    built = gaussian_process_predictive(mean, variance)
    assert built.epistemic is None
    assert built.total_uncertainty is None
    assert built.metadata == ()


# -----------------------------------------------------------------------------
# sample_based_predictive
# -----------------------------------------------------------------------------


def test_sample_based_predictive_reduces_samples_to_empirical_moments() -> None:
    """``mean`` / ``variance`` match the hand-written ``jnp.mean/var(samples, 0)``."""
    samples = jnp.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    built = sample_based_predictive(samples, metadata=(("method", "npe"),))
    stored_variance = built.variance
    stored_samples = built.samples
    assert stored_variance is not None
    assert stored_samples is not None
    assert jnp.array_equal(built.mean, jnp.mean(samples, axis=0))
    assert jnp.array_equal(stored_variance, jnp.var(samples, axis=0))
    assert jnp.array_equal(stored_samples, samples)
    assert built.metadata == (("method", "npe"),)


def test_sample_based_predictive_defaults_empty_metadata() -> None:
    built = sample_based_predictive(jnp.zeros((4, 2)))
    assert built.metadata == ()


def test_sample_based_predictive_is_jit_compatible() -> None:
    """The sampler factory assembles a predictive inside a jitted function."""

    @jax.jit
    def build(samples: jax.Array) -> jax.Array:
        predictive = sample_based_predictive(samples)
        recovered_variance = predictive.variance
        assert recovered_variance is not None
        return predictive.mean.sum() + recovered_variance.sum()

    out = build(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert jnp.isfinite(out)


# -----------------------------------------------------------------------------
# replace_predictive_metadata
# -----------------------------------------------------------------------------


def test_replace_predictive_metadata_preserves_arrays_and_stamps_metadata() -> None:
    """Metadata refresh keeps every array field and re-stamps provenance."""
    base = PredictiveDistribution(
        mean=jnp.asarray([1.0, 2.0]),
        variance=jnp.asarray([0.5, 0.6]),
        epistemic=jnp.asarray([0.5, 0.6]),
        total_uncertainty=jnp.asarray([0.5, 0.6]),
        metadata=(("stale", "value"),),
    )
    refreshed = replace_predictive_metadata(
        base,
        estimator="bernoulli_markov_vi_gp",
        likelihood="bernoulli",
        link="logit",
        paper="Khan & Lin 2017",
    )
    # Arrays untouched.
    assert refreshed.variance is not None
    assert refreshed.epistemic is not None
    assert base.variance is not None
    assert base.epistemic is not None
    assert jnp.array_equal(refreshed.mean, base.mean)
    assert jnp.array_equal(refreshed.variance, base.variance)
    assert jnp.array_equal(refreshed.epistemic, base.epistemic)
    # Metadata replaced with the canonical ordering: method, source_package,
    # estimator, paper, likelihood, link.
    expected = compose_method_metadata(
        method=DefaultStrategy.GAUSSIAN_PROCESS.value,
        source_package="opifex.uncertainty.markov",
        extra=(
            ("estimator", "bernoulli_markov_vi_gp"),
            ("paper", "Khan & Lin 2017"),
            ("likelihood", "bernoulli"),
            ("link", "logit"),
        ),
    )
    assert refreshed.metadata == expected


def test_replace_predictive_metadata_omits_paper_when_none() -> None:
    """A ``None`` paper drops the paper entry but keeps likelihood/link."""
    base = PredictiveDistribution(mean=jnp.asarray([0.0]), variance=jnp.asarray([1.0]))
    refreshed = replace_predictive_metadata(
        base, estimator="e", likelihood="lik", link="lnk", paper=None
    )
    keys = [k for k, _ in refreshed.metadata]
    assert "paper" not in keys
    assert ("likelihood", "lik") in refreshed.metadata
    assert ("link", "lnk") in refreshed.metadata


# -----------------------------------------------------------------------------
# latent_variance
# -----------------------------------------------------------------------------


def test_latent_variance_returns_variance_when_present() -> None:
    """The helper unwraps the variance array unchanged."""
    variance = jnp.asarray([0.1, 0.2])
    predictive = PredictiveDistribution(mean=jnp.asarray([0.0, 0.0]), variance=variance)
    assert jnp.array_equal(latent_variance(predictive), variance)


def test_latent_variance_raises_runtime_error_when_missing() -> None:
    """A ``None`` variance raises (survives ``python -O`` unlike ``assert``)."""
    predictive = PredictiveDistribution(mean=jnp.asarray([0.0]))
    with pytest.raises(RuntimeError, match="no variance"):
        latent_variance(predictive)


# -----------------------------------------------------------------------------
# interpolate_smoothed_state — correctness + transform compatibility
# -----------------------------------------------------------------------------


def _smoothed_inputs(seed: int = 0, n: int = 8) -> dict:
    """Build a plausible smoothed-state trajectory for a Matern-3/2 kernel."""
    key = jax.random.PRNGKey(seed)
    kernel = matern32_kernel(variance=1.0, lengthscale=0.6)
    times_train = jnp.sort(jax.random.uniform(key, (n,), minval=0.0, maxval=5.0))
    state_dim = kernel.state_dim
    means = 0.1 * jax.random.normal(jax.random.fold_in(key, 1), (n, state_dim))
    # PSD covariances via A A^T + small jitter.
    a = jax.random.normal(jax.random.fold_in(key, 2), (n, state_dim, state_dim))
    covs = jnp.einsum("nij,nkj->nik", a, a) + 1e-2 * jnp.eye(state_dim)
    return {
        "state_space_kernel": kernel,
        "times_train": times_train,
        "smoothed_state_means": means,
        "smoothed_state_covs": covs,
    }


def test_interpolate_smoothed_state_returns_expected_shapes_and_floor() -> None:
    """Output shapes match the test grid and variances respect the floor."""
    inputs = _smoothed_inputs()
    times_test = jnp.linspace(-1.0, 6.0, 11)  # spans before-first + interior
    means, variances = interpolate_smoothed_state(times_test=times_test, **inputs)
    assert means.shape == (11,)
    assert variances.shape == (11,)
    assert bool(jnp.all(variances >= 1e-6))
    assert bool(jnp.all(jnp.isfinite(means)))


def test_interpolate_smoothed_state_is_jit_grad_vmap_compatible() -> None:
    """The interpolation compiles under jit, differentiates, and vmaps."""
    inputs = _smoothed_inputs()
    times_test = jnp.linspace(0.0, 5.0, 7)

    @jax.jit
    def scalarised(test_times: jax.Array) -> jax.Array:
        means, variances = interpolate_smoothed_state(times_test=test_times, **inputs)
        return means.sum() + variances.sum()

    value = scalarised(times_test)
    assert jnp.isfinite(value)

    grad = jax.grad(scalarised)(times_test)
    assert grad.shape == times_test.shape
    assert bool(jnp.all(jnp.isfinite(grad)))

    batched = jax.vmap(scalarised)(jnp.stack([times_test, times_test + 0.1]))
    assert batched.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(batched)))


def test_gaussian_process_predictive_is_jit_compatible() -> None:
    """The factory assembles a predictive inside a jitted function."""

    @jax.jit
    def build(mean: jax.Array, variance: jax.Array) -> jax.Array:
        predictive = gaussian_process_predictive(
            mean, variance, epistemic=variance, total_uncertainty=variance
        )
        recovered_variance = predictive.variance
        assert recovered_variance is not None
        return predictive.mean.sum() + recovered_variance.sum()

    out = build(jnp.asarray([1.0, 2.0]), jnp.asarray([0.1, 0.2]))
    assert jnp.isfinite(out)
