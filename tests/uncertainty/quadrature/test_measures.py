"""Tests for integration measures used in Bayesian quadrature.

A measure ``π`` defines the integration domain and the density against
which a function is integrated: ``∫ f(x) π(dx)``. The two canonical
measures are the (possibly diagonal-covariance) Gaussian and the
Lebesgue measure over a hyperrectangular domain.

Canonical reference (line-by-line port):
* ``../emukit/emukit/quadrature/measures/gaussian_measure.py`` —
  :class:`GaussianMeasure` field semantics.
* ``../emukit/emukit/quadrature/measures/lebesgue_measure.py`` —
  :class:`LebesgueMeasure` field semantics.

References
----------
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*,
  Statistical Science 34(1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.quadrature import GaussianMeasure, LebesgueMeasure


def test_gaussian_measure_stores_mean_and_variance_as_arrays() -> None:
    """``GaussianMeasure`` is a frozen Pattern-A dataclass."""
    import dataclasses as dc

    measure = GaussianMeasure(mean=jnp.asarray([0.0, 1.0]), variance=jnp.asarray([1.0, 2.0]))
    assert dc.is_dataclass(measure)
    assert measure.mean.shape == (2,)
    assert measure.variance.shape == (2,)
    assert measure.input_dim == 2
    with pytest.raises(dc.FrozenInstanceError):
        measure.mean = jnp.zeros(2)  # type: ignore[misc]


def test_gaussian_measure_rejects_non_positive_variance() -> None:
    """Diagonal variance must be entry-wise positive."""
    with pytest.raises(ValueError, match="variance must be positive"):
        GaussianMeasure(mean=jnp.zeros(2), variance=jnp.asarray([1.0, -0.5]))


def test_gaussian_measure_rejects_shape_mismatch() -> None:
    """``mean`` and ``variance`` must share the leading dimension."""
    with pytest.raises(ValueError, match="same shape"):
        GaussianMeasure(mean=jnp.zeros(2), variance=jnp.ones(3))


def test_gaussian_measure_samples_match_target_moments() -> None:
    """``sample`` yields samples with the requested mean / variance."""
    measure = GaussianMeasure(mean=jnp.asarray([1.0, -2.0]), variance=jnp.asarray([4.0, 0.25]))
    samples = measure.sample(num_samples=20000, key=jax.random.PRNGKey(0))
    assert samples.shape == (20000, 2)
    assert jnp.allclose(jnp.mean(samples, axis=0), measure.mean, atol=0.1)
    assert jnp.allclose(jnp.var(samples, axis=0), measure.variance, atol=0.2)


def test_lebesgue_measure_stores_lower_and_upper_bounds() -> None:
    """``LebesgueMeasure`` is a frozen Pattern-A dataclass over a box."""
    import dataclasses as dc

    measure = LebesgueMeasure(lower=jnp.asarray([0.0, -1.0]), upper=jnp.asarray([1.0, 1.0]))
    assert dc.is_dataclass(measure)
    assert measure.input_dim == 2
    assert jnp.allclose(measure.volume, jnp.asarray(2.0), atol=1e-6)


def test_lebesgue_measure_rejects_inverted_bounds() -> None:
    """Each ``lower[i] < upper[i]`` constraint is checked."""
    with pytest.raises(ValueError, match="lower bound must be strictly less"):
        LebesgueMeasure(lower=jnp.asarray([0.0, 1.0]), upper=jnp.asarray([1.0, 0.5]))


def test_lebesgue_measure_samples_lie_inside_bounds() -> None:
    """Uniform samples fall within ``[lower, upper]`` element-wise."""
    measure = LebesgueMeasure(lower=jnp.asarray([0.0, -1.0]), upper=jnp.asarray([2.0, 3.0]))
    samples = measure.sample(num_samples=5000, key=jax.random.PRNGKey(1))
    assert samples.shape == (5000, 2)
    assert jnp.all(samples >= measure.lower)
    assert jnp.all(samples <= measure.upper)


def test_measures_are_jit_compatible_via_sample() -> None:
    """Both measures' ``sample`` runs inside ``jax.jit``."""
    gaussian = GaussianMeasure(mean=jnp.zeros(2), variance=jnp.ones(2))
    lebesgue = LebesgueMeasure(lower=jnp.zeros(2), upper=jnp.ones(2))

    @jax.jit
    def draw_gaussian(key: jax.Array) -> jax.Array:
        return gaussian.sample(num_samples=128, key=key)

    @jax.jit
    def draw_lebesgue(key: jax.Array) -> jax.Array:
        return lebesgue.sample(num_samples=128, key=key)

    assert jnp.all(jnp.isfinite(draw_gaussian(jax.random.PRNGKey(2))))
    assert jnp.all(jnp.isfinite(draw_lebesgue(jax.random.PRNGKey(3))))
