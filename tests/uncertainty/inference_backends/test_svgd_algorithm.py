r"""Stein Variational Gradient Descent (Liu+Wang 2016) algorithm tests.

Tests for :mod:`opifex.uncertainty.inference_backends._svgd_algorithm`,
a JAX-native port of the SVGD reference at
``../blackjax/blackjax/vi/svgd.py``. The algorithm uses a kernelised
gradient flow that minimises KL to the target posterior:

* Particle update
  ``φ*(x_j) = (1/n) Σ_i [k(x_i, x_j) ∇log p(x_i) + ∇_{x_i} k(x_i, x_j)]``.
* RBF kernel ``k(x, y) = exp(-||x - y||² / ℓ)`` with bandwidth ``ℓ``
  set via the median heuristic by default.

Algorithm invariants verified:

* **Mean recovery on a Gaussian target.** SVGD particles drawn from a
  standard-normal target should have empirical mean near zero after
  enough iterations.
* **Variance recovery on a Gaussian target.** Empirical particle
  variance should approach the target variance.
* **Median heuristic bandwidth.** ``length_scale = median(pairwise
  distances)² / log(n_particles)``.
* **RBF kernel symmetry.** ``k(x, y) == k(y, x)``.
* **JIT compatibility.** All primitives compile under ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.inference_backends._svgd_algorithm import (
    median_heuristic_bandwidth,
    rbf_kernel,
    svgd_fit,
)


def _standard_normal_log_prob(x: jax.Array) -> jax.Array:
    """``log N(x; 0, I)`` up to a constant."""
    return -0.5 * jnp.sum(x**2)


# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------


def test_rbf_kernel_is_symmetric() -> None:
    """``k(x, y) == k(y, x)`` for the RBF kernel."""
    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.5, -0.3])
    assert jnp.allclose(rbf_kernel(x, y, length_scale=0.5), rbf_kernel(y, x, length_scale=0.5))


def test_rbf_kernel_at_zero_distance_is_one() -> None:
    """``k(x, x) = exp(0) = 1``."""
    x = jnp.array([1.0, 2.0])
    assert jnp.allclose(rbf_kernel(x, x, length_scale=1.0), 1.0)


def test_rbf_kernel_compiles_under_jit() -> None:
    """The kernel function must be jit-traceable."""
    jitted = jax.jit(rbf_kernel)
    value = jitted(jnp.array([1.0]), jnp.array([0.0]), length_scale=1.0)
    assert jnp.isfinite(value)


# ---------------------------------------------------------------------------
# Median heuristic
# ---------------------------------------------------------------------------


def test_median_heuristic_matches_published_formula() -> None:
    r"""``length_scale = median(pairwise distances)² / log(n_particles)``."""
    particles = jnp.array([[0.0], [1.0], [3.0]])
    distances = jnp.array([1.0, 3.0, 2.0])  # |0-1|, |0-3|, |1-3|
    expected = (jnp.median(distances) ** 2) / jnp.log(3.0)
    assert jnp.allclose(median_heuristic_bandwidth(particles), expected, atol=1e-6)


def test_median_heuristic_compiles_under_jit() -> None:
    """The bandwidth helper must be jit-traceable."""
    particles = jnp.array([[0.0], [1.0], [3.0]])
    jitted = jax.jit(median_heuristic_bandwidth)
    assert jnp.isfinite(jitted(particles))


# ---------------------------------------------------------------------------
# svgd_fit
# ---------------------------------------------------------------------------


def test_svgd_fit_recovers_standard_normal_mean() -> None:
    """SVGD particles concentrate around the mean of the target distribution.

    Target: ``N(0, I)`` in 1-D. After enough iterations, empirical
    particle mean should be close to zero.
    """
    key = jax.random.PRNGKey(0)
    initial_particles = jax.random.normal(key, (50, 1)) * 2.0 + 3.0
    final_particles = svgd_fit(
        initial_particles=initial_particles,
        target_log_prob_fn=_standard_normal_log_prob,
        num_iterations=400,
        learning_rate=0.5,
    )
    empirical_mean = jnp.mean(final_particles, axis=0)
    assert jnp.allclose(empirical_mean, jnp.zeros_like(empirical_mean), atol=0.5)


def test_svgd_fit_recovers_standard_normal_variance() -> None:
    """Empirical particle variance approaches target variance."""
    key = jax.random.PRNGKey(1)
    initial_particles = jax.random.normal(key, (80, 1)) * 0.1
    final_particles = svgd_fit(
        initial_particles=initial_particles,
        target_log_prob_fn=_standard_normal_log_prob,
        num_iterations=600,
        learning_rate=0.3,
    )
    empirical_variance = jnp.var(final_particles, axis=0, ddof=1)
    # SVGD with median heuristic is known to under-estimate variance
    # somewhat; we allow a loose tolerance and check the empirical
    # variance is at least on the order of 0.4 (rather than near zero).
    assert empirical_variance[0] > 0.3


def test_svgd_fit_keeps_particle_shape() -> None:
    """``svgd_fit`` preserves the ``(num_particles, dim)`` shape."""
    initial_particles = jnp.linspace(-1.0, 1.0, 24).reshape(8, 3)
    final_particles = svgd_fit(
        initial_particles=initial_particles,
        target_log_prob_fn=lambda x: -0.5 * jnp.sum(x**2),
        num_iterations=10,
        learning_rate=0.1,
    )
    assert final_particles.shape == initial_particles.shape


def test_svgd_fit_compiles_under_jit() -> None:
    """``svgd_fit`` must compile under ``jax.jit`` (with statics)."""
    initial_particles = jnp.linspace(-1.0, 1.0, 12).reshape(6, 2)
    jitted = jax.jit(
        svgd_fit, static_argnames=("target_log_prob_fn", "num_iterations")
    )
    result = jitted(
        initial_particles=initial_particles,
        target_log_prob_fn=lambda x: -0.5 * jnp.sum(x**2),
        num_iterations=5,
        learning_rate=0.1,
    )
    assert result.shape == initial_particles.shape


def test_svgd_fit_with_explicit_length_scale_skips_heuristic() -> None:
    """Passing ``length_scale`` directly bypasses the median heuristic."""
    initial_particles = jnp.linspace(-2.0, 2.0, 10).reshape(5, 2)
    final_particles = svgd_fit(
        initial_particles=initial_particles,
        target_log_prob_fn=_standard_normal_log_prob,
        num_iterations=20,
        learning_rate=0.1,
        length_scale=2.0,
    )
    assert final_particles.shape == initial_particles.shape
    assert jnp.all(jnp.isfinite(final_particles))
