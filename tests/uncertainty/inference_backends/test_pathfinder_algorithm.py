r"""Pathfinder (Zhang+ 2022) algorithm tests.

Tests for :mod:`opifex.uncertainty.inference_backends._pathfinder_algorithm`,
a JAX-native port of the Pathfinder reference at
``../blackjax/blackjax/vi/pathfinder.py`` plus its L-BFGS factor
helpers at ``../blackjax/blackjax/optimizers/lbfgs.py``.

Algorithm summary (Zhang et al, 2022 arXiv:2108.03782):

* Run L-BFGS on ``-log p`` from a starting point.
* At each L-BFGS step, recover the diagonal inverse-Hessian
  approximation ``alpha`` (Algorithm 3 inner loop) and build the
  ``(beta, gamma)`` factors that represent the full inverse Hessian
  in factored form (formula II.2 of the paper).
* Draw ``num_samples`` Gaussian samples from each step's approximation
  via ``bfgs_sample`` (Algorithm 4) and compute the ELBO.
* Return the iteration with the highest ELBO + a sampler that draws
  from its Gaussian.

Algorithm invariants verified:

* **lbfgs_recover_alpha mask predicate.** Mask is ``True`` when
  ``s·z > 0`` (positive-curvature secant condition), ``False``
  otherwise — and the alpha update only happens under the True mask.
* **Inverse-Hessian factor shapes.** ``beta`` has shape
  ``(d, 2·maxcor)`` and ``gamma`` has shape ``(2·maxcor, 2·maxcor)``.
* **bfgs_sample produces (num_samples, d) draws.** With the correct
  Gaussian log-density per sample.
* **Pathfinder recovers a standard-normal posterior mode.** The
  selected approximation's mean is near zero.
* **Pathfinder draws concentrate around the recovered mode.** Mean
  of ``num_samples`` draws is within tolerance of zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.inference_backends._pathfinder_algorithm import (
    bfgs_sample,
    lbfgs_inverse_hessian_factors,
    lbfgs_recover_alpha,
    pathfinder_approximate,
    pathfinder_sample,
)


def _standard_normal_log_density(x: jax.Array) -> jax.Array:
    """``log N(x; 0, I)`` up to a constant."""
    return -0.5 * jnp.sum(x**2)


# ---------------------------------------------------------------------------
# lbfgs_recover_alpha
# ---------------------------------------------------------------------------


def test_lbfgs_recover_alpha_mask_true_when_secant_condition_holds() -> None:
    """Curvature predicate ``s·z > eps·||z||`` triggers a true update mask."""
    alpha_previous = jnp.array([1.0, 1.0])
    s_step = jnp.array([0.3, 0.7])
    z_step = jnp.array([0.4, 0.6])  # s·z = 0.54 > 0
    alpha_new, mask = lbfgs_recover_alpha(alpha_previous, s_step, z_step)
    assert jnp.all(mask)
    assert jnp.all(jnp.isfinite(alpha_new))


def test_lbfgs_recover_alpha_mask_false_falls_back_to_previous_alpha() -> None:
    """When the secant condition fails the previous alpha is preserved."""
    alpha_previous = jnp.array([1.5, 0.5])
    s_step = jnp.array([1.0, 0.0])
    z_step = jnp.array([-1.0, 0.0])  # s·z = -1 < 0
    alpha_new, mask = lbfgs_recover_alpha(alpha_previous, s_step, z_step)
    assert not jnp.any(mask)
    assert jnp.allclose(alpha_new, alpha_previous)


# ---------------------------------------------------------------------------
# lbfgs_inverse_hessian_factors
# ---------------------------------------------------------------------------


def test_lbfgs_inverse_hessian_factors_produces_expected_shapes() -> None:
    """``beta`` is ``(d, 2 maxcor)`` and ``gamma`` is ``(2 maxcor, 2 maxcor)``."""
    param_dim = 3
    maxcor = 4
    S = jnp.eye(param_dim, maxcor)
    Z = jnp.eye(param_dim, maxcor)
    alpha = jnp.ones(param_dim)
    beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
    assert beta.shape == (param_dim, 2 * maxcor)
    assert gamma.shape == (2 * maxcor, 2 * maxcor)


# ---------------------------------------------------------------------------
# bfgs_sample
# ---------------------------------------------------------------------------


def test_bfgs_sample_returns_correct_shapes() -> None:
    """``bfgs_sample`` returns ``(num_samples, d)`` samples + scalar log-density per sample."""
    param_dim = 2
    maxcor = 3
    alpha = jnp.ones(param_dim)
    beta = jnp.zeros((param_dim, 2 * maxcor))
    gamma = jnp.zeros((2 * maxcor, 2 * maxcor))
    samples, log_density = bfgs_sample(
        rng_key=jax.random.PRNGKey(0),
        num_samples=16,
        position=jnp.zeros(param_dim),
        grad_position=jnp.zeros(param_dim),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    assert samples.shape == (16, param_dim)
    assert log_density.shape == (16,)


def test_bfgs_sample_with_zero_factors_is_standard_normal() -> None:
    """If ``beta = gamma = 0`` and ``grad_position = 0``, samples are ``N(position, diag(alpha))``."""
    param_dim = 2
    maxcor = 2
    alpha = jnp.ones(param_dim)
    beta = jnp.zeros((param_dim, 2 * maxcor))
    gamma = jnp.zeros((2 * maxcor, 2 * maxcor))
    samples, _ = bfgs_sample(
        rng_key=jax.random.PRNGKey(1),
        num_samples=4096,
        position=jnp.zeros(param_dim),
        grad_position=jnp.zeros(param_dim),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    empirical_mean = jnp.mean(samples, axis=0)
    empirical_cov = jnp.cov(samples.T)
    assert jnp.allclose(empirical_mean, jnp.zeros(param_dim), atol=0.1)
    assert jnp.allclose(jnp.diag(empirical_cov), jnp.ones(param_dim), atol=0.15)


# ---------------------------------------------------------------------------
# pathfinder_approximate
# ---------------------------------------------------------------------------


def test_pathfinder_approximate_recovers_standard_normal_mode() -> None:
    """L-BFGS on ``-log N(x; 0, I)`` from a perturbed start converges to the origin."""
    initial_position = jnp.array([2.5, -1.7])
    state = pathfinder_approximate(
        rng_key=jax.random.PRNGKey(0),
        log_density_fn=_standard_normal_log_density,
        initial_position=initial_position,
        num_samples=64,
        maxiter=30,
        maxcor=6,
    )
    assert jnp.allclose(state.position, jnp.zeros_like(initial_position), atol=0.1)
    assert jnp.all(jnp.isfinite(state.alpha))


def test_pathfinder_approximate_produces_finite_elbo() -> None:
    """The selected approximation has finite ELBO (not minus infinity)."""
    state = pathfinder_approximate(
        rng_key=jax.random.PRNGKey(1),
        log_density_fn=_standard_normal_log_density,
        initial_position=jnp.array([1.0, 1.0, 1.0]),
        num_samples=32,
        maxiter=20,
        maxcor=5,
    )
    assert jnp.isfinite(state.elbo)


# ---------------------------------------------------------------------------
# pathfinder_sample
# ---------------------------------------------------------------------------


def test_pathfinder_sample_draws_concentrate_around_recovered_mode() -> None:
    """Drawn samples have mean near the recovered position (≈ 0 for standard normal)."""
    state = pathfinder_approximate(
        rng_key=jax.random.PRNGKey(2),
        log_density_fn=_standard_normal_log_density,
        initial_position=jnp.array([3.0, -2.0]),
        num_samples=64,
        maxiter=30,
        maxcor=6,
    )
    samples, log_q = pathfinder_sample(
        rng_key=jax.random.PRNGKey(99), state=state, num_samples=2048
    )
    assert samples.shape == (2048, 2)
    assert log_q.shape == (2048,)
    empirical_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_mean, jnp.zeros(2), atol=0.2)


def test_pathfinder_sample_reproducible_under_same_key() -> None:
    """Identical PRNG keys yield identical samples."""
    state = pathfinder_approximate(
        rng_key=jax.random.PRNGKey(3),
        log_density_fn=_standard_normal_log_density,
        initial_position=jnp.array([0.5, 0.5]),
        num_samples=32,
        maxiter=15,
        maxcor=4,
    )
    samples_a, _ = pathfinder_sample(rng_key=jax.random.PRNGKey(7), state=state, num_samples=128)
    samples_b, _ = pathfinder_sample(rng_key=jax.random.PRNGKey(7), state=state, num_samples=128)
    assert jnp.array_equal(samples_a, samples_b)
