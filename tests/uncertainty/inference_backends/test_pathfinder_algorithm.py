r"""Pathfinder (Zhang+ 2022) algorithm tests.

Tests for :mod:`opifex.uncertainty.inference_backends._pathfinder_algorithm`,
the JAX-native Pathfinder primitives and their L-BFGS inverse-Hessian
factor helpers.

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

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from opifex.uncertainty.inference_backends._pathfinder_algorithm import (
    bfgs_sample,
    lbfgs_inverse_hessian_factors,
    lbfgs_recover_alpha,
    pathfinder_approximate,
    pathfinder_sample,
)


_EPS32 = float(np.finfo(np.float32).eps)

# Float32 error of log q against float64, in units of eps32 times the magnitude of its terms: at
# most 1.30 for diagonal covariances (dimensions 8 to 20000, alpha 0.01 to 50) and 2.12 for
# factored covariances from 32 random quadratics up to 200 dimensions (condition numbers up to 20).
_LOG_Q_ULPS = 5.0


def _standard_normal_log_density(x: jax.Array) -> jax.Array:
    """``log N(x; 0, I)`` up to a constant."""
    return -0.5 * jnp.sum(x**2)


def _log_q_rounding_bound(log_det: float, squared_noise: np.ndarray, dim: int) -> np.ndarray:
    """Float32 rounding of ``-(log|Sigma| + u^T u + N log 2 pi) / 2``, in units of its terms."""
    return _LOG_Q_ULPS * _EPS32 * (abs(log_det) + squared_noise + dim * math.log(2.0 * math.pi))


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


@pytest.mark.parametrize("alpha_value", [0.5, 2.0])
@pytest.mark.parametrize("dim", [127, 128, 3000])
def test_bfgs_sample_log_density_is_the_diagonal_gaussian_in_high_dimensions(
    dim: int, alpha_value: float
) -> None:
    """With ``beta = gamma = 0`` the draws are ``N(position, diag(alpha))`` and ``log q`` is theirs.

    Algorithm 4, step 7 of Zhang et al. (2022) evaluates
    ``log|Sigma| = log|diag(alpha)| + 2 log|L~|`` as sums of logarithms. In float32 the product of
    128 factors of 2 overflows and the product of 127 factors of 0.5 underflows to zero, so a log of
    the product is not finite.
    """
    samples, log_q = bfgs_sample(
        rng_key=jax.random.PRNGKey(0),
        num_samples=4,
        position=jnp.zeros(dim),
        grad_position=jnp.zeros(dim),
        alpha=jnp.full((dim,), alpha_value),
        beta=jnp.zeros((dim, 6)),
        gamma=jnp.zeros((6, 6)),
    )
    draws = np.asarray(samples, dtype=np.float64)
    reference = norm.logpdf(draws, loc=0.0, scale=math.sqrt(alpha_value)).sum(axis=1)
    bound = _log_q_rounding_bound(
        dim * math.log(alpha_value), (draws**2 / alpha_value).sum(axis=1), dim
    )
    error = np.abs(np.asarray(log_q, dtype=np.float64) - reference)
    assert np.all(np.isfinite(np.asarray(log_q)))
    assert np.all(error <= bound), error / bound


def test_bfgs_sample_log_density_is_the_factored_gaussian() -> None:
    """``log q`` is the density of ``N(position, diag(alpha) + beta gamma beta^T)`` (formula II.2).

    The gradient is zero so the mean is the position; the factors come from curvature pairs of a
    quadratic, and the reference evaluates the full covariance in float64.
    """
    rng = np.random.default_rng(0)
    dim, maxcor = 5, 3
    basis = rng.normal(size=(dim, dim))
    hessian = basis @ basis.T + dim * np.eye(dim)
    position_steps = rng.normal(size=(dim, maxcor))
    alpha = np.full(dim, 0.3)
    beta, gamma = lbfgs_inverse_hessian_factors(
        jnp.asarray(position_steps, dtype=jnp.float32),
        jnp.asarray(hessian @ position_steps, dtype=jnp.float32),
        jnp.asarray(alpha, dtype=jnp.float32),
    )
    position = rng.normal(size=dim)
    samples, log_q = bfgs_sample(
        rng_key=jax.random.PRNGKey(1),
        num_samples=256,
        position=jnp.asarray(position, dtype=jnp.float32),
        grad_position=jnp.zeros(dim),
        alpha=jnp.asarray(alpha, dtype=jnp.float32),
        beta=beta,
        gamma=gamma,
    )
    beta64 = np.asarray(beta, dtype=np.float64)
    covariance = np.diag(alpha) + beta64 @ np.asarray(gamma, dtype=np.float64) @ beta64.T
    covariance = 0.5 * (covariance + covariance.T)
    draws = np.asarray(samples, dtype=np.float64)
    distribution = multivariate_normal(mean=position, cov=covariance)  # pyright: ignore[reportArgumentType]
    reference = distribution.logpdf(draws)
    centred = draws - position
    squared_noise = np.einsum("mi,ij,mj->m", centred, np.linalg.inv(covariance), centred)
    bound = _log_q_rounding_bound(np.linalg.slogdet(covariance)[1], squared_noise, dim)
    error = np.abs(np.asarray(log_q, dtype=np.float64) - reference)
    assert np.all(error <= bound), error / bound


# Float32 errors against the paper's formulas in float64, measured over 20 seeds before and after
# the rewrite: the alpha update is within 3.45 eps32 of the rounding scale of its three terms
# (dimensions 5 to 5000); beta within 1.05 eps32 (|beta| + 1) and gamma within
# 0.236 eps32 cond(E) (max|gamma| + 1) (dimensions 10 to 2000, cond(E) <= 3.5).
_ALPHA_ULPS = 7.0
_BETA_ULPS = 2.5
_GAMMA_ULPS = 0.5


def test_lbfgs_recover_alpha_matches_algorithm_3() -> None:
    """The diagonal update is line 9 of Algorithm 3 of Zhang et al. (2022).

    ``1 / alpha_n = a / (b alpha'_n) + z_n^2 / b - a s_n^2 / (b c alpha'^2_n)`` with
    ``a = z^T diag(alpha') z``, ``b = z^T s`` and ``c = s^T diag(alpha')^{-1} s`` (line 7). The
    pair comes from a diagonal quadratic, so ``s^T z > 0`` and the update fires.
    """
    rng = np.random.default_rng(3)
    dim = 40
    alpha_previous = rng.uniform(0.2, 3.0, size=dim)
    position_step = rng.normal(size=dim)
    gradient_step = rng.uniform(0.5, 2.0, size=dim) * position_step
    alpha_new, mask = lbfgs_recover_alpha(
        jnp.asarray(alpha_previous, dtype=jnp.float32),
        jnp.asarray(position_step, dtype=jnp.float32),
        jnp.asarray(gradient_step, dtype=jnp.float32),
    )
    a = np.sum(alpha_previous * gradient_step**2)
    b = gradient_step @ position_step
    c = np.sum(position_step**2 / alpha_previous)
    terms = (
        a / (b * alpha_previous),
        gradient_step**2 / b,
        a * position_step**2 / (b * c * alpha_previous**2),
    )
    inverse = terms[0] + terms[1] - terms[2]
    reference = 1.0 / inverse
    # Rounding of the three terms, relative to the inverse they sum to.
    bound = _ALPHA_ULPS * _EPS32 * (terms[0] + terms[1] + terms[2]) / np.abs(inverse)
    relative_error = np.abs(np.asarray(alpha_new, dtype=np.float64) - reference) / reference
    assert bool(np.all(np.asarray(mask)))
    assert np.all(relative_error <= bound), relative_error / bound


def test_lbfgs_inverse_hessian_factors_match_algorithm_4() -> None:
    """``beta`` and ``gamma`` are steps 3 and 4 of Algorithm 4 of Zhang et al. (2022).

    ``E`` is the upper triangle of ``S^T Z`` with diagonal ``eta``,
    ``beta = [diag(alpha) Z, S]`` and
    ``gamma = [[0, -E^{-1}], [-E^{-T}, E^{-T} (diag(eta) + Z^T diag(alpha) Z) E^{-1}]]``.
    """
    rng = np.random.default_rng(4)
    dim, maxcor = 30, 4
    basis = rng.normal(size=(dim, dim))
    hessian = basis @ basis.T / dim + np.eye(dim)
    position_steps = rng.normal(size=(dim, maxcor))
    gradient_steps = hessian @ position_steps
    alpha = rng.uniform(0.2, 2.0, size=dim)
    beta, gamma = lbfgs_inverse_hessian_factors(
        jnp.asarray(position_steps, dtype=jnp.float32),
        jnp.asarray(gradient_steps, dtype=jnp.float32),
        jnp.asarray(alpha, dtype=jnp.float32),
    )
    upper = np.triu(position_steps.T @ gradient_steps)
    inverse_upper = np.linalg.inv(upper)
    scaled_gradients = alpha[:, None] * gradient_steps
    beta_reference = np.hstack([scaled_gradients, position_steps])
    gamma_reference = np.block(
        [
            [np.zeros((maxcor, maxcor)), -inverse_upper],
            [
                -inverse_upper.T,
                inverse_upper.T
                @ (np.diag(np.diag(upper)) + gradient_steps.T @ scaled_gradients)
                @ inverse_upper,
            ],
        ]
    )
    beta_error = np.abs(np.asarray(beta, dtype=np.float64) - beta_reference)
    assert np.all(beta_error <= _BETA_ULPS * _EPS32 * (np.abs(beta_reference) + 1.0)), (
        beta_error.max()
    )
    condition = np.linalg.cond(upper)
    gamma_error = np.abs(np.asarray(gamma, dtype=np.float64) - gamma_reference)
    gamma_bound = _GAMMA_ULPS * _EPS32 * condition * (np.abs(gamma_reference).max() + 1.0)
    assert np.all(gamma_error <= gamma_bound), gamma_error.max() / gamma_bound


# Float32 error of half the difference between draws at g and -g against Sigma g, in units of
# eps32 times |x| + |Sigma g| + 1: at most 1.70 over 18 random quadratics up to 200 dimensions.
_MEAN_ULPS = 4.0


def test_bfgs_sample_moves_the_mean_by_sigma_times_the_gradient() -> None:
    """Draws at gradients ``g`` and ``-g`` with the same noise differ by ``2 Sigma g``, one sign.

    With ``Sigma = diag(alpha) + beta gamma beta^T`` (formula II.2 of Zhang et al., 2022) the mean
    moves by ``Sigma g`` with a sign fixed by the gradient convention. The noise cancels between the
    two calls, so half their difference must equal ``Sigma g`` or ``-Sigma g`` in every component
    and draw.
    """
    rng = np.random.default_rng(5)
    dim, maxcor = 50, 6
    basis = rng.normal(size=(dim, dim))
    hessian = basis @ basis.T / dim + np.eye(dim)
    position_steps = rng.normal(size=(dim, maxcor))
    alpha = rng.uniform(0.1, 2.0, size=dim)
    beta, gamma = lbfgs_inverse_hessian_factors(
        jnp.asarray(position_steps, dtype=jnp.float32),
        jnp.asarray(hessian @ position_steps, dtype=jnp.float32),
        jnp.asarray(alpha, dtype=jnp.float32),
    )
    gradient = rng.normal(size=dim)

    def draws(sign: float) -> np.ndarray:
        samples, _ = bfgs_sample(
            rng_key=jax.random.PRNGKey(5),
            num_samples=4,
            position=jnp.full((dim,), 1.5),
            grad_position=jnp.asarray(sign * gradient, dtype=jnp.float32),
            alpha=jnp.asarray(alpha, dtype=jnp.float32),
            beta=beta,
            gamma=gamma,
        )
        return np.asarray(samples, dtype=np.float64)

    draws_plus, draws_minus = draws(1.0), draws(-1.0)
    beta64 = np.asarray(beta, dtype=np.float64)
    covariance = np.diag(alpha) + beta64 @ np.asarray(gamma, dtype=np.float64) @ beta64.T
    shift = covariance @ gradient
    half_difference = (draws_plus - draws_minus) / 2.0
    bound = _MEAN_ULPS * _EPS32 * (np.abs(draws_plus) + np.abs(shift) + 1.0)
    matches_plus = bool(np.all(np.abs(half_difference - shift) <= bound))
    matches_minus = bool(np.all(np.abs(half_difference + shift) <= bound))
    assert matches_plus or matches_minus, np.abs(half_difference).max()


def _square_intermediates(jaxpr: object, dim: int) -> list[str]:
    """Intermediates with two or more axes of length ``dim`` or more, in a jaxpr and sub-jaxprs."""
    found: list[str] = []

    def walk(inner: object) -> None:
        for equation in getattr(inner, "eqns", ()):
            for variable in equation.outvars:
                shape = getattr(variable.aval, "shape", ())
                if sum(1 for size in shape if size >= dim) >= 2:
                    found.append(f"{equation.primitive}{tuple(shape)}")
            for parameter in equation.params.values():
                items = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
                for item in items:
                    if hasattr(item, "jaxpr"):
                        walk(item.jaxpr)
                    elif hasattr(item, "eqns"):
                        walk(item)

    walk(jaxpr)
    return found


def test_pathfinder_primitives_do_not_materialise_dimension_squared_intermediates() -> None:
    """The diagonal update, the factors and the sampler stay linear in the dimension.

    Algorithm 4 of Zhang et al. (2022) costs ``O(J N + J^2)`` per draw. ``diag(alpha)`` as a
    matrix, or ``beta @ gamma @ beta^T`` evaluated left to right, creates ``N x N`` intermediates.
    """
    dim, maxcor = 500, 6
    alpha = jnp.full((dim,), 0.7)
    position_steps = jnp.ones((dim, maxcor))
    gradient_steps = jnp.ones((dim, maxcor))
    beta, gamma = lbfgs_inverse_hessian_factors(position_steps, gradient_steps, alpha)
    jaxprs = {
        "lbfgs_recover_alpha": jax.make_jaxpr(lbfgs_recover_alpha)(
            alpha, jnp.ones(dim), jnp.ones(dim)
        ),
        "lbfgs_inverse_hessian_factors": jax.make_jaxpr(lbfgs_inverse_hessian_factors)(
            position_steps, gradient_steps, alpha
        ),
        "bfgs_sample": jax.make_jaxpr(
            lambda key, position, gradient, diagonal, factor, core: bfgs_sample(
                rng_key=key,
                num_samples=4,
                position=position,
                grad_position=gradient,
                alpha=diagonal,
                beta=factor,
                gamma=core,
            )
        )(jax.random.PRNGKey(0), jnp.zeros(dim), jnp.zeros(dim), alpha, beta, gamma),
    }
    found = {name: _square_intermediates(jaxpr, dim) for name, jaxpr in jaxprs.items()}
    assert not any(found.values()), found


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
