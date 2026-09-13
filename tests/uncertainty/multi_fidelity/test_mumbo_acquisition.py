r"""MUMBO multi-fidelity Bayesian-optimisation acquisition — slice 35.

Tests the Moss, Leslie, Rayson 2020 MUMBO acquisition built on top of
the linear multi-fidelity GP from slice 33. MUMBO extends MES
(Max-value Entropy Search) to multi-fidelity by adding fidelity as a
candidate dimension and weighting information gain by query cost.

The opifex implementation accepts a fitted multi-fidelity GP state and
a batch of ``(x, fidelity_level)`` candidates, and returns one
acquisition score per candidate. Cost weighting is applied by the
caller (the canonical recipe divides the acquisition by per-level
cost; see ``MUMBO/cost``).
"""

from __future__ import annotations

import functools
import math
from itertools import pairwise
from typing import cast, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.stats import norm


if TYPE_CHECKING:
    from collections.abc import Callable

    from opifex.uncertainty.multi_fidelity import LinearMultiFidelityGPState


_EPS32 = float(np.finfo(np.float32).eps)
_SIMPSON_POINTS = 5001


def test_mumbo_acquisition_returns_one_score_per_candidate() -> None:
    """``mumbo_acquisition`` returns ``(m,)`` for ``m`` candidate (x, level) pairs."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_candidates = jnp.linspace(0.05, 0.95, 8).reshape(-1, 1)
    candidate_levels = jnp.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = mumbo_acquisition(
        state=state,
        x_candidates=x_candidates,
        candidate_levels=candidate_levels,
        target_level=1,
        rng_key=jax.random.PRNGKey(0),
        grid_size=200,
        num_gumbel_samples=8,
    )
    assert scores.shape == (8,)
    assert jnp.all(jnp.isfinite(scores))


def test_mumbo_acquisition_is_non_negative() -> None:
    """MUMBO is an information-gain quantity — non-negative up to numerical noise."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 15).reshape(-1, 1)
    x_high = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_candidates = jnp.linspace(0.05, 0.95, 6).reshape(-1, 1)
    candidate_levels = jnp.zeros((6,), dtype=jnp.int32)
    scores = mumbo_acquisition(
        state=state,
        x_candidates=x_candidates,
        candidate_levels=candidate_levels,
        target_level=1,
        rng_key=jax.random.PRNGKey(1),
        grid_size=200,
        num_gumbel_samples=8,
    )
    assert jnp.all(scores > -1e-4)


def test_mumbo_acquisition_low_fidelity_candidate_carries_information() -> None:
    """A low-fidelity candidate carries non-trivial information about the high-fidelity max.

    With a strong AR(1) coupling (scaling factor 0.95), observing the low fidelity near the
    maximum of the target reduces uncertainty about that maximum: at ``x = 0.25`` the float64
    eq. 5 of Moss et al. (2020) gives 1.26e-3 nats for these samples. Away from the maximum the
    information vanishes: at ``x = 0.45`` every sample of the maximum lies 14.6 standard deviations
    above the target mean and eq. 5 gives 1.3e-49.
    """
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 16).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(0.95,),
        noise_std=0.05,
    )
    x_test = jnp.array([[0.25]])
    score = mumbo_acquisition(
        state=state,
        x_candidates=x_test,
        candidate_levels=jnp.array([0]),
        target_level=1,
        rng_key=jax.random.PRNGKey(2),
        grid_size=200,
        num_gumbel_samples=12,
    )
    assert jnp.all(jnp.isfinite(score))
    assert float(score[0]) > 1e-6


# -----------------------------------------------------------------------------
# Samples of the maximum (Wang & Jegelka 2017, §3.1)
# -----------------------------------------------------------------------------


def _mean_field_maximum_quantile(probability: float, means: np.ndarray, stds: np.ndarray) -> float:
    """Level ``y`` with ``prod_i Phi((y - mu_i) / sigma_i) = probability``, in float64."""

    def objective(level: float) -> float:
        return float(np.sum(norm.logcdf((level - means) / stds))) - math.log(probability)

    return cast(
        "float",
        optimize.brentq(
            objective, float(np.min(means - 10.0 * stds)), float(np.max(means + 10.0 * stds))
        ),
    )


def test_gumbel_samples_match_the_quartiles_of_the_mean_field_maximum() -> None:
    """The Gumbel's quartiles match those of ``prod_i Phi((y - mu_i) / sigma_i)``.

    Wang & Jegelka (2017), §3.1, fit ``a - b log(-log r)`` at ``r = 0.25`` and ``r = 0.75``. The fit
    interpolates the CDF on 200 levels, so each fitted quartile lies within one level spacing of the
    exact quartile. The sample quartiles add Monte Carlo error ``sqrt(p (1 - p) / n) / density``, and a
    Gumbel of scale ``b`` has density ``-p log(p) / b`` at its quantile ``p``.
    """
    from opifex.uncertainty.multi_fidelity.mumbo import _gumbel_fit_and_sample

    rng = np.random.default_rng(0)
    means = rng.normal(0.0, 0.5, 200)
    stds = rng.uniform(0.1, 0.5, 200)
    num_samples = 200_000
    samples = np.asarray(
        _gumbel_fit_and_sample(
            grid_means=jnp.asarray(means, dtype=jnp.float32),
            grid_stds=jnp.asarray(stds, dtype=jnp.float32),
            num_samples=num_samples,
            rng_key=jax.random.PRNGKey(0),
        ),
        dtype=np.float64,
    )
    lower, upper = (_mean_field_maximum_quantile(p, means, stds) for p in (0.25, 0.75))
    level_spacing = float(np.max(means + 5.0 * stds) - np.min(means - 5.0 * stds)) / 199.0
    scale = (upper - lower) / math.log(math.log(4.0) / math.log(4.0 / 3.0))
    for probability, exact in ((0.25, lower), (0.75, upper)):
        density = -probability * math.log(probability) / scale
        standard_error = math.sqrt(probability * (1.0 - probability) / num_samples) / density
        error = abs(float(np.quantile(samples, probability)) - exact)
        assert error <= level_spacing + 5.0 * standard_error, (probability, error)


# -----------------------------------------------------------------------------
# Information gain for one sample of the maximum (Moss et al. 2020, eq. 5)
# -----------------------------------------------------------------------------


def _eq5_reference(correlation: float, gamma: float) -> float:
    """Float64 eq. 5 of Moss et al. (2020) for one sample, by quadrature of the ESG of Appendix A.1.

    ``Z = (y - mu_f) / sqrt(sigma_f^2 + sigma^2) | g < g*`` has density
    ``phi(t) Phi((gamma - rho t) / sqrt(1 - rho^2)) / Phi(gamma)``. The range is split at the mean and
    around the step of the skew factor at ``t = gamma / rho``. The quadrature checks that the density
    integrates to one and that its mean is ``-rho phi(gamma) / Phi(gamma)``. At ``|rho| = 1`` eq. 5 is
    MES (Wang & Jegelka 2017, eq. 6).
    """
    log_cdf_gamma = float(norm.logcdf(gamma))
    ratio = math.exp(float(norm.logpdf(gamma)) - log_cdf_gamma)
    if abs(correlation) == 1.0:
        return 0.5 * gamma * ratio - log_cdf_gamma
    skew = math.sqrt(1.0 - correlation**2)
    mean = -correlation * ratio
    std = math.sqrt(1.0 - correlation**2 * ratio * (gamma + ratio))

    def log_density(theta: float) -> float:
        return (
            float(norm.logpdf(theta) + norm.logcdf((gamma - correlation * theta) / skew))
            - log_cdf_gamma
        )

    lower, upper = mean - 40.0 * std, mean + 40.0 * std
    edges = {lower, upper, mean}
    if correlation != 0.0:
        width = skew / abs(correlation)
        for offset in (-20.0, -5.0, -2.0, 0.0, 2.0, 5.0, 20.0):
            point = gamma / correlation + offset * width
            if lower < point < upper:
                edges.add(point)
    breakpoints = sorted(edges)

    def integral(function: Callable[[float], float]) -> float:
        return sum(
            float(integrate.quad(function, a, b, limit=2000, epsabs=1e-15, epsrel=1e-12)[0])
            for a, b in pairwise(breakpoints)
        )

    assert abs(integral(lambda t: math.exp(log_density(t))) - 1.0) < 1e-10
    assert abs(integral(lambda t: t * math.exp(log_density(t))) - mean) < 1e-10
    last_term = integral(
        lambda t: math.exp(log_density(t)) * float(norm.logcdf((gamma - correlation * t) / skew))
    )
    return 0.5 * correlation**2 * gamma * ratio - log_cdf_gamma + last_term


@functools.cache
def _phi_log_phi_area() -> float:
    """``C = int |Phi(u) log Phi(u)| du``, the area of the skew factor's transition in eq. 5."""
    return float(
        integrate.quad(
            lambda u: -float(norm.cdf(u)) * float(norm.logcdf(u)), -40.0, 40.0, limit=500
        )[0]
    )


def _eq5_float32_tolerance(correlation: float, gamma: float, num_quadrature_points: int) -> float:
    """Float32 error bound of eq. 5 by composite Simpson's rule.

    Rounding: 32 float32 units in the last place of ``1 + |log Phi(gamma)| + rho^2 |gamma| r / 2``,
    twice the largest coefficient measured (15.6, at ``rho = 0.5`` and ``gamma = -10``, where the
    expectation nearly cancels ``-log Phi(gamma)``).

    Step: when ``s = sqrt(1 - rho^2)`` is narrower than one Simpson panel ``2h``, with node spacing
    ``h = 16 std(Z) / (n - 1)``, the integrand ``p log Phi(u)`` is a spike of height at most
    ``phi(gamma / rho) / (e Phi(gamma))`` and area at most ``C s phi(gamma / rho) / (|rho| Phi(gamma))``.
    Simpson's rule misses the area or over-weights the spike by at most one node weight ``4h / 3``.

    At 5001 points the measured errors stay below 0.70 of this bound over 360 points with ``rho``
    from -0.95 to 1 and ``gamma`` from -10 to 6; with the step regime cut at ``s <= h`` instead, one
    point with ``s / h = 1.06`` exceeds it.
    """
    log_cdf_gamma = float(norm.logcdf(gamma))
    ratio = math.exp(float(norm.logpdf(gamma)) - log_cdf_gamma)
    rounding = (
        32.0 * _EPS32 * (1.0 + abs(log_cdf_gamma) + 0.5 * correlation**2 * abs(gamma) * ratio)
    )
    std = math.sqrt(max(1.0 - correlation**2 * ratio * (gamma + ratio), 1e-10))
    spacing = 16.0 * std / (num_quadrature_points - 1)
    skew = math.sqrt(max(1.0 - correlation**2, 1e-10))
    if correlation == 0.0 or skew > 2.0 * spacing:
        return rounding
    height = math.exp(float(norm.logpdf(gamma / correlation)) - log_cdf_gamma)
    step = height * (4.0 * spacing / (3.0 * math.e) + _phi_log_phi_area() * skew / abs(correlation))
    return rounding + step


@pytest.mark.parametrize(
    "correlation", [-0.95, -0.5, 0.0, 0.5, 0.9, 0.99, 0.9999, 0.99999, 0.999999, 1.0]
)
def test_information_gain_matches_eq5_for_one_sample_of_the_maximum(correlation: float) -> None:
    """Eq. 5 of Moss et al. (2020) for ``g < g*``, which reduces to MES at ``rho = 1`` (§3.2).

    The reference is evaluated at the float32 inputs the implementation receives, and each error is
    held to :func:`_eq5_float32_tolerance`, which includes the first-order step error where
    ``sqrt(1 - rho^2)`` is narrower than a Simpson panel.
    """
    from opifex.uncertainty.multi_fidelity.mumbo import _mumbo_information_gain

    gammas = np.asarray([-10.0, -5.0, -2.0, -1.25, 0.0, 0.5, 1.5, 3.0, 6.0], dtype=np.float32)
    rho = np.float32(correlation)
    scores = jax.vmap(
        lambda gamma: _mumbo_information_gain(
            correlation=jnp.asarray(rho), gamma=gamma, num_quadrature_points=_SIMPSON_POINTS
        )
    )(jnp.asarray(gammas))
    reference = np.asarray([_eq5_reference(float(rho), float(gamma)) for gamma in gammas])
    tolerance = np.asarray(
        [_eq5_float32_tolerance(float(rho), float(gamma), _SIMPSON_POINTS) for gamma in gammas]
    )
    error = np.abs(np.asarray(scores, dtype=np.float64) - reference)
    assert np.all(error <= tolerance), error / tolerance


def test_information_gain_requires_an_odd_number_of_simpson_points() -> None:
    """Composite Simpson's rule needs an even number of intervals."""
    from opifex.uncertainty.multi_fidelity.mumbo import _mumbo_information_gain

    with pytest.raises(ValueError, match="odd"):
        _mumbo_information_gain(
            correlation=jnp.float32(0.5), gamma=jnp.float32(0.0), num_quadrature_points=1000
        )


# -----------------------------------------------------------------------------
# End to end: the bivariate predictive of the target and the noisy observation
# -----------------------------------------------------------------------------


def _fitted_state() -> LinearMultiFidelityGPState:
    from opifex.uncertainty.multi_fidelity import fit_linear_multi_fidelity_gp

    x_low = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 4).reshape(-1, 1)
    return fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(
            jnp.sin(2.0 * jnp.pi * x_low.flatten()) + 0.3,
            jnp.sin(2.0 * jnp.pi * x_high.flatten()),
        ),
        lengthscales=(0.25, 0.3),
        output_scales=(1.0, 0.4),
        scaling_factors=(0.8,),
        noise_std=0.2,
    )


def test_mumbo_scores_use_the_target_predictive_and_the_observation_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each score is eq. 5 averaged over samples of the maximum of the target level ``g``.

    Appendix A of Moss et al. (2020): ``gamma = (g* - mu_g) / sigma_g`` uses the target's latent
    predictive, and ``rho = Sigma / (sigma_g sqrt(sigma_f^2 + sigma^2))`` includes the observation
    noise ``sigma^2``. The reference builds these in float64 from the kernel. Each score is held to
    the mean of :func:`_eq5_float32_tolerance` over the samples; float32 rounding of the GP
    posterior added at most 5.8e-7 for noise standard deviations from 0.001 to 0.5.
    """
    import opifex.uncertainty.multi_fidelity.mumbo as mumbo_module
    from opifex.uncertainty.multi_fidelity import linear_multi_fidelity_kernel, mumbo_acquisition

    maxima = np.asarray([1.1, 1.3, 1.6])
    monkeypatch.setattr(
        mumbo_module, "_gumbel_fit_and_sample", lambda **_: jnp.asarray(maxima, dtype=jnp.float32)
    )
    state = _fitted_state()
    x_candidates = np.asarray([[0.25], [0.25], [0.3], [0.2]])
    levels = np.asarray([0, 1, 0, 0])
    scores = mumbo_acquisition(
        state=state,
        x_candidates=jnp.asarray(x_candidates, dtype=jnp.float32),
        candidate_levels=jnp.asarray(levels),
        target_level=1,
        rng_key=jax.random.PRNGKey(0),
        grid_size=50,
        num_gumbel_samples=3,
    )

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.asarray(
            linear_multi_fidelity_kernel(
                jnp.asarray(first, dtype=jnp.float32),
                jnp.asarray(second, dtype=jnp.float32),
                lengthscales=state.lengthscales,
                output_scales=state.output_scales,
                scaling_factors=state.scaling_factors,
                base_kernel_fn=state.base_kernel_fn,
            ),
            dtype=np.float64,
        )

    training_inputs = np.asarray(state.x_augmented, dtype=np.float64)
    noise_variance = state.noise_std**2
    gram = kernel(training_inputs, training_inputs) + noise_variance * np.eye(len(training_inputs))
    targets = np.asarray(state.y_train, dtype=np.float64)
    reference, tolerance = [], []
    for x, level in zip(x_candidates, levels, strict=True):
        candidate = np.concatenate([x, [float(level)]])[None, :]
        target = np.concatenate([x, [1.0]])[None, :]
        k_candidate = kernel(candidate, training_inputs)
        k_target = kernel(target, training_inputs)
        target_mean = (k_target @ np.linalg.solve(gram, targets)).item()
        target_var = (kernel(target, target) - k_target @ np.linalg.solve(gram, k_target.T)).item()
        candidate_var = (
            kernel(candidate, candidate) - k_candidate @ np.linalg.solve(gram, k_candidate.T)
        ).item()
        covariance = (
            kernel(candidate, target) - k_candidate @ np.linalg.solve(gram, k_target.T)
        ).item()
        correlation = covariance / math.sqrt(target_var * (candidate_var + noise_variance))
        gammas = (maxima - target_mean) / math.sqrt(target_var)
        reference.append(np.mean([_eq5_reference(correlation, gamma) for gamma in gammas]))
        tolerance.append(
            np.mean(
                [_eq5_float32_tolerance(correlation, gamma, _SIMPSON_POINTS) for gamma in gammas]
            )
        )
    error = np.abs(np.asarray(scores, dtype=np.float64) - np.asarray(reference))
    assert np.all(error <= np.asarray(tolerance)), (error, tolerance)


def test_mumbo_is_jit_compatible_and_differentiable_in_the_candidates() -> None:
    """The score compiles, and its gradient in the candidate inputs is finite."""
    from opifex.uncertainty.multi_fidelity import mumbo_acquisition

    state = _fitted_state()
    levels = jnp.asarray([0, 1, 0])

    def total(x_candidates: jax.Array) -> jax.Array:
        return jnp.sum(
            mumbo_acquisition(
                state=state,
                x_candidates=x_candidates,
                candidate_levels=levels,
                target_level=1,
                rng_key=jax.random.PRNGKey(3),
                grid_size=100,
                num_gumbel_samples=4,
            )
        )

    x_candidates = jnp.asarray([[0.2], [0.5], [0.8]])
    assert bool(jnp.allclose(jax.jit(total)(x_candidates), total(x_candidates), atol=1e-6))
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(x_candidates))))
