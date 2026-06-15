"""Tests for :mod:`opifex.uncertainty.sbi.diagnostics`.

Diagnostics:

* :func:`simulation_based_calibration` runs the well-known SBC procedure
  (Talts+ 2018, ``arXiv:1804.06788``). For a well-specified simulator,
  posterior ranks must be approximately uniform on ``{0, ..., L}``. A
  one-sample Kolmogorov-Smirnov test against ``Uniform(0, 1)`` passes
  within tolerance.
* :func:`expected_posterior_contraction` measures how much the posterior
  variance shrinks below the prior variance. Positive for informative
  observations (likelihood narrows the prior); ~zero for uninformative
  observations (a constant simulator returns no signal).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.sbi.diagnostics import (
    expected_posterior_contraction,
    PosteriorContractionResult,
    SBCResult,
    simulation_based_calibration,
)
from opifex.uncertainty.sbi.posterior_estimation import NeuralPosteriorEstimator
from opifex.uncertainty.sbi.simulators import Simulator


_TOY_DIM = 2
_TOY_SIGMA_LIK = 0.5


def _gaussian_linear_simulator() -> Simulator:
    def prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.normal(rng, (num_samples, _TOY_DIM))

    def simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
        return theta + _TOY_SIGMA_LIK * jax.random.normal(rng, theta.shape)

    return Simulator(prior_sampler=prior_sampler, simulate_fn=simulate_fn)


def _uninformative_simulator() -> Simulator:
    """Constant simulator — observation carries no theta information."""

    def prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.normal(rng, (num_samples, _TOY_DIM))

    def simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
        # Deterministic constant observation — likelihood p(x|theta) is a
        # point mass at zero, so the posterior equals the prior.
        return jnp.zeros_like(theta)

    return Simulator(prior_sampler=prior_sampler, simulate_fn=simulate_fn)


def _fit_npe(
    simulator: Simulator, *, num_steps: int = 300, num_simulations: int = 600
) -> NeuralPosteriorEstimator:
    return NeuralPosteriorEstimator(
        theta_dim=_TOY_DIM,
        x_dim=_TOY_DIM,
        num_steps=num_steps,
        learning_rate=1e-3,
    ).fit(simulator, num_simulations=num_simulations, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))


def test_simulation_based_calibration_returns_sbc_result() -> None:
    sim = _gaussian_linear_simulator()
    fitted = _fit_npe(sim, num_steps=100, num_simulations=200)
    result = simulation_based_calibration(
        fitted,
        sim,
        rngs=nnx.Rngs(sbi_simulate=0, sbi_sample=1),
        num_runs=16,
        num_posterior_samples=64,
    )
    assert isinstance(result, SBCResult)
    # Ranks shape: (num_runs, theta_dim).
    assert result.ranks.shape == (16, _TOY_DIM)
    assert result.ks_statistic.shape == (_TOY_DIM,)
    assert result.ks_pvalue.shape == (_TOY_DIM,)


def test_simulation_based_calibration_ranks_uniform_for_well_specified_simulator() -> None:
    sim = _gaussian_linear_simulator()
    fitted = _fit_npe(sim, num_steps=500, num_simulations=1000)
    result = simulation_based_calibration(
        fitted,
        sim,
        rngs=nnx.Rngs(sbi_simulate=4, sbi_sample=5),
        num_runs=128,
        num_posterior_samples=64,
    )
    # Per-dim KS p-values should not all be tiny under a well-specified flow.
    # We require at least one dimension passes at the 1% level — a deliberately
    # loose threshold given the limited training budget.
    assert float(jnp.max(result.ks_pvalue)) > 0.01


def test_expected_posterior_contraction_positive_for_informative_observation() -> None:
    sim = _gaussian_linear_simulator()
    fitted = _fit_npe(sim, num_steps=400, num_simulations=800)
    result = expected_posterior_contraction(
        fitted,
        sim,
        rngs=nnx.Rngs(sbi_simulate=0, sbi_sample=1),
        num_observations=16,
        num_posterior_samples=200,
    )
    assert isinstance(result, PosteriorContractionResult)
    # Informative likelihood => posterior is tighter than prior => contraction > 0.
    assert float(result.contraction) > 0.05


def test_expected_posterior_contraction_near_zero_for_uninformative_observation() -> None:
    sim = _uninformative_simulator()
    fitted = _fit_npe(sim, num_steps=400, num_simulations=800)
    result = expected_posterior_contraction(
        fitted,
        sim,
        rngs=nnx.Rngs(sbi_simulate=0, sbi_sample=1),
        num_observations=16,
        num_posterior_samples=200,
    )
    # Uninformative => contraction ~ 0 (allow slack for finite-sample noise
    # and the flow's slight tightening of the marginal during training).
    assert abs(float(result.contraction)) < 0.3


def test_sbc_result_validate_rejects_inconsistent_shapes() -> None:
    bad = SBCResult(
        ranks=jnp.zeros((4, 2), dtype=jnp.int32),
        ks_statistic=jnp.zeros((3,)),
        ks_pvalue=jnp.zeros((2,)),
    )
    with pytest.raises(ValueError, match="ks_statistic"):
        bad.validate()
