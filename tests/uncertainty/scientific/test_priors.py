"""Tests for IWP / IOUP / Matern probabilistic-ODE prior SDE builders.

Each prior returns the continuous-time linear SDE ``(drift, dispersion)``
of a state-space prior on the unknown ODE solution. The drift matrix
encodes the order-``q+1`` integrator dynamics (companion-matrix
structure) and the dispersion matrix selects the noise channel.

Canonical references (PORTED, not imported):
* ``../probnum/src/probnum/randprocs/markov/integrator/_iwp.py`` — IWP.
* ``../probnum/src/probnum/randprocs/markov/integrator/_ioup.py`` — IOUP
  (scalar driftspeed).
* ``../probnum/src/probnum/randprocs/markov/integrator/_matern.py`` —
  Matérn-(q+1/2) prior.
* ``../ProbNumDiffEq.jl/src/priors/ioup.jl:103-117`` — IOUP three
  rate-parameter modes (scalar / vector / matrix).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific import (
    IOUPPriorSpec,
    IWPPriorSpec,
    MaternPriorSpec,
)


def test_iwp_spec_wrap_returns_sde_with_kronecker_structure() -> None:
    """IWP prior: drift is shift matrix; dispersion picks last derivative."""
    spec = IWPPriorSpec(num_derivatives=2, wiener_process_dimension=1)
    drift, dispersion = spec.build_sde()
    # Per-dim drift is shift matrix [[0,1,0],[0,0,1],[0,0,0]].
    expected_drift = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert jnp.allclose(drift, expected_drift, atol=1e-7)
    # Dispersion is e_{q+1} so the noise enters the highest derivative.
    expected_dispersion = jnp.asarray([[0.0], [0.0], [1.0]])
    assert jnp.allclose(dispersion, expected_dispersion, atol=1e-7)


def test_iwp_spec_with_multidimensional_state_uses_kronecker() -> None:
    """For ``wiener_process_dimension=d``, total state dim is ``d * (q+1)``."""
    spec = IWPPriorSpec(num_derivatives=1, wiener_process_dimension=3)
    drift, dispersion = spec.build_sde()
    assert drift.shape == (3 * 2, 3 * 2)
    assert dispersion.shape == (3 * 2, 3)


def test_ioup_spec_scalar_rate_sets_bottom_diagonal_to_rate_verbatim() -> None:
    """IOUP with scalar rate: ``drift[-1, -1] = rate`` per dim.

    Follows the Julia ``priors/ioup.jl:103-117`` convention where the
    user passes the drift block entry directly (already signed) rather
    than passing a positive ``driftspeed`` that the implementation then
    negates. This generalises uniformly to vector / matrix rates where
    the user supplies arbitrary drift blocks.
    """
    spec = IOUPPriorSpec(num_derivatives=1, wiener_process_dimension=1, rate_parameter=-2.0)
    drift, _ = spec.build_sde()
    expected = jnp.asarray([[0.0, 1.0], [0.0, -2.0]])
    assert jnp.allclose(drift, expected, atol=1e-7)


def test_ioup_spec_vector_rate_uses_diagonal_rate_in_bottom_block() -> None:
    """IOUP with vector rate: each output dim gets its own scalar rate."""
    spec = IOUPPriorSpec(
        num_derivatives=0,
        wiener_process_dimension=3,
        rate_parameter=jnp.asarray([-1.0, -2.0, -3.0]),
    )
    drift, _ = spec.build_sde()
    # state_dim = 3 * 1 = 3; drift = diag(-1, -2, -3) for q=0.
    expected = jnp.diag(jnp.asarray([-1.0, -2.0, -3.0]))
    assert jnp.allclose(drift, expected, atol=1e-7)


def test_ioup_spec_matrix_rate_couples_output_dimensions() -> None:
    """IOUP with matrix rate: bottom-right block encodes inter-dim coupling."""
    rate_matrix = jnp.asarray([[-1.0, 0.5], [0.5, -2.0]])
    spec = IOUPPriorSpec(
        num_derivatives=0,
        wiener_process_dimension=2,
        rate_parameter=rate_matrix,
    )
    drift, _ = spec.build_sde()
    assert jnp.allclose(drift, rate_matrix, atol=1e-7)


def test_ioup_spec_rate_mode_tagged_in_family_tags() -> None:
    """Constructor advertises which rate mode was selected."""
    scalar = IOUPPriorSpec(num_derivatives=1, wiener_process_dimension=1, rate_parameter=1.0)
    vector = IOUPPriorSpec(
        num_derivatives=1,
        wiener_process_dimension=2,
        rate_parameter=jnp.asarray([1.0, 2.0]),
    )
    matrix = IOUPPriorSpec(
        num_derivatives=1,
        wiener_process_dimension=2,
        rate_parameter=jnp.eye(2),
    )
    assert scalar.rate_mode == "scalar"
    assert vector.rate_mode == "vector"
    assert matrix.rate_mode == "matrix"


def test_matern_spec_drift_uses_binomial_coefficients() -> None:
    """Matérn-(q+1/2) drift bottom row: ``-binom(q+1, i) * lambda^(q+1-i)``."""
    spec = MaternPriorSpec(num_derivatives=1, wiener_process_dimension=1, lengthscale=1.0)
    drift, _ = spec.build_sde()
    # ν = num_derivatives + 0.5 = 1.5; λ = sqrt(3) / ℓ = sqrt(3).
    # Bottom row: [-binom(2, 0) * λ^2, -binom(2, 1) * λ^1] = [-3, -2 sqrt(3)].
    expected_bottom = jnp.asarray([-3.0, -2.0 * jnp.sqrt(3.0)])
    assert jnp.allclose(drift[-1, :], expected_bottom, atol=1e-5)


def test_matern_spec_matches_statespace_matern32_for_q_equals_1() -> None:
    """``MaternPriorSpec(q=1)`` equals the Matern-3/2 kernel SDE."""
    from opifex.uncertainty.statespace import matern32_kernel

    spec = MaternPriorSpec(num_derivatives=1, wiener_process_dimension=1, lengthscale=0.5)
    drift, _ = spec.build_sde()
    kernel = matern32_kernel(variance=1.0, lengthscale=0.5)
    # Matern-3/2 = MaternPriorSpec(q=1) up to the dispersion normalisation.
    assert jnp.allclose(drift, kernel.feedback, atol=1e-5)


def test_prior_specs_wrap_returns_concrete_lti_sde_handle() -> None:
    """``wrap`` is now concrete — returns the (drift, dispersion) SDE pair."""
    from opifex.uncertainty.registry import DefaultStrategy, UQCapability

    capability = UQCapability(default_strategy=DefaultStrategy.PROBABILISTIC_NUMERICS)
    spec = IWPPriorSpec(num_derivatives=1, wiener_process_dimension=1)
    drift, dispersion = spec.wrap(model=None, capability=capability)
    assert drift.shape == (2, 2)
    assert dispersion.shape == (2, 1)


def test_ioup_spec_rejects_mismatched_vector_rate_shape() -> None:
    """A vector rate length must equal ``wiener_process_dimension``."""
    with pytest.raises(ValueError, match="rate_parameter"):
        IOUPPriorSpec(
            num_derivatives=1,
            wiener_process_dimension=2,
            rate_parameter=jnp.asarray([1.0, 2.0, 3.0]),
        )
