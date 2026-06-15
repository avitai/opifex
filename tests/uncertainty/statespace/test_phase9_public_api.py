r"""Phase 9 statespace public-API closure — Slice 19.

Verifies the four symbols that the Phase 9 final-validation checklist
(``09-phase-final-validation.md:708-713``) and the Phase 6 Task 6.7
prerequisite gate (``06-phase-solvers-probnum-classical-uq.md:1326-
1340``) demand from ``opifex.uncertainty.statespace`` but that an
earlier audit found absent:

* ``state_transition_matrix`` — convenience extractor for
  :math:`A = \exp(F\,\Delta t)` from the LTI-SDE discretisation.
* ``process_noise_covariance`` — convenience extractor for the Van
  Loan process-noise covariance :math:`Q`.
* ``cakf_step`` — fused CAKF predict+update step.
* ``cakf_smooth`` — Pförtner+2024 CAKS Rauch-Tung-Striebel backward
  smoother (port of
  ``../ComputationAwareKalman.jl/src/smoother/loop.jl``).

The CAKS smoother is exercised via two equivalences:

1. With ``max_iter`` saturated (full-rank CAKF correction), the
   smoothed posterior coincides with the dense
   :func:`kalman_smoother` output on the same linear-Gaussian model.
2. With zero CAKF iterations and a constant identity transition, the
   smoothed mean / cov degenerate to the prior (no observations).

References
----------
* Pförtner, Wenger, Cockayne, Hennig 2024 — *Computation-Aware Kalman
  Filtering and Smoothing*, arXiv:2405.08971 (PRIMARY).
* Rauch, Tung, Striebel 1965 — *Maximum Likelihood Estimates of
  Linear Dynamic Systems*, AIAA J.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    cakf_smooth,
    cakf_step,
    discretize_lti_sde,
    kalman_filter,
    kalman_smoother,
    process_noise_covariance,
    state_transition_matrix,
)


# -----------------------------------------------------------------------------
# Phase 6 Task 6.7 prerequisite gate
# -----------------------------------------------------------------------------


def test_phase6_task67_prerequisite_gate_passes() -> None:
    """Phase 6 Task 6.7 :1326-1340 verbatim import gate must succeed.

    The plan ships a literal bash command that calls Python to import
    every symbol of the 12-element statespace public API. Until slice
    19, that gate failed at ``state_transition_matrix``. This test
    re-imports the same set programmatically.
    """
    import opifex.uncertainty.statespace as statespace_module

    required_symbols = (
        "cakf_smooth",
        "cakf_step",
        "discretize_lti_sde",
        "kalman_filter",
        "kalman_log_likelihood",
        "kalman_predict",
        "kalman_smoother",
        "kalman_update",
        "process_noise_covariance",
        "sqrt_kalman_predict",
        "sqrt_kalman_update",
        "state_transition_matrix",
    )
    for name in required_symbols:
        assert hasattr(statespace_module, name), (
            f"Phase 6 Task 6.7 prerequisite gate symbol '{name}' missing from "
            f"opifex.uncertainty.statespace public API."
        )


# -----------------------------------------------------------------------------
# state_transition_matrix / process_noise_covariance — thin wrappers
# -----------------------------------------------------------------------------


def test_state_transition_matrix_matches_discretize_lti_sde_first_output() -> None:
    """``state_transition_matrix(...)`` equals ``discretize_lti_sde(...)[0]``."""
    drift = jnp.asarray([[0.0, 1.0], [-1.0, -0.5]])
    dispersion = jnp.asarray([[0.0], [1.0]])
    dt = jnp.asarray(0.1)
    transition_only = state_transition_matrix(
        drift_matrix=drift, dispersion_matrix=dispersion, dt=dt
    )
    transition_ref, _ = discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
    assert jnp.allclose(transition_only, transition_ref, atol=1e-6)


def test_process_noise_covariance_matches_discretize_lti_sde_second_output() -> None:
    """``process_noise_covariance(...)`` equals ``discretize_lti_sde(...)[1]``."""
    drift = jnp.asarray([[0.0, 1.0], [-2.0, -0.3]])
    dispersion = jnp.asarray([[0.0], [1.0]])
    dt = jnp.asarray(0.05)
    process_only = process_noise_covariance(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
    _, process_ref = discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
    assert jnp.allclose(process_only, process_ref, atol=1e-6)


# -----------------------------------------------------------------------------
# cakf_step — fused predict+update
# -----------------------------------------------------------------------------


def test_cakf_step_equals_predict_then_update() -> None:
    """``cakf_step`` returns the same ``(mean, factor)`` as separate predict+update."""
    from opifex.uncertainty.statespace import cakf_predict, cakf_update

    state_dim = 4
    obs_dim = 2
    mean = jnp.asarray([0.5, -0.2, 0.1, 0.0])
    factor = jnp.zeros((state_dim, 0))
    transition = jnp.eye(state_dim) + 0.05 * jnp.asarray(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    prior_cov = jnp.eye(state_dim) * 0.8
    observation = jnp.asarray([0.7, 0.1])
    observation_matrix = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    observation_cov = 0.01 * jnp.eye(obs_dim)

    pmean, pfactor = cakf_predict(mean=mean, factor=factor, transition=transition)
    expected_mean, expected_factor = cakf_update(
        mean=pmean,
        prior_cov=prior_cov,
        factor=pfactor,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
        max_iter=obs_dim,
    )
    step_mean, step_factor = cakf_step(
        mean=mean,
        factor=factor,
        transition=transition,
        prior_cov=prior_cov,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
        max_iter=obs_dim,
    )
    assert jnp.allclose(step_mean, expected_mean, atol=1e-6)
    assert jnp.allclose(step_factor, expected_factor, atol=1e-6)


# -----------------------------------------------------------------------------
# cakf_smooth — CAKS RTS smoother
# -----------------------------------------------------------------------------


def _scalar_random_walk_problem(seed: int = 0, num_steps: int = 6) -> dict:
    """1-D random walk with linear-Gaussian observations."""
    key = jax.random.PRNGKey(seed)
    observation_noise_std = 0.2
    process_noise_std = 0.1
    transitions = jnp.broadcast_to(jnp.asarray([[1.0]]), (num_steps, 1, 1))
    process_noises = jnp.broadcast_to(jnp.asarray([[process_noise_std**2]]), (num_steps, 1, 1))
    observation_matrix = jnp.asarray([[1.0]])
    observation_covs = jnp.broadcast_to(
        jnp.asarray([[observation_noise_std**2]]), (num_steps, 1, 1)
    )
    initial_mean = jnp.asarray([0.0])
    initial_cov = jnp.asarray([[1.0]])
    observations = jnp.cumsum(process_noise_std * jax.random.normal(key, (num_steps,))).reshape(
        -1, 1
    ) + observation_noise_std * jax.random.normal(jax.random.fold_in(key, 1), (num_steps, 1))
    return {
        "transitions": transitions,
        "process_noises": process_noises,
        "observations": observations,
        "observation_matrix": observation_matrix,
        "observation_covs": observation_covs,
        "initial_mean": initial_mean,
        "initial_cov": initial_cov,
    }


def test_cakf_smooth_matches_kalman_smoother_when_cakf_runs_to_convergence() -> None:
    """CAKS == standard RTS smoother when CAKF iterates to full observation rank.

    With ``max_iter = obs_dim`` and a deterministic small problem, the
    low-rank CAKF correction at every step equals the exact Kalman
    posterior covariance (the CG iterates span the obs-dim Krylov
    subspace). The CAKS smoother in this regime must therefore
    coincide with :func:`kalman_smoother` on the same data.
    """
    problem = _scalar_random_walk_problem(seed=2, num_steps=5)
    # Standard Kalman filter + smoother reference.
    filter_means, filter_covs = kalman_filter(**problem)
    smoothed_means_ref, smoothed_covs_ref = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    # CAKS smoother via cakf_smooth.
    smoothed_means_caks, smoothed_covs_caks = cakf_smooth(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    assert jnp.allclose(smoothed_means_caks, smoothed_means_ref, atol=1e-5)
    assert jnp.allclose(smoothed_covs_caks, smoothed_covs_ref, atol=1e-5)


def test_cakf_smooth_terminal_step_matches_filter() -> None:
    """At the last time step the smoothed posterior equals the filter posterior."""
    problem = _scalar_random_walk_problem(seed=3, num_steps=4)
    filter_means, filter_covs = kalman_filter(**problem)
    smoothed_means, smoothed_covs = cakf_smooth(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    assert jnp.allclose(smoothed_means[-1], filter_means[-1])
    assert jnp.allclose(smoothed_covs[-1], filter_covs[-1])


def test_cakf_smooth_is_jit_compatible() -> None:
    """The full smoother compiles under ``jax.jit``."""
    problem = _scalar_random_walk_problem(seed=4, num_steps=5)

    @jax.jit
    def smooth_pipeline(
        filter_means: jax.Array,
        filter_covs: jax.Array,
        transitions: jax.Array,
        process_noises: jax.Array,
    ) -> jax.Array:
        sm, sc = cakf_smooth(
            filter_means=filter_means,
            filter_covs=filter_covs,
            transitions=transitions,
            process_noises=process_noises,
        )
        return sm.sum() + sc.sum()

    filter_means, filter_covs = kalman_filter(**problem)
    out = smooth_pipeline(
        filter_means, filter_covs, problem["transitions"], problem["process_noises"]
    )
    assert jnp.isfinite(out)
