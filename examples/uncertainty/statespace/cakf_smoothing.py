# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Compute-aware Kalman filtering on sparsely observed linear systems

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | JAX, linear-Gaussian state-space models |

## Overview

We compare the compute-aware Kalman filter (CAKF, Pförtner+ 2024,
arXiv:2405.08971) against the exact sequential Kalman filter on a small
linear-Gaussian state-space system with **sparse observations**.

CAKF runs at most ``max_iter`` CG sweeps per update step and propagates a
low-rank correction factor to the prior covariance. With ``max_iter``
equal to the observation dimension the CAKF posterior is the exact Kalman
posterior; here the observation is a scalar, so one sweep is enough.

The filter loop is :func:`opifex.uncertainty.statespace.cakf_filter`, a
port of ComputationAwareKalman.jl's ``filter!``: each step predicts,
updates and truncates the factor, carrying it to the next step.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import cakf_filter, kalman_filter


# %% [markdown]
"""
## Generate a small linear-Gaussian trajectory

State dimension 2 (position + velocity), scalar observation of position
only — a classic constant-velocity model.
"""


# %%
def _build_constant_velocity_model(
    num_steps: int, dt: float = 0.1
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(transitions, process_noises, H, R, initial_state)``."""
    transition = jnp.array([[1.0, dt], [0.0, 1.0]])
    process_noise_step = jnp.array([[1e-3, 0.0], [0.0, 1e-3]])
    observation_matrix = jnp.array([[1.0, 0.0]])
    observation_cov = jnp.array([[5e-2]])
    initial_state = jnp.array([0.0, 1.0])
    transitions = jnp.broadcast_to(transition, (num_steps, 2, 2))
    process_noises = jnp.broadcast_to(process_noise_step, (num_steps, 2, 2))
    return transitions, process_noises, observation_matrix, observation_cov, initial_state


# %%
def _simulate(
    rng_key: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Roll out one trajectory with Gaussian process + observation noise."""
    num_steps = transitions.shape[0]
    process_chol = jnp.linalg.cholesky(process_noises[0])
    observation_chol = jnp.linalg.cholesky(observation_cov)

    def step(state: jax.Array, inputs: tuple[jax.Array, jax.Array, jax.Array]):
        transition, process_noise_sample, observation_noise_sample = inputs
        next_state = transition @ state + process_chol @ process_noise_sample
        observation = observation_matrix @ next_state + observation_chol @ observation_noise_sample
        return next_state, (next_state, observation)

    process_key, observation_key = jax.random.split(rng_key)
    process_samples = jax.random.normal(process_key, (num_steps, 2))
    observation_samples = jax.random.normal(observation_key, (num_steps, 1))
    _, (states, observations) = jax.lax.scan(
        step, initial_state, (transitions, process_samples, observation_samples)
    )
    return states, observations


# %% [markdown]
"""
## Run the example
"""


# %%
def main() -> dict[str, jax.Array | float | int]:
    """Compare CAKF posterior mean against exact Kalman on sparse obs."""
    rng_key = jax.random.key(0)
    num_steps = 24
    (
        transitions,
        process_noises,
        observation_matrix,
        observation_cov_single,
        initial_state,
    ) = _build_constant_velocity_model(num_steps)
    states, observations = _simulate(
        rng_key,
        transitions,
        process_noises,
        observation_matrix,
        observation_cov_single,
        initial_state,
    )

    # Mask: keep only every third observation.
    observation_mask = jnp.arange(num_steps) % 3 == 0
    large_noise = jnp.eye(1) * 1e6
    observation_covs = jnp.where(
        observation_mask[:, None, None], observation_cov_single, large_noise
    )

    initial_mean = jnp.zeros(2)
    initial_cov = jnp.eye(2)
    exact_means, exact_covs = jax.jit(kalman_filter)(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    # CAKF sees the same masked observation noise as the exact filter.
    cakf_means, cakf_factors, prior_covs = jax.jit(cakf_filter, static_argnames=("max_iter",))(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        max_iter=1,
    )
    cakf_covs = prior_covs - cakf_factors @ jnp.swapaxes(cakf_factors, -1, -2)

    mean_relative = jnp.linalg.norm(cakf_means - exact_means) / jnp.linalg.norm(exact_means)
    cov_relative = jnp.linalg.norm(cakf_covs - exact_covs) / jnp.linalg.norm(exact_covs)
    cakf_vs_truth = jnp.mean(jnp.linalg.norm(cakf_means - states, axis=-1))
    exact_vs_truth = jnp.mean(jnp.linalg.norm(exact_means - states, axis=-1))

    summary = {
        "num_steps": int(num_steps),
        "observed_fraction": float(jnp.mean(observation_mask.astype(jnp.float32))),
        "max_iter": 1,
        "cakf_vs_exact_mean_relative": mean_relative,
        "cakf_vs_exact_cov_relative": cov_relative,
        "cakf_vs_truth_mean_l2": cakf_vs_truth,
        "exact_vs_truth_mean_l2": exact_vs_truth,
        "exact_final_trace": jnp.trace(exact_covs[-1]),
    }
    for name, value in summary.items():
        print(f"{name}: {value}")
    return summary


# %%
if __name__ == "__main__":
    main()
