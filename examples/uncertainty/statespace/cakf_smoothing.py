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

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, linear-Gaussian state-space models |

## Overview

We compare the compute-aware Kalman filter (CAKF, Pförtner+ 2024,
arXiv:2405.08971) against the exact sequential Kalman filter on a small
linear-Gaussian state-space system with **sparse observations**.

CAKF runs ``max_iter`` CG sweeps per update step and propagates a
low-rank correction factor to the prior covariance. As ``max_iter``
approaches the observation dimension, the CAKF posterior mean converges
to the exact Kalman posterior mean.

Pure JAX. The CAKF primitives — ``cakf_predict`` and ``cakf_update`` —
live in :mod:`opifex.uncertainty.statespace.cakf`. We use them
sequentially rather than via a higher-level ``cakf_smooth`` wrapper
(which is not part of the public surface in this slice).
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace.cakf import cakf_predict, cakf_update
from opifex.uncertainty.statespace.kalman import kalman_filter


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
## Sequential CAKF roll-out

`cakf_step` is not part of the slice's public API; we use
`cakf_predict` + `cakf_update` directly inside a `jax.lax.scan` loop.
"""


# %%
def _run_cakf(
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observations: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
    max_iter: int,
) -> tuple[jax.Array, jax.Array]:
    """Run a sequential CAKF filter and return posterior means + factors.

    The factor matrix grows by ``max_iter`` columns per ``cakf_update``
    step. We pre-allocate a *fixed-width* sliding window of ``max_iter``
    columns and discard older columns at the start of each step — this
    keeps the carry shapes static (a hard JAX requirement under
    ``jax.lax.scan``).
    """
    state_dim = initial_mean.shape[0]
    initial_factor = jnp.zeros((state_dim, max_iter))

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ):
        mean, factor, prior_cov = carry
        observation, transition, process_noise, observation_cov_step = inputs
        # Discard the previous-step columns to keep the carry shape fixed.
        empty_factor = jnp.zeros_like(factor[:, :0])
        predicted_mean, predicted_factor = cakf_predict(
            mean=mean, factor=empty_factor, transition=transition
        )
        predicted_cov = transition @ prior_cov @ transition.T + process_noise
        updated_mean, updated_factor = cakf_update(
            mean=predicted_mean,
            prior_cov=predicted_cov,
            factor=predicted_factor,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov_step,
            max_iter=max_iter,
        )
        return (updated_mean, updated_factor, predicted_cov), (updated_mean, updated_factor)

    observation_covs = jnp.broadcast_to(observation_cov, (observations.shape[0], 1, 1))
    _, (means, factors) = jax.lax.scan(
        step,
        (initial_mean, initial_factor, initial_cov),
        (observations, transitions, process_noises, observation_covs),
    )
    return means, factors


# %% [markdown]
"""
## Run the example
"""


# %%
def main() -> dict[str, jax.Array | float | int]:
    """Compare CAKF posterior mean against exact Kalman on sparse obs."""
    rng_key = jax.random.PRNGKey(0)
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

    cakf_jit = jax.jit(_run_cakf, static_argnames=("max_iter",))
    cakf_means, _ = cakf_jit(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov_single,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        max_iter=1,
    )

    cakf_vs_exact = jnp.mean(jnp.linalg.norm(cakf_means - exact_means, axis=-1))
    cakf_vs_truth = jnp.mean(jnp.linalg.norm(cakf_means - states, axis=-1))
    exact_vs_truth = jnp.mean(jnp.linalg.norm(exact_means - states, axis=-1))

    return {
        "num_steps": int(num_steps),
        "observed_fraction": float(jnp.mean(observation_mask.astype(jnp.float32))),
        "max_iter": 1,
        "cakf_vs_exact_mean_l2": cakf_vs_exact,
        "cakf_vs_truth_mean_l2": cakf_vs_truth,
        "exact_vs_truth_mean_l2": exact_vs_truth,
        "exact_final_trace": jnp.trace(exact_covs[-1]),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for label, value in summary.items():
        print(f"{label}: {value}")
