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
# Fenrir vs DALTON likelihoods on a linear-drift ODE inverse problem

| Property | Value |
|---|---|
| **Level** | Advanced |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, linear Gaussian state-space models, ODE likelihoods |

## Overview

Reference paper:

* Tronarp, Bosch, Hennig 2022 — *Fenrir: Physics-Enhanced Regression
  for Initial Value Problems*, ICML, arXiv:2202.01287.
* Wu, Lange, Saumier, Cockayne 2023 — *Data-Adaptive Probabilistic
  Likelihood Approximation for ODEs*, arXiv:2306.05566.

We pose a small ODE inverse problem: estimate the decay rate ``θ`` in
``dy/dt = -θ y`` given noisy data. The closed-form solution is
``y(t) = exp(-θ t)``. We discretise the SDE for the latent state on a
uniform time grid, run a *linear-Gaussian* forward filter, then
evaluate two complementary log-likelihoods of the data:

* **Fenrir** — backward smoothing pass that conditions on the data only
  at the end (well-suited to *well-specified* observation noise).
* **DALTON** — three-term combinator
  ``data_ll + with_pn_ll - without_pn_ll`` (robust under
  *misspecified* observation noise).

Pure JAX, no NNX state — exercises the pure-array kernel path.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp

from opifex.uncertainty.scientific._likelihoods import dalton_data_loglik, fenrir_data_loglik
from opifex.uncertainty.statespace.kalman import kalman_filter, kalman_log_likelihood


# %% [markdown]
"""
## Build a discrete linear-Gaussian model approximating ``dy/dt = -θ y``

Single-state representation (``y`` only). The transition matrix at
step size ``dt`` is ``A = exp(-θ dt)`` and the process noise is a
small fixed jitter so the filter remains numerically well-conditioned.
"""


# %%
def _build_decay_model(
    *,
    theta: float,
    dt: float,
    num_steps: int,
    process_noise: float,
    observation_noise: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    transition = jnp.array([[jnp.exp(-theta * dt)]])
    transitions = jnp.broadcast_to(transition, (num_steps, 1, 1))
    process_noises = jnp.full((num_steps, 1, 1), process_noise)
    observation_matrix = jnp.eye(1)
    observation_covs = jnp.full((num_steps, 1, 1), observation_noise)
    return transitions, process_noises, observation_matrix, observation_covs


# %%
def _simulate_observations(
    rng_key: jax.Array,
    *,
    theta: float,
    dt: float,
    num_steps: int,
    observation_noise: float,
) -> jax.Array:
    """Generate noisy observations of ``y(t) = exp(-θ t)``."""
    times = dt * jnp.arange(1, num_steps + 1)
    truth = jnp.exp(-theta * times)
    noise = jax.random.normal(rng_key, (num_steps,)) * jnp.sqrt(observation_noise)
    return (truth + noise)[:, None]


# %% [markdown]
"""
## Score one ``θ`` candidate with both likelihoods
"""


# %%
def _score(
    *,
    theta: float,
    observations: jax.Array,
    dt: float,
    assumed_observation_noise: float,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(fenrir_loglik, dalton_loglik)`` for one candidate ``θ``."""
    num_steps = observations.shape[0]
    transitions, process_noises, observation_matrix, observation_covs = _build_decay_model(
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        process_noise=1e-6,
        observation_noise=assumed_observation_noise,
    )
    initial_mean = jnp.array([1.0])
    initial_cov = jnp.eye(1) * 1e-6

    # Unconditioned forward filter — needed for Fenrir's backward sweep.
    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=jnp.zeros_like(observations),
        observation_matrix=observation_matrix,
        observation_covs=observation_covs * 1e12,  # effectively no obs.
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    data_mask = jnp.ones(num_steps, dtype=bool)
    fenrir_loglik = fenrir_data_loglik(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=transitions,
        process_noises=process_noises,
        data=observations,
        data_mask=data_mask,
        observation_matrix=observation_matrix,
        observation_cov=observation_covs[0],
    )

    # DALTON ingredients — log-likelihood of a data-conditioned forward
    # pass plus the differential in solver PN likelihoods.
    with_pn_ll = kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    without_pn_ll = kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=jnp.zeros_like(observations),
        observation_matrix=observation_matrix,
        observation_covs=observation_covs * 1e12,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    data_ll = with_pn_ll
    dalton_loglik = dalton_data_loglik(data_ll, with_pn_ll, without_pn_ll)
    return fenrir_loglik, dalton_loglik


# %% [markdown]
"""
## Run the example: well-specified vs misspecified noise regime
"""


# %%
def main() -> dict[str, jax.Array | float]:
    """Score Fenrir and DALTON likelihoods in well-specified and misspecified regimes."""
    rng_key = jax.random.PRNGKey(0)
    dt = 0.1
    num_steps = 20
    true_theta = 0.5
    true_observation_noise = 1e-2

    observations = _simulate_observations(
        rng_key,
        theta=true_theta,
        dt=dt,
        num_steps=num_steps,
        observation_noise=true_observation_noise,
    )

    score_jit = jax.jit(_score, static_argnames=())
    well_specified_fenrir, well_specified_dalton = score_jit(
        theta=true_theta,
        observations=observations,
        dt=dt,
        assumed_observation_noise=true_observation_noise,
    )
    # Underestimate observation noise by 100x: misspecified regime.
    misspecified_fenrir, misspecified_dalton = score_jit(
        theta=true_theta,
        observations=observations,
        dt=dt,
        assumed_observation_noise=true_observation_noise * 1e-2,
    )

    return {
        "true_theta": float(true_theta),
        "well_specified_fenrir_loglik": well_specified_fenrir,
        "well_specified_dalton_loglik": well_specified_dalton,
        "misspecified_fenrir_loglik": misspecified_fenrir,
        "misspecified_dalton_loglik": misspecified_dalton,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for label, value in summary.items():
        print(f"{label}: {value}")
