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
# Bayesian quadrature evidence: VanillaBQ vs WSABI-L vs Monte Carlo

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX |

## Overview

Estimate the model evidence ``Z = ∫ exp(-x²/2) p(x) dx`` against the
standard normal measure on R. The closed-form value is

    Z* = ∫ exp(-x²/2) N(x; 0, 1) dx = 1 / sqrt(2)  ≈ 0.7071067812

Three estimators at a *matched* point budget:

* ``vanilla_bayesian_quadrature`` — GP posterior mean + variance with an
  RBF kernel under the Gaussian measure.
* ``wsabi_l_bayesian_quadrature`` — WSABI-L bounded-integrand BQ that
  models ``f(x) = α + 0.5 g(x)²`` with a GP on ``g``.
* ``bayesian_monte_carlo`` — plain Monte Carlo with the same Gaussian
  samples used to seed BQ.

Pure JAX, no NNX state.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp

from opifex.uncertainty.quadrature.bayesian_monte_carlo import bayesian_monte_carlo
from opifex.uncertainty.quadrature.bayesian_quadrature import (
    vanilla_bayesian_quadrature,
    wsabi_l_bayesian_quadrature,
)


# %% [markdown]
"""
## Integrand and ground truth
"""


# %%
def integrand_fn(point: jax.Array) -> jax.Array:
    """``f(x) = exp(-x²/2)`` mapped on shape ``(d,)`` -> scalar."""
    return jnp.exp(-0.5 * jnp.sum(point**2))


GROUND_TRUTH = 1.0 / jnp.sqrt(2.0)


# %% [markdown]
"""
## Run the three estimators at matched budget
"""


# %%
def main() -> dict[str, jax.Array | float]:
    """Compare VanillaBQ + WSABI-L vs MC against the closed-form Z."""
    rng_key = jax.random.PRNGKey(0)
    num_points = 16

    sample_key = rng_key
    measure_mean = jnp.zeros(1)
    measure_variance = jnp.ones(1)
    kernel_lengthscales = jnp.full(1, 1.0)
    kernel_amplitude = jnp.asarray(1.0)

    samples = (
        jax.random.normal(sample_key, (num_points, 1)) * jnp.sqrt(measure_variance) + measure_mean
    )
    values = jax.vmap(integrand_fn)(samples)

    vanilla_jit = jax.jit(vanilla_bayesian_quadrature)
    vanilla_mean, vanilla_variance = vanilla_jit(
        points=samples,
        values=values,
        measure_mean=measure_mean,
        measure_variance=measure_variance,
        kernel_lengthscales=kernel_lengthscales,
        kernel_amplitude=kernel_amplitude,
    )

    wsabi_jit = jax.jit(wsabi_l_bayesian_quadrature)
    wsabi_mean = wsabi_jit(
        points=samples,
        values=values,
        offset=jnp.asarray(0.0),
        measure_mean=measure_mean,
        measure_variance=measure_variance,
        kernel_lengthscales=kernel_lengthscales,
        kernel_amplitude=kernel_amplitude,
    )

    monte_carlo_estimate = bayesian_monte_carlo(integrand=integrand_fn, samples=samples)

    vanilla_absolute_error = jnp.abs(vanilla_mean - GROUND_TRUTH)
    wsabi_absolute_error = jnp.abs(wsabi_mean - GROUND_TRUTH)
    monte_carlo_absolute_error = jnp.abs(monte_carlo_estimate.mean - GROUND_TRUTH)

    return {
        "ground_truth": float(GROUND_TRUTH),
        "vanilla_mean": vanilla_mean,
        "vanilla_variance": vanilla_variance,
        "vanilla_absolute_error": vanilla_absolute_error,
        "wsabi_mean": wsabi_mean,
        "wsabi_absolute_error": wsabi_absolute_error,
        "monte_carlo_mean": monte_carlo_estimate.mean,
        "monte_carlo_variance": monte_carlo_estimate.variance,
        "monte_carlo_absolute_error": monte_carlo_absolute_error,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for label, value in summary.items():
        print(f"{label}: {value}")
