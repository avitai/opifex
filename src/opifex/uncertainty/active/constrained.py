r"""Constrained Bayesian optimisation — Slice 23 (audit finding #4b).

Ports ``../trieste/acquisition/function/function.py:790`` (Expected
Constrained Improvement) and
``../trieste/acquisition/function/multi_objective.py:415`` (Expected
Constrained Hypervolume Improvement).

Both variants multiply the unconstrained acquisition (EI or EHVI) by
the probability that all inequality constraints
``g_k(x) ≤ 0`` are satisfied at the candidate. Under independent
Gaussian-posterior constraints:

.. math::

    P(\text{feasible} \mid x)
        = \prod_{k=1}^{K} \Phi\!\left(
            -\frac{\mu_{g_k}(x)}{\sigma_{g_k}(x)}
          \right).

References
----------
* Gardner, Kusner, Xu, Weinberger, Cunningham 2014 — *Bayesian
  Optimization with Inequality Constraints*, ICML.
* Letham, Karrer, Ottoni, Bakshy 2018 — *Constrained Bayesian
  Optimization with Noisy Experiments*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


def probability_of_feasibility(
    *,
    constraint_means: jax.Array,
    constraint_stds: jax.Array,
) -> jax.Array:
    r"""Per-candidate ``P(all constraints satisfied)`` under independent Gaussians.

    Args:
        constraint_means: ``(N, K)`` posterior means of each
            constraint at each candidate (``g_k(x_i) ≤ 0`` is
            satisfied when ``μ_{g_k}(x_i)`` is sufficiently negative).
        constraint_stds: ``(N, K)`` posterior standard deviations.

    Returns:
        ``(N,)`` per-candidate feasibility probability.
    """
    z_scores = -constraint_means / jnp.maximum(constraint_stds, 1e-12)
    per_constraint = jnorm.cdf(z_scores)
    return jnp.prod(per_constraint, axis=-1)


def expected_constrained_improvement(
    *,
    ei_scores: jax.Array,
    constraint_means: jax.Array,
    constraint_stds: jax.Array,
) -> jax.Array:
    r"""Gardner+ 2014 ECI: ``EI(x) · P(feasible | x)``."""
    pof = probability_of_feasibility(
        constraint_means=constraint_means, constraint_stds=constraint_stds
    )
    return ei_scores * pof


def expected_constrained_hypervolume_improvement(
    *,
    ehvi_scores: jax.Array,
    constraint_means: jax.Array,
    constraint_stds: jax.Array,
) -> jax.Array:
    r"""Letham+ 2018 ECHVI: ``EHVI(x) · P(feasible | x)``."""
    pof = probability_of_feasibility(
        constraint_means=constraint_means, constraint_stds=constraint_stds
    )
    return ehvi_scores * pof


__all__ = [
    "expected_constrained_hypervolume_improvement",
    "expected_constrained_improvement",
    "probability_of_feasibility",
]
