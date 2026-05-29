r"""Tests for ``active/constrained.py`` — Slice 23 (audit finding #4b).

Phase 8 Task 8.3 mandates Expected Constrained Improvement +
Expected Constrained Hypervolume Improvement (Gardner+ 2014; port of
``../trieste/acquisition/function/function.py:790`` and
``multi_objective.py:415``).

The constrained variants multiply the unconstrained acquisition (EI
or EHVI) by the probability that all constraints are satisfied at
the candidate. With ``g_k(x)`` denoting the ``k``-th constraint
(satisfied when ``g_k(x) ≤ 0``) under a Gaussian posterior:

    P(feasible | x) = Π_k Φ(-μ_{g_k}(x) / σ_{g_k}(x)),

so ECI(x) = EI(x) · P(feasible | x).

References
----------
* Gardner, Kusner, Xu, Weinberger, Cunningham 2014 — *Bayesian
  Optimization with Inequality Constraints*, ICML.
* Letham+ 2018 — *Constrained Bayesian Optimization with Noisy
  Experiments* (noisy-EHVI baseline).
"""

from __future__ import annotations

import jax.numpy as jnp


def test_probability_of_feasibility_collapses_to_one_when_constraints_are_easy() -> None:
    """A constraint with very negative mean has near-1 feasibility probability."""
    from opifex.uncertainty.active.constrained import probability_of_feasibility

    constraint_means = jnp.array([[-10.0, -8.0]])  # extremely satisfied
    constraint_stds = jnp.array([[1.0, 1.0]])
    prob = probability_of_feasibility(
        constraint_means=constraint_means, constraint_stds=constraint_stds
    )
    assert float(prob[0]) > 0.99


def test_probability_of_feasibility_collapses_to_zero_when_constraints_violated() -> None:
    """A constraint with very positive mean has near-0 feasibility probability."""
    from opifex.uncertainty.active.constrained import probability_of_feasibility

    constraint_means = jnp.array([[10.0]])
    constraint_stds = jnp.array([[1.0]])
    prob = probability_of_feasibility(
        constraint_means=constraint_means, constraint_stds=constraint_stds
    )
    assert float(prob[0]) < 0.01


def test_expected_constrained_improvement_equals_ei_times_pof() -> None:
    """ECI = EI · P(feasible) (Gardner+ 2014 §3)."""
    from opifex.uncertainty.active.constrained import expected_constrained_improvement

    ei_scores = jnp.array([0.5, 0.2, 0.3])
    constraint_means = jnp.array([[-2.0], [2.0], [-2.0]])
    constraint_stds = jnp.array([[0.5], [0.5], [0.5]])
    eci = expected_constrained_improvement(
        ei_scores=ei_scores,
        constraint_means=constraint_means,
        constraint_stds=constraint_stds,
    )
    # Index 1 has positive constraint mean (infeasible) → near-zero ECI.
    # Indices 0 and 2 are feasible → ECI ≈ EI.
    assert float(eci[1]) < float(eci[0]) * 0.01
    assert float(eci[0]) > 0.99 * float(ei_scores[0])
    assert float(eci[2]) > 0.99 * float(ei_scores[2])


def test_expected_constrained_hypervolume_improvement_equals_ehvi_times_pof() -> None:
    """ECHVI = EHVI · P(feasible) (Letham+ 2018; trieste multi_objective.py:415)."""
    from opifex.uncertainty.active.constrained import (
        expected_constrained_hypervolume_improvement,
    )

    ehvi_scores = jnp.array([0.4, 0.6, 0.5])
    constraint_means = jnp.array([[-1.0], [-1.0], [3.0]])
    constraint_stds = jnp.array([[0.5], [0.5], [0.5]])
    echvi = expected_constrained_hypervolume_improvement(
        ehvi_scores=ehvi_scores,
        constraint_means=constraint_means,
        constraint_stds=constraint_stds,
    )
    # Index 2 is infeasible → near-zero ECHVI.
    assert float(echvi[2]) < float(echvi[0]) * 0.01
