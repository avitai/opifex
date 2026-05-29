r"""CAKF pluggable policy callable — Slice 24 (audit finding #6).

Task 6.3 design notes (``notes/04-task-6.3-expansion-design.md
:392-394``) require the CAKF update to ship pluggable policy
callables — CG (default; current behaviour), coordinate, and random
— ported from
``../ComputationAwareKalman.jl/src/filter/policy.jl:1-46``.

The opifex update step exposes a ``policy`` argument selecting one
of three strategies:

* ``CAKFPolicy.CG`` — search direction = current residual (default;
  CG-as-greedy).
* ``CAKFPolicy.COORDINATE`` — coordinate descent: direction =
  ``e_k`` for iteration ``k``.
* ``CAKFPolicy.RANDOM`` — direction = random standard-normal draw.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_cakf_policy_enum_has_three_named_strategies() -> None:
    """``CAKFPolicy`` carries the three named strategies from the design notes."""
    from opifex.uncertainty.statespace.cakf import CAKFPolicy

    assert CAKFPolicy.CG.value == "cg"
    assert CAKFPolicy.COORDINATE.value == "coordinate"
    assert CAKFPolicy.RANDOM.value == "random"


def _toy_update_inputs() -> dict:
    """Small linear-Gaussian observation problem for the CAKF update."""
    state_dim = 4
    obs_dim = 2
    mean = jnp.array([0.5, -0.2, 0.1, 0.0])
    factor = jnp.zeros((state_dim, 0))
    prior_cov = jnp.eye(state_dim) * 0.8
    observation = jnp.array([0.7, 0.1])
    observation_matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    observation_cov = 0.01 * jnp.eye(obs_dim)
    return {
        "mean": mean,
        "prior_cov": prior_cov,
        "factor": factor,
        "observation": observation,
        "observation_matrix": observation_matrix,
        "observation_cov": observation_cov,
        "max_iter": obs_dim,
    }


def test_cakf_update_default_policy_is_cg_and_preserves_prior_behaviour() -> None:
    """The default policy (CG) matches the unparameterised behaviour."""
    from opifex.uncertainty.statespace.cakf import cakf_update, CAKFPolicy

    inputs = _toy_update_inputs()
    default = cakf_update(**inputs)
    explicit_cg = cakf_update(**inputs, policy=CAKFPolicy.CG)
    assert jnp.allclose(default[0], explicit_cg[0], atol=1e-7)
    assert jnp.allclose(default[1], explicit_cg[1], atol=1e-7)


def test_cakf_update_coordinate_policy_returns_finite_state() -> None:
    """The coordinate-descent policy produces a finite ``(mean, factor)``."""
    from opifex.uncertainty.statespace.cakf import cakf_update, CAKFPolicy

    inputs = _toy_update_inputs()
    mean_new, factor_new = cakf_update(**inputs, policy=CAKFPolicy.COORDINATE)
    assert jnp.all(jnp.isfinite(mean_new))
    assert jnp.all(jnp.isfinite(factor_new))


def test_cakf_update_random_policy_uses_key_for_determinism() -> None:
    """Random policy with the same key returns identical results (reproducibility)."""
    from opifex.uncertainty.statespace.cakf import cakf_update, CAKFPolicy

    inputs = _toy_update_inputs()
    key = jax.random.PRNGKey(0)
    a_mean, a_factor = cakf_update(**inputs, policy=CAKFPolicy.RANDOM, key=key)
    b_mean, b_factor = cakf_update(**inputs, policy=CAKFPolicy.RANDOM, key=key)
    assert jnp.allclose(a_mean, b_mean)
    assert jnp.allclose(a_factor, b_factor)


def test_cakf_update_random_policy_changes_with_key() -> None:
    """Different keys explore different Krylov subspaces at sub-Krylov budget.

    With ``max_iter < obs_dim`` the random policy has not yet spanned
    the full Krylov subspace, so different PRNG keys produce different
    intermediate posteriors. (At ``max_iter == obs_dim`` all valid
    policies converge to the same exact posterior.)
    """
    from opifex.uncertainty.statespace.cakf import cakf_update, CAKFPolicy

    # Pick a 3-observation problem and only iterate once: random
    # policy then explores a 1-D Krylov subspace whose orientation
    # depends on the key.
    state_dim = 4
    mean = jnp.array([0.5, -0.2, 0.1, 0.0])
    factor = jnp.zeros((state_dim, 0))
    prior_cov = jnp.eye(state_dim) * 0.8
    observation = jnp.array([0.7, 0.1, 0.3])
    observation_matrix = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    observation_cov = 0.01 * jnp.eye(3)
    inputs = {
        "mean": mean,
        "prior_cov": prior_cov,
        "factor": factor,
        "observation": observation,
        "observation_matrix": observation_matrix,
        "observation_cov": observation_cov,
        "max_iter": 1,
    }
    a_mean, _ = cakf_update(**inputs, policy=CAKFPolicy.RANDOM, key=jax.random.PRNGKey(0))
    b_mean, _ = cakf_update(**inputs, policy=CAKFPolicy.RANDOM, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(a_mean, b_mean)
