"""JAX-native data-likelihood ports: Fenrir + DALTON.

Tests for :mod:`opifex.uncertainty.scientific._likelihoods`. The module
implements two log-likelihood combinators referenced by the
probabilistic-numerics catalogue:

* :func:`fenrir_data_loglik` — Tronarp et al, ICML 2022 (arXiv:2202.01287).
  Post-solve smoothing data likelihood. Sibling reference
  ``ProbNumDiffEq.jl/src/data_likelihoods/fenrir.jl:30-128``.
* :func:`dalton_data_loglik` — Wu et al, 2023 (arXiv:2306.05566). Three-
  term combinator ``data_ll + with_pn_ll - without_pn_ll``. Sibling
  reference ``ProbNumDiffEq.jl/src/data_likelihoods/dalton.jl:23-76``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.registry import UQCapability
from opifex.uncertainty.scientific._likelihoods import (
    dalton_data_loglik,
    fenrir_data_loglik,
)
from opifex.uncertainty.scientific.probabilistic_numerics import (
    DaltonAdapterSpec,
    FenrirAdapterSpec,
)
from opifex.uncertainty.statespace.kalman import kalman_log_likelihood


def _build_lg_ssm() -> dict[str, Any]:
    """Build a small linear-Gaussian state-space model for likelihood tests."""
    state_dim, obs_dim, num_steps = 2, 1, 5
    transition = jnp.array([[0.9, 0.1], [0.0, 0.9]])
    process_noise = jnp.eye(state_dim) * 0.1
    observation_matrix = jnp.array([[1.0, 0.0]])
    observation_cov = jnp.eye(obs_dim) * 0.5
    initial_mean = jnp.zeros(state_dim)
    initial_cov = jnp.eye(state_dim)

    transitions = jnp.broadcast_to(transition, (num_steps, state_dim, state_dim))
    process_noises = jnp.broadcast_to(process_noise, (num_steps, state_dim, state_dim))

    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (num_steps, obs_dim))

    return {
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "num_steps": num_steps,
        "transitions": transitions,
        "process_noises": process_noises,
        "observation_matrix": observation_matrix,
        "observation_cov": observation_cov,
        "initial_mean": initial_mean,
        "initial_cov": initial_cov,
        "data": data,
    }


def _unconditioned_filter(
    transitions: jax.Array,
    process_noises: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure-prediction filter (no measurement updates) over the grid."""

    def body(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        mean, cov = carry
        transition, process_noise = inputs
        new_mean = transition @ mean
        new_cov = transition @ cov @ transition.T + process_noise
        return (new_mean, new_cov), (new_mean, new_cov)

    _, (means, covs) = jax.lax.scan(
        body, (initial_mean, initial_cov), (transitions, process_noises)
    )
    return means, covs


# ---------------------------------------------------------------------------
# Fenrir backward-conditioning log-likelihood
# ---------------------------------------------------------------------------


def test_fenrir_full_data_matches_forward_kalman_marginal_loglik() -> None:
    """Fenrir backward conditioning equals the forward Kalman marginal log-likelihood.

    Bayes' chain rule identity: ``p(y_1, ..., y_N)`` decomposed forward as
    ``prod_k p(y_k | y_{0:k-1})`` (standard KF innovation log-densities)
    equals the same probability decomposed backward via Fenrir.
    """
    cfg = _build_lg_ssm()
    obs_covs = jnp.broadcast_to(
        cfg["observation_cov"], (cfg["num_steps"], cfg["obs_dim"], cfg["obs_dim"])
    )
    forward_ll = kalman_log_likelihood(
        transitions=cfg["transitions"],
        process_noises=cfg["process_noises"],
        observations=cfg["data"],
        observation_matrix=cfg["observation_matrix"],
        observation_covs=obs_covs,
        initial_mean=cfg["initial_mean"],
        initial_cov=cfg["initial_cov"],
    )

    means, covs = _unconditioned_filter(
        cfg["transitions"], cfg["process_noises"], cfg["initial_mean"], cfg["initial_cov"]
    )

    data_mask = jnp.ones(cfg["num_steps"], dtype=bool)
    fenrir_ll = fenrir_data_loglik(
        filter_means=means,
        filter_covs=covs,
        transitions=cfg["transitions"],
        process_noises=cfg["process_noises"],
        data=cfg["data"],
        data_mask=data_mask,
        observation_matrix=cfg["observation_matrix"],
        observation_cov=cfg["observation_cov"],
    )

    assert jnp.isfinite(fenrir_ll)
    assert jnp.allclose(fenrir_ll, forward_ll, atol=1e-5, rtol=1e-5)


def test_fenrir_single_data_point_at_end_equals_predictive_log_density() -> None:
    """One observation at the last step reduces to a single predictive log-density."""
    cfg = _build_lg_ssm()
    means, covs = _unconditioned_filter(
        cfg["transitions"], cfg["process_noises"], cfg["initial_mean"], cfg["initial_cov"]
    )
    data_mask = jnp.zeros(cfg["num_steps"], dtype=bool).at[-1].set(True)
    fenrir_ll = fenrir_data_loglik(
        filter_means=means,
        filter_covs=covs,
        transitions=cfg["transitions"],
        process_noises=cfg["process_noises"],
        data=cfg["data"],
        data_mask=data_mask,
        observation_matrix=cfg["observation_matrix"],
        observation_cov=cfg["observation_cov"],
    )

    m_end, p_end = means[-1], covs[-1]
    obs_mat = cfg["observation_matrix"]
    pred_mean = obs_mat @ m_end
    pred_cov = obs_mat @ p_end @ obs_mat.T + cfg["observation_cov"]
    expected_ll = jax.scipy.stats.multivariate_normal.logpdf(
        cfg["data"][-1], pred_mean, pred_cov
    )
    assert jnp.allclose(fenrir_ll, expected_ll, atol=1e-6)


def test_fenrir_with_all_false_mask_returns_zero_loglik() -> None:
    """No observations means no log-likelihood contribution."""
    cfg = _build_lg_ssm()
    means, covs = _unconditioned_filter(
        cfg["transitions"], cfg["process_noises"], cfg["initial_mean"], cfg["initial_cov"]
    )
    data_mask = jnp.zeros(cfg["num_steps"], dtype=bool)
    fenrir_ll = fenrir_data_loglik(
        filter_means=means,
        filter_covs=covs,
        transitions=cfg["transitions"],
        process_noises=cfg["process_noises"],
        data=cfg["data"],
        data_mask=data_mask,
        observation_matrix=cfg["observation_matrix"],
        observation_cov=cfg["observation_cov"],
    )
    assert jnp.allclose(fenrir_ll, jnp.asarray(0.0))


def test_fenrir_compiles_under_jit() -> None:
    """Fenrir must compile under ``jax.jit`` for hyperparameter learning."""
    cfg = _build_lg_ssm()
    means, covs = _unconditioned_filter(
        cfg["transitions"], cfg["process_noises"], cfg["initial_mean"], cfg["initial_cov"]
    )
    data_mask = jnp.ones(cfg["num_steps"], dtype=bool)
    jitted = jax.jit(fenrir_data_loglik)
    ll = jitted(
        filter_means=means,
        filter_covs=covs,
        transitions=cfg["transitions"],
        process_noises=cfg["process_noises"],
        data=cfg["data"],
        data_mask=data_mask,
        observation_matrix=cfg["observation_matrix"],
        observation_cov=cfg["observation_cov"],
    )
    assert jnp.isfinite(ll)


def test_fenrir_gradient_with_respect_to_observation_cov_is_finite() -> None:
    """Fenrir must be differentiable w.r.t. ``observation_cov`` for learning."""
    cfg = _build_lg_ssm()
    means, covs = _unconditioned_filter(
        cfg["transitions"], cfg["process_noises"], cfg["initial_mean"], cfg["initial_cov"]
    )
    data_mask = jnp.ones(cfg["num_steps"], dtype=bool)

    def loss(log_obs_scale: jax.Array) -> jax.Array:
        return fenrir_data_loglik(
            filter_means=means,
            filter_covs=covs,
            transitions=cfg["transitions"],
            process_noises=cfg["process_noises"],
            data=cfg["data"],
            data_mask=data_mask,
            observation_matrix=cfg["observation_matrix"],
            observation_cov=jnp.exp(log_obs_scale) * jnp.eye(cfg["obs_dim"]),
        )

    grad = jax.grad(loss)(jnp.asarray(0.0))
    assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# DALTON three-term combinator
# ---------------------------------------------------------------------------


def test_dalton_formula_combines_three_loglikelihoods_linearly() -> None:
    """DALTON formula: ``data_ll + with_pn_ll - without_pn_ll``."""
    result = dalton_data_loglik(
        data_ll=jnp.asarray(5.0),
        with_pn_ll=jnp.asarray(3.0),
        without_pn_ll=jnp.asarray(2.0),
    )
    assert jnp.allclose(result, 6.0)


def test_dalton_reduces_to_data_ll_when_pn_likelihoods_cancel() -> None:
    """If PN likelihoods are equal (no data-adaptation effect), DALTON = ``data_ll``."""
    data_ll = jnp.asarray(5.0)
    pn = jnp.asarray(7.5)
    result = dalton_data_loglik(data_ll=data_ll, with_pn_ll=pn, without_pn_ll=pn)
    assert jnp.allclose(result, data_ll)


def test_dalton_compiles_under_jit() -> None:
    """DALTON must compile under ``jax.jit``."""
    jitted = jax.jit(dalton_data_loglik)
    result = jitted(
        data_ll=jnp.asarray(5.0),
        with_pn_ll=jnp.asarray(3.0),
        without_pn_ll=jnp.asarray(2.0),
    )
    assert jnp.allclose(result, 6.0)


def test_dalton_gradient_is_well_defined() -> None:
    """DALTON gradient w.r.t. inputs follows the linear-combination identity."""
    grad_data_ll = jax.grad(dalton_data_loglik, argnums=0)(
        jnp.asarray(5.0), jnp.asarray(3.0), jnp.asarray(2.0)
    )
    grad_without_pn = jax.grad(dalton_data_loglik, argnums=2)(
        jnp.asarray(5.0), jnp.asarray(3.0), jnp.asarray(2.0)
    )
    assert jnp.allclose(grad_data_ll, 1.0)
    assert jnp.allclose(grad_without_pn, -1.0)


# ---------------------------------------------------------------------------
# Adapter-spec wrap() concretization
# ---------------------------------------------------------------------------


def test_fenrir_adapter_spec_wrap_returns_fenrir_callable() -> None:
    """``FenrirAdapterSpec.wrap`` returns the Fenrir likelihood function."""
    spec = FenrirAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is fenrir_data_loglik


def test_dalton_adapter_spec_wrap_returns_dalton_callable() -> None:
    """``DaltonAdapterSpec.wrap`` returns the DALTON combinator."""
    spec = DaltonAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is dalton_data_loglik
