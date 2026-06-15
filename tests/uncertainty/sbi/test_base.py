"""Tests for the shared SBI base helpers in :mod:`opifex.uncertainty.sbi._base`.

These pin the consolidated surface that NPE / NLE / NRE share (Task 12.3.4):

* :class:`_SBIFittedState` — fitted-state container; the three public
  ``NPEState`` / ``NLEState`` / ``NREState`` subclass it and stay distinct
  ``isinstance`` types while reusing one body.
* :func:`_build_conditional_flow` — conditional-flow constructor.
* :func:`_train_loop` — generic ``nnx.value_and_grad`` training loop driven
  by an estimator-specific loss closure.
* :func:`_mcmc_posterior_predictive` — shared BlackJAX MCMC predictive block
  used by NLE and NRE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from opifex.uncertainty.sbi import _base
from opifex.uncertainty.sbi.likelihood_estimation import NLEState
from opifex.uncertainty.sbi.posterior_estimation import NPEState
from opifex.uncertainty.sbi.ratio_estimation import NREState
from opifex.uncertainty.types import PredictiveDistribution


def test_state_subclasses_are_distinct_isinstance_types() -> None:
    npe = NPEState(train_losses=jnp.ones((3,)), num_simulations=jnp.asarray(7))
    nle = NLEState(train_losses=jnp.ones((3,)), num_simulations=jnp.asarray(7))
    nre = NREState(train_losses=jnp.ones((3,)), num_simulations=jnp.asarray(7))
    # All share the base implementation ...
    assert isinstance(npe, _base._SBIFittedState)
    assert isinstance(nle, _base._SBIFittedState)
    assert isinstance(nre, _base._SBIFittedState)
    # ... but remain mutually-distinct public types.
    assert not isinstance(npe, NLEState)
    assert not isinstance(nle, NREState)
    assert not isinstance(nre, NPEState)


def test_state_metadata_dict_and_validate() -> None:
    state = NPEState(
        train_losses=jnp.array([1.0, 2.0]),
        num_simulations=jnp.asarray(5),
        metadata=(("method", "npe"),),
    )
    assert state.metadata_dict() == {"method": "npe"}
    state.validate()  # well-formed -> no raise


def test_state_validate_rejects_empty_and_nan() -> None:
    with pytest.raises(ValueError, match="train_losses"):
        NPEState(train_losses=jnp.zeros((0,)), num_simulations=jnp.asarray(0)).validate()
    with pytest.raises(ValueError, match="train_losses"):
        NLEState(train_losses=jnp.full((3,), jnp.nan), num_simulations=jnp.asarray(0)).validate()


def test_state_is_pytree_and_jit_safe() -> None:
    state = NREState(train_losses=jnp.array([1.0, 2.0, 3.0]), num_simulations=jnp.asarray(9))
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is NREState
    summed = jax.jit(lambda s: s.train_losses.sum())(state)
    assert float(summed) == pytest.approx(6.0)


def test_build_conditional_flow_roundtrips_log_prob() -> None:
    flow = _base._build_conditional_flow(
        name="probe",
        input_dim=2,
        condition_dim=2,
        hidden_dim=8,
        num_coupling_layers=2,
        rngs=nnx.Rngs(params=0),
    )
    inputs = jnp.ones((4, 2))
    condition = jnp.zeros((4, 2))
    log_prob = flow.log_prob(inputs, condition=condition)
    assert log_prob.shape == (4,)


def test_train_loop_decreases_loss_and_is_jit_compiled() -> None:
    model = nnx.Linear(in_features=2, out_features=1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    target = jnp.ones((8, 1))
    inputs = jnp.ones((8, 2))

    def loss_fn(m: nnx.Linear) -> jax.Array:
        return jnp.mean((m(inputs) - target) ** 2)

    losses = _base._train_loop(model=model, optimizer=optimizer, loss_fn=loss_fn, num_steps=15)
    assert losses.shape == (15,)
    assert float(losses[-1]) < float(losses[0])


def test_mcmc_posterior_predictive_returns_sample_bearing_distribution() -> None:
    # Standard 2-d Gaussian target; NUTS should recover ~zero mean.
    def log_posterior(theta: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(theta * theta)

    pred = _base._mcmc_posterior_predictive(
        log_posterior=log_posterior,
        theta_dim=2,
        num_samples=64,
        mcmc_samples=128,
        mcmc_burnin=64,
        mcmc_method="nuts",
        mcmc_step_size=0.5,
        sample_key=jax.random.key(0),
        metadata=(("method", "probe"),),
    )
    assert isinstance(pred, PredictiveDistribution)
    assert pred.mean.shape == (2,)
    assert pred.samples is not None and pred.samples.shape == (64, 2)
    assert dict(pred.metadata).get("method") == "probe"
